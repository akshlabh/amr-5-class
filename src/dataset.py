"""
dataset.py — Class-filtered RML2016.10a data loader
====================================================
TF2 / Keras 3 compatible.  Drop-in replacement for the original dataset2016.py
with the key addition of a `selected_classes` parameter that restricts loading
to only the requested modulation types.

Verified class names from RML2016.10a_dict.pkl:
    ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK',
     'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

Predefined subsets
------------------
FIVE_CLASS  : ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']  — primary model
FOUR_CLASS  : ['QPSK', '8PSK', 'QAM16', 'QAM64']           — ablation (no BPSK)
ALL_CLASSES : all 11 — only for reference / full-dataset runs

Fixed-split support (split_file)
---------------------------------
When `split_file` is supplied to load_data():
  • If the file exists  → indices are loaded from it (fast, deterministic).
  • If the file is new  → indices are generated with `seed`, saved, then used.

This decouples the random seed for the DATA SPLIT from the seed used for
weight initialisation.  All four 4-class ablation models can then use
different `seed` values for model initialisation while still being evaluated
on an IDENTICAL test set — making accuracy comparisons scientifically valid.
"""

import os
import pickle
import numpy as np

# ── Class subsets ─────────────────────────────────────────────────────────────
ALL_CLASSES  = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
                'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

FIVE_CLASS   = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']   # primary
FOUR_CLASS   = ['QPSK', '8PSK', 'QAM16', 'QAM64']            # ablation


def normalize_samples(X: np.ndarray) -> np.ndarray:
    """
    Per-sample RMS power normalization  (hardware AGC equivalent).

    Each IQ segment is divided by its own RMS amplitude so that every sample
    enters the network with unit mean-square power.  This prevents LSTM gate
    saturation caused by the large dynamic range of RML2016.10a signals across
    SNR levels and modulation types.

    The original AMR-Benchmark code defined l2_normalize() in dataset2016.py
    but never called it; skipping normalization causes the LSTM to saturate
    (tanh/sigmoid output ≈ ±1, gradients ≈ 0) and the model stays at random
    chance (≈20 % for 5 classes) for the entire training run.

    Parameters
    ----------
    X : np.ndarray  shape (N, 2, 128)  raw IQ segments

    Returns
    -------
    np.ndarray  same shape, each sample scaled to RMS ≈ 1
    """
    # RMS computed jointly over I, Q and all 128 time-steps
    rms = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))  # (N,1,1)
    return X / (rms + 1e-10)


def load_data(filename: str,
              selected_classes: list = None,
              seed: int = 2016,
              split_file: str = None):
    """
    Load RML2016.10a and optionally filter to a subset of modulation classes.

    Parameters
    ----------
    filename         : str   Absolute path to RML2016.10a_dict.pkl
    selected_classes : list  List of class name strings to include.
                             If None, all 11 classes are loaded.
                             Order determines the one-hot encoding index.
                             Example: FIVE_CLASS  or  FOUR_CLASS
    seed             : int   NumPy random seed.
                             • When split_file is None (or new): controls the
                               train/val/test split generation.
                             • When split_file already exists: split indices are
                               loaded from file; seed only affects downstream
                               shuffle order (minor, cosmetic).
    split_file       : str or None
                             Path to a .npz file for a fixed train/val/test split.
                             • None (default) → split is generated fresh each run
                               using `seed`; behaviour is identical to the
                               original code.
                             • Path that exists → indices loaded from file
                               (all models share the same test set regardless
                               of their `seed` value).
                             • Path that does not exist yet → split generated
                               from `seed`, saved to this path for future runs.

    Returns
    -------
    (mods, snrs, lbl),
    (X_train, Y_train),
    (X_val,   Y_val),
    (X_test,  Y_test),
    (train_idx, val_idx, test_idx)
    """
    # ── Load pickle ───────────────────────────────────────────────────────────
    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

    all_mods = sorted(list(set(k[0] for k in Xd.keys())))
    snrs     = sorted(list(set(k[1] for k in Xd.keys())))

    # ── Apply class filter ────────────────────────────────────────────────────
    if selected_classes is not None:
        missing = set(selected_classes) - set(all_mods)
        if missing:
            raise ValueError(
                f"Requested classes not found in dataset: {missing}\n"
                f"Available classes: {all_mods}"
            )
        # Preserve the caller-specified order (determines one-hot indices)
        mods = [m for m in selected_classes if m in all_mods]
    else:
        mods = all_mods

    print(f"[dataset] Loading {len(mods)} classes: {mods}")
    print(f"[dataset] SNR range: {snrs[0]} dB to {snrs[-1]} dB  ({len(snrs)} levels)")

    # ── Build data arrays (no random calls here) ──────────────────────────────
    X, lbl = [], []
    for mod in mods:
        for snr in snrs:
            block = Xd[(mod, snr)]                      # shape: (1000, 2, 128)
            X.append(block)
            for _ in range(block.shape[0]):
                lbl.append((mod, snr))

    X = np.vstack(X)                                    # (N, 2, 128)

    # ── Post-load sanity checks ───────────────────────────────────────────────
    assert len(mods) == len(selected_classes if selected_classes is not None else all_mods), \
        (f"Class count mismatch: expected {len(selected_classes)}, "
         f"got {len(mods)}: {mods}")
    expected_n = len(mods) * len(snrs) * 1000
    assert X.shape[0] == expected_n, \
        (f"Sample count mismatch: expected {expected_n} "
         f"({len(mods)} classes × {len(snrs)} SNRs × 1000), got {X.shape[0]}")

    n_examples = X.shape[0]

    # ── Generate or load train/val/test split ─────────────────────────────────
    if split_file is not None and os.path.exists(split_file):
        # Fast path: load pre-generated fixed split
        _sp        = np.load(split_file)
        train_idx  = _sp['train_idx'].tolist()
        val_idx    = _sp['val_idx'].tolist()
        test_idx   = _sp['test_idx'].tolist()
        print(f"[dataset] Loaded fixed split from: {split_file}  "
              f"(train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)})")
        # Validate sizes match current dataset
        assert max(train_idx + val_idx + test_idx) < n_examples, \
            f"Split file index out of range for {n_examples}-sample dataset"
    else:
        # Generate split with seed (same algorithm as original code)
        np.random.seed(seed)
        train_idx, val_idx = [], []
        for _b in range(len(mods) * len(snrs)):
            _base   = _b * 1000
            _t_idx  = list(np.random.choice(
                range(_base, _base + 1000), size=600, replace=False))
            _v_pool = list(set(range(_base, _base + 1000)) - set(_t_idx))
            _v_idx  = list(np.random.choice(_v_pool, size=200, replace=False))
            train_idx.extend(_t_idx)
            val_idx.extend(_v_idx)
        test_idx = list(set(range(n_examples)) - set(train_idx) - set(val_idx))

        if split_file is not None:
            # Save for all future runs (auto-bootstrapping)
            _dir = os.path.dirname(os.path.abspath(split_file))
            os.makedirs(_dir, exist_ok=True)
            np.savez(split_file,
                     train_idx=np.array(train_idx, dtype=np.int32),
                     val_idx=np.array(val_idx,   dtype=np.int32),
                     test_idx=np.array(test_idx,  dtype=np.int32))
            print(f"[dataset] Saved fixed split to: {split_file}  "
                  f"(train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)})")

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # ── One-hot encoding ──────────────────────────────────────────────────────
    def to_onehot(indices):
        oh = np.zeros((len(indices), len(mods)), dtype=np.float32)
        oh[np.arange(len(indices)), indices] = 1.0
        return oh

    Y_train = to_onehot([mods.index(lbl[i][0]) for i in train_idx])
    Y_val   = to_onehot([mods.index(lbl[i][0]) for i in val_idx])
    Y_test  = to_onehot([mods.index(lbl[i][0]) for i in test_idx])

    X_train = normalize_samples(X[train_idx])
    X_val   = normalize_samples(X[val_idx])
    X_test  = normalize_samples(X[test_idx])

    # Sanity-check: report amplitude stats after normalization
    rms_train = np.sqrt(np.mean(X_train ** 2))
    print(f"[dataset] After normalization — train RMS: {rms_train:.4f}  "
          f"(should be ≈1.0)")

    print(f"[dataset] Split: {len(train_idx)} train | "
          f"{len(val_idx)} val | {len(test_idx)} test samples")

    return (
        (mods, snrs, lbl),
        (X_train, Y_train),
        (X_val,   Y_val),
        (X_test,  Y_test),
        (train_idx, val_idx, test_idx)
    )


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path/to/RML2016.10a_dict.pkl>")
        sys.exit(1)

    path = sys.argv[1]
    print("\n=== 5-class load ===")
    (mods, snrs, lbl), (Xtr, Ytr), (Xv, Yv), (Xte, Yte), _ = \
        load_data(path, FIVE_CLASS)
    print(f"X_train: {Xtr.shape}  Y_train: {Ytr.shape}")
    print(f"Classes: {mods}")

    print("\n=== 4-class load ===")
    (mods4, _, _), (Xtr4, Ytr4), _, _, _ = load_data(path, FOUR_CLASS)
    print(f"X_train: {Xtr4.shape}  Y_train: {Ytr4.shape}")
    print(f"Classes: {mods4}")
