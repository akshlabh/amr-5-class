"""
evaluate_attention.py — Standalone evaluation for MCLDNN-Attention experiment
==============================================================================
Usage
-----
    python src/evaluate_attention.py \\
        --datasetpath /kaggle/input/rml201610a-dict/RML2016.10a_dict.dat \\
        --attn_weights experiments/5class_attention/checkpoints/best_model.weights.h5 \\
        --baseline_weights experiments/5class_baseline/checkpoints/best_model.weights.h5

What this script does
---------------------
1. Load attention model and baseline MCLDNN weights.
2. Load test data with snr_range=(-6,6) and shuffle_split=True + seed=2016
   — identical parameters to train.py, so test split is the same.
3. Per-SNR accuracy comparison: attention model vs. baseline (filtered to
   the same -6..+6 dB SNR range for a fair comparison).
4. Save comparison.csv to experiments/5class_attention/.
5. Extract and save attention weights (.npy) per SNR level.
6. Plot attention temporal profiles (fill_between style, same as SHAP Task 2).
7. Print summary table: SNR | Baseline Acc | Attention Acc | Delta | Attn PAR.
"""

import os
import sys

# ── Repo-root path fix ────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import csv
import numpy as np

import keras
keras.mixed_precision.set_global_policy('float32')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.seed import set_all_seeds
set_all_seeds(2016)

from src.dataset                  import load_data, FIVE_CLASS
from src.models.mcldnn_attention  import (build_mcldnn_attention,
                                          build_mcldnn_attention_extractor)
from src.models.mcldnn            import MCLDNN
from src.utils.mltools            import calculate_confusion_matrix
from src.utils.attention_analysis import (
    extract_attention_weights,
    mean_attention_profile,
    attention_par
)


SNR_RANGE   = (-6, 6)
SELECTED_SNRS = [-6, -4, -2, 0, 2, 4, 6]
CLASSES     = FIVE_CLASS
BATCH_SIZE  = 400
N_ATTN      = 50   # samples per SNR for attention extraction


def _parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate and compare MCLDNN-Attention vs MCLDNN-Baseline'
    )
    p.add_argument('--datasetpath', required=True,
                   help='Path to RML2016.10a_dict.dat / .pkl')
    p.add_argument('--attn_weights', required=True,
                   help='Path to attention model .weights.h5 checkpoint')
    p.add_argument('--baseline_weights', required=True,
                   help='Path to baseline MCLDNN .weights.h5 checkpoint')
    p.add_argument('--out_dir', default='experiments/5class_attention',
                   help='Output directory for results, figures, npy files')
    return p.parse_args()


def _prepare_inputs(X: np.ndarray):
    """Return the three-input list expected by both MCLDNN variants."""
    return [
        np.expand_dims(X, axis=3).astype('float32'),           # (N, 2, 128, 1)
        np.expand_dims(X[:, 0, :], axis=2).astype('float32'),  # (N, 128, 1)
        np.expand_dims(X[:, 1, :], axis=2).astype('float32'),  # (N, 128, 1)
    ]


def main():
    args = _parse_args()

    out_dir  = args.out_dir
    fig_dir  = os.path.join(out_dir, 'figures')
    res_dir  = out_dir
    attn_dir = out_dir
    for d in [fig_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Load test data (same params as training) ──────────────────────────────
    print("[eval_attention] Loading test data ...")
    (mods, snrs, lbl), _, _, (X_test, Y_test), (_, _, test_idx) = load_data(
        args.datasetpath,
        selected_classes=CLASSES,
        seed=2016,
        snr_range=SNR_RANGE,
        shuffle_split=True
    )
    n_classes = len(mods)
    test_SNRs = np.array([lbl[i][1] for i in test_idx])
    inp_test  = _prepare_inputs(X_test)
    Y_true    = np.argmax(Y_test, axis=1)
    print(f"[eval_attention] Test set: {X_test.shape[0]} samples, "
          f"SNR range: {snrs[0]}..{snrs[-1]} dB")

    # ── Build and load attention model (single-output for accuracy eval) ──────────
    print("\n[eval_attention] Building attention model ...")
    attn_model = build_mcldnn_attention(classes=n_classes)
    if not os.path.exists(args.attn_weights):
        print(f"[eval_attention] WARNING: Attention weights not found: "
              f"{args.attn_weights}")
        print("[eval_attention] Running with random weights for shape verification.")
    else:
        attn_model.load_weights(args.attn_weights)
        print(f"[eval_attention] Loaded attention weights: {args.attn_weights}")

    # Build extractor model (dual-output) for attention weight extraction
    print("[eval_attention] Building attention extractor model ...")
    extractor = build_mcldnn_attention_extractor(
        classes=n_classes,
        weights_path=args.attn_weights if os.path.exists(args.attn_weights) else None
    )

    # ── Build and load baseline model ─────────────────────────────────────────
    print("\n[eval_attention] Building baseline MCLDNN ...")
    baseline_model = MCLDNN(classes=n_classes)
    baseline_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(clipnorm=1.0),
        metrics=['accuracy']
    )
    if not os.path.exists(args.baseline_weights):
        print(f"[eval_attention] WARNING: Baseline weights not found: "
              f"{args.baseline_weights}")
    else:
        baseline_model.load_weights(args.baseline_weights)
        print(f"[eval_attention] Loaded baseline weights: {args.baseline_weights}")

    # ── Per-SNR accuracy comparison ───────────────────────────────────────────
    print("\n[eval_attention] Computing per-SNR accuracies ...")
    rows = []
    for snr in snrs:
        mask   = test_SNRs == snr
        if mask.sum() == 0:
            continue
        X_snr  = [b[mask] for b in inp_test]
        Y_snr  = Y_test[mask]

        # Attention model: single output → plain ndarray
        pred_attn_prob = attn_model.predict(X_snr, verbose=0,
                                            batch_size=BATCH_SIZE)
        _, cor_a, ncor_a = calculate_confusion_matrix(Y_snr, pred_attn_prob, mods)
        acc_attn = cor_a / (cor_a + ncor_a)

        # Baseline model: single output
        pred_base_prob = baseline_model.predict(X_snr, verbose=0,
                                                batch_size=BATCH_SIZE)
        _, cor_b, ncor_b = calculate_confusion_matrix(Y_snr, pred_base_prob, mods)
        acc_base = cor_b / (cor_b + ncor_b)

        delta = acc_attn - acc_base
        rows.append({'snr': snr, 'acc_baseline': acc_base,
                     'acc_attention': acc_attn, 'delta': delta})
        print(f"  SNR={snr:+3d} dB | Base={acc_base:.3f} | "
              f"Attn={acc_attn:.3f} | Delta={delta:+.3f}")

    # Save comparison CSV
    comp_path = os.path.join(res_dir, 'comparison.csv')
    with open(comp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['snr', 'acc_baseline',
                                               'acc_attention', 'delta'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval_attention] Saved comparison CSV: {comp_path}")

    # ── Extract attention weights per SNR (using extractor model) ─────────────────
    print("\n[eval_attention] Extracting attention weights per SNR ...")
    # Use single-output model for correct/incorrect classification
    Y_pred_all    = attn_model.predict(inp_test, batch_size=BATCH_SIZE, verbose=0)
    Y_pred_cls    = np.argmax(Y_pred_all, axis=1)
    correct_mask  = (Y_pred_cls == Y_true)

    attn_par_per_snr = {}
    for snr in SELECTED_SNRS:
        if snr not in snrs:
            print(f"  SNR={snr} dB not in loaded data, skipping")
            continue
        mask = np.where((test_SNRs == snr) & correct_mask)[0]
        if len(mask) == 0:
            print(f"  SNR={snr:+d} dB: no correctly classified samples, skipping")
            continue
        chosen = mask[:N_ATTN] if len(mask) >= N_ATTN else mask
        X_chosen = [b[chosen] for b in inp_test]

        # Use the extractor model (dual-output) to get attn_weights
        attn_w = extract_attention_weights(extractor, X_chosen,
                                           n_samples=len(chosen))
        # attn_w: (N, 4, 124, 124)

        npy_path = os.path.join(attn_dir,
                                f'attn_weights_snr{snr:+d}.npy')
        np.save(npy_path, attn_w)
        print(f"  SNR={snr:+d} dB | {len(chosen)} samples | saved → {npy_path}")

        par_vals = attention_par(attn_w)  # (N,)
        attn_par_per_snr[snr] = par_vals.mean()

    # ── Plot attention temporal profiles per SNR ───────────────────────────────
    print("\n[eval_attention] Plotting attention profiles ...")
    TIME_STEPS = np.arange(124)
    colors = plt.cm.viridis(np.linspace(0, 1, len(SELECTED_SNRS)))

    for idx, snr in enumerate(SELECTED_SNRS):
        npy_path = os.path.join(attn_dir, f'attn_weights_snr{snr:+d}.npy')
        if not os.path.exists(npy_path):
            continue
        attn_w   = np.load(npy_path)                   # (N, 4, 124, 124)
        profiles  = mean_attention_profile(attn_w)      # (N, 124)
        mean_prof = profiles.mean(axis=0)               # (124,)
        std_prof  = profiles.std(axis=0)                # (124,)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(TIME_STEPS,
                        np.maximum(0, mean_prof - std_prof),
                        mean_prof + std_prof,
                        alpha=0.25, color=colors[idx])
        ax.plot(TIME_STEPS, mean_prof, color=colors[idx],
                linewidth=2.0, label=f'Mean attention (SNR={snr:+d} dB)')
        peak_t = np.argmax(mean_prof)
        ax.axvline(peak_t, color='black', linestyle='--',
                   alpha=0.6, linewidth=1.2)
        ax.text(peak_t + 2, mean_prof.max() * 0.9,
                f't={peak_t}', fontsize=9, color='black')
        ax.set_xlabel('Time step (0–123)', fontsize=12)
        ax.set_ylabel('Mean attention weight', fontsize=12)
        ax.set_title(f'MCLDNN-Attention: Temporal Profile  (SNR={snr:+d} dB)',
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(fig_dir, f'attn_profile_snr{snr:+d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {save_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'SNR':>6} | {'Baseline':>10} | {'Attention':>10} | "
          f"{'Delta':>8} | {'Attn PAR':>10}")
    print("  " + "─" * 55)
    for r in rows:
        snr   = r['snr']
        par   = attn_par_per_snr.get(snr, float('nan'))
        print(f"  {snr:>+6d} | {r['acc_baseline']:>10.3f} | "
              f"{r['acc_attention']:>10.3f} | {r['delta']:>+8.3f} | "
              f"{par:>10.3f}")
    print("=" * 70)
    print(f"\n[eval_attention] All outputs saved under {out_dir}/")


if __name__ == '__main__':
    main()
