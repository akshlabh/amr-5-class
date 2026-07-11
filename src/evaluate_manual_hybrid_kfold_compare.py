"""
evaluate_manual_hybrid_kfold_compare.py
======================================

Manual SNR-aware hybrid using k-fold trained attention models.

No alpha search. No alpha smoothing. No monotonic trend forcing.

Default rule:
    For each fold and each SNR <= low_snr_max, use the fold validation split
    to choose whichever model is better: normal attention or differential
    attention. For SNR > low_snr_max, use normal attention.

This mirrors the earlier non-kfold hybrid logic, but repeats it across folds.

This script compares:
    1. LSTM/MCLDNN baseline from the existing single-run baseline CSV
    2. Existing non-kfold manual hybrid from experiments/5class_hybrid_snr_aware_attention
    3. New k-fold manual hybrid mean ± std across folds
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["KERAS_BACKEND"] = "tensorflow"


FIVE_CLASS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare LSTM baseline, existing non-kfold hybrid, and k-fold manual hybrid."
    )
    p.add_argument("--datasetpath", required=True)
    p.add_argument(
        "--normal-kfold-dir",
        default="experiments/5class_attention_kfold/kfold",
    )
    p.add_argument(
        "--diff-kfold-dir",
        default="experiments/5class_diffattention_kfold/kfold",
    )
    p.add_argument(
        "--lstm-baseline-csv",
        default="experiments/5class_baseline/results/acc_per_snr.csv",
    )
    p.add_argument(
        "--nonkfold-hybrid-csv",
        default="experiments/5class_hybrid_snr_aware_attention/results/acc_per_snr.csv",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_manual_hybrid_kfold_compare",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--low-snr-max", type=int, default=2)
    p.add_argument(
        "--selection-mode",
        choices=("validation_best", "fixed_threshold"),
        default="validation_best",
        help=(
            "validation_best mirrors the earlier non-kfold hybrid: choose normal/diff "
            "per SNR using validation accuracy. fixed_threshold always uses diff for "
            "SNR <= low_snr_max."
        ),
    )
    p.add_argument(
        "--min-val-delta",
        type=float,
        default=0.0,
        help="In validation_best mode, diff must beat normal validation accuracy by this margin.",
    )
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--seed", type=int, default=2016)
    return p.parse_args()


def _onehot(class_indices: list[int], n_classes: int) -> np.ndarray:
    out = np.zeros((len(class_indices), n_classes), dtype=np.float32)
    out[np.arange(len(class_indices)), class_indices] = 1.0
    return out


def _accuracy(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    return float(np.mean(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)))


def _load_raw_dataset(datasetpath: str, classes: list[str]):
    Xd = pickle.load(open(datasetpath, "rb"), encoding="iso-8859-1")
    snrs = sorted(set(k[1] for k in Xd.keys()))
    X_raw, lbl = [], []
    for mod in classes:
        for snr in snrs:
            block = Xd[(mod, snr)]
            X_raw.append(block)
            for _ in range(block.shape[0]):
                lbl.append((mod, snr))
    return np.vstack(X_raw), lbl, snrs


def _load_acc_csv(path: str | Path, accuracy_col: str = "accuracy") -> dict[int, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    out: dict[int, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "snr" not in row:
                continue
            out[int(float(row["snr"]))] = float(row[accuracy_col])
    return out


def _load_existing_hybrid_csv(path: str | Path) -> dict[int, float]:
    return _load_acc_csv(path, accuracy_col="hybrid_accuracy")


def _require_fold_weights(kfold_dir: Path, fold: int) -> Path:
    path = kfold_dir / f"fold_{fold}" / "best_model.weights.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold weights: {path}\n"
            "Run src/train_kfold.py for the corresponding config first."
        )
    return path


def _build_normal_attention_model(n_classes: int):
    from src.models.mcldnn_attention import build_mcldnn_attention

    return build_mcldnn_attention(
        classes=n_classes,
        dropout_rate=0.6,
        learning_rate=0.0005,
    )


def _build_diff_attention_model(n_classes: int):
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention

    return build_mcldnn_diffattention(
        classes=n_classes,
        dropout_rate=0.5,
        learning_rate=0.0005,
    )


def _predict_fold_model(builder, weights: Path, inputs, batch_size: int):
    import keras

    model = builder()
    model.load_weights(weights)
    pred = model.predict(inputs, batch_size=batch_size, verbose=1)
    del model
    keras.backend.clear_session()
    return pred


def _save_fold_rows(rows: list[dict], path: Path):
    fields = [
        "fold",
        "snr",
        "selected_model",
        "normal_attention_accuracy",
        "diff_attention_accuracy",
        "manual_hybrid_kfold_accuracy",
        "normal_val_accuracy",
        "diff_val_accuracy",
        "diff_minus_normal_val",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_summary(rows: list[dict],
                  lstm_acc: dict[int, float],
                  nonkfold_hybrid_acc: dict[int, float],
                  path: Path) -> list[dict]:
    snrs = sorted({int(r["snr"]) for r in rows})
    summary = []
    for snr in snrs:
        vals = np.array(
            [float(r["manual_hybrid_kfold_accuracy"]) for r in rows if int(r["snr"]) == snr],
            dtype=np.float32,
        )
        row = {
            "snr": snr,
            "lstm_baseline_accuracy": lstm_acc.get(snr, ""),
            "nonkfold_manual_hybrid_accuracy": nonkfold_hybrid_acc.get(snr, ""),
            "kfold_manual_hybrid_mean": float(np.mean(vals)),
            "kfold_manual_hybrid_std": float(np.std(vals)),
            "kfold_minus_lstm_baseline": (
                float(np.mean(vals)) - lstm_acc[snr] if snr in lstm_acc else ""
            ),
            "kfold_minus_nonkfold_hybrid": (
                float(np.mean(vals)) - nonkfold_hybrid_acc[snr]
                if snr in nonkfold_hybrid_acc else ""
            ),
        }
        summary.append(row)

    fields = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    return summary


def _plot_summary(summary: list[dict], fig_dir: Path, low_snr_max: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([r["snr"] for r in summary], dtype=np.int32)
    lstm = np.array([100 * float(r["lstm_baseline_accuracy"]) for r in summary])
    nonkfold = np.array([100 * float(r["nonkfold_manual_hybrid_accuracy"]) for r in summary])
    kfold_mean = np.array([100 * r["kfold_manual_hybrid_mean"] for r in summary])
    kfold_std = np.array([100 * r["kfold_manual_hybrid_std"] for r in summary])

    plt.figure(figsize=(12, 6))
    plt.plot(snrs, lstm, marker="x", linewidth=2.4,
             color="#9467bd", label="LSTM baseline")
    plt.plot(snrs, nonkfold, marker="o", linewidth=2.4,
             color="#1f77b4", label="Existing manual hybrid without k-fold")
    plt.plot(snrs, kfold_mean, marker="D", linewidth=2.8,
             color="#ff7f0e", label="Manual hybrid with k-fold mean")
    plt.fill_between(
        snrs,
        kfold_mean - kfold_std,
        kfold_mean + kfold_std,
        color="#ff7f0e",
        alpha=0.16,
        label="K-fold ±1 std",
    )
    plt.axvline(low_snr_max, color="black", linestyle="--", alpha=0.55,
                label=f"validation selection limited to SNR <= {low_snr_max} dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Manual SNR-aware Hybrid: K-fold vs Existing Non-kfold vs LSTM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "manual_hybrid_kfold_vs_nonkfold_vs_lstm_acc_vs_snr.png",
                dpi=240, bbox_inches="tight")
    plt.close()

    delta_nonkfold = kfold_mean - nonkfold
    delta_lstm = kfold_mean - lstm
    plt.figure(figsize=(12, 5))
    plt.plot(snrs, delta_nonkfold, marker="D", linewidth=2.5,
             label="K-fold hybrid - existing non-kfold hybrid")
    plt.plot(snrs, delta_lstm, marker="x", linewidth=2.5,
             label="K-fold hybrid - LSTM baseline")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(low_snr_max, color="black", linestyle="--", alpha=0.55)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta accuracy (percentage points)")
    plt.title("Manual Hybrid K-fold Improvement/Difference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "manual_hybrid_kfold_delta_vs_comparisons.png",
                dpi=240, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()

    import keras
    from src.dataset import normalize_samples
    from src.train_kfold import make_kfold_splits, prepare_inputs

    keras.mixed_precision.set_global_policy("float32")

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = Path(args.normal_kfold_dir)
    diff_dir = Path(args.diff_kfold_dir)
    lstm_acc = _load_acc_csv(args.lstm_baseline_csv, accuracy_col="accuracy")
    nonkfold_hybrid_acc = _load_existing_hybrid_csv(args.nonkfold_hybrid_csv)

    print("\n============================================================")
    print("  Experiment : manual_hybrid_kfold_compare")
    print("  Rule       : validation-best normal/diff per low-SNR value")
    print("  Alpha      : disabled/not used")
    print(f"  Mode       : {args.selection_mode}")
    print(f"  Threshold  : {args.low_snr_max} dB")
    print(f"  Output dir : {out_dir}")
    print("============================================================\n")

    X_raw, lbl, snrs = _load_raw_dataset(args.datasetpath, FIVE_CLASS)
    n_blocks = len(FIVE_CLASS) * len(snrs)
    test_idx, fold_idx = make_kfold_splits(
        n_blocks=n_blocks,
        block_size=1000,
        n_folds=args.n_folds,
        test_per_block=200,
        seed=args.seed,
    )

    X_test = normalize_samples(X_raw[test_idx])
    Y_test = _onehot([FIVE_CLASS.index(lbl[i][0]) for i in test_idx], len(FIVE_CLASS))
    test_snrs = np.array([lbl[i][1] for i in test_idx], dtype=np.int32)
    inp_test = prepare_inputs(X_test)

    fold_rows = []
    for fold in range(args.n_folds):
        print(f"\n[manual-kfold-hybrid] Fold {fold + 1}/{args.n_folds}")
        normal_weights = _require_fold_weights(normal_dir, fold)
        diff_weights = _require_fold_weights(diff_dir, fold)

        val_idx = fold_idx[fold]
        X_val = normalize_samples(X_raw[val_idx])
        Y_val = _onehot([FIVE_CLASS.index(lbl[i][0]) for i in val_idx], len(FIVE_CLASS))
        val_snrs = np.array([lbl[i][1] for i in val_idx], dtype=np.int32)
        inp_val = prepare_inputs(X_val)

        normal_val_pred = _predict_fold_model(
            lambda: _build_normal_attention_model(len(FIVE_CLASS)),
            normal_weights,
            inp_val,
            args.batch_size,
        )
        diff_val_pred = _predict_fold_model(
            lambda: _build_diff_attention_model(len(FIVE_CLASS)),
            diff_weights,
            inp_val,
            args.batch_size,
        )

        selected_by_snr = {}
        normal_val_acc_by_snr = {}
        diff_val_acc_by_snr = {}
        for snr in snrs:
            val_mask = val_snrs == snr
            normal_val_acc = _accuracy(Y_val[val_mask], normal_val_pred[val_mask])
            diff_val_acc = _accuracy(Y_val[val_mask], diff_val_pred[val_mask])
            normal_val_acc_by_snr[int(snr)] = normal_val_acc
            diff_val_acc_by_snr[int(snr)] = diff_val_acc
            if args.selection_mode == "fixed_threshold":
                selected_by_snr[int(snr)] = (
                    "diff_attention" if snr <= args.low_snr_max else "normal_attention"
                )
            elif snr <= args.low_snr_max and diff_val_acc > normal_val_acc + args.min_val_delta:
                selected_by_snr[int(snr)] = "diff_attention"
            else:
                selected_by_snr[int(snr)] = "normal_attention"

        normal_pred = _predict_fold_model(
            lambda: _build_normal_attention_model(len(FIVE_CLASS)),
            normal_weights,
            inp_test,
            args.batch_size,
        )
        diff_pred = _predict_fold_model(
            lambda: _build_diff_attention_model(len(FIVE_CLASS)),
            diff_weights,
            inp_test,
            args.batch_size,
        )

        use_diff = np.array([
            selected_by_snr[int(s)] == "diff_attention"
            for s in test_snrs
        ], dtype=bool)[:, None]
        hybrid_pred = np.where(use_diff, diff_pred, normal_pred)

        for snr in snrs:
            mask = test_snrs == snr
            fold_rows.append({
                "fold": fold,
                "snr": int(snr),
                "selected_model": selected_by_snr[int(snr)],
                "normal_attention_accuracy": _accuracy(Y_test[mask], normal_pred[mask]),
                "diff_attention_accuracy": _accuracy(Y_test[mask], diff_pred[mask]),
                "manual_hybrid_kfold_accuracy": _accuracy(Y_test[mask], hybrid_pred[mask]),
                "normal_val_accuracy": normal_val_acc_by_snr[int(snr)],
                "diff_val_accuracy": diff_val_acc_by_snr[int(snr)],
                "diff_minus_normal_val": (
                    diff_val_acc_by_snr[int(snr)] - normal_val_acc_by_snr[int(snr)]
                ),
            })

    _save_fold_rows(fold_rows, res_dir / "manual_hybrid_kfold_fold_acc_per_snr.csv")
    summary = _save_summary(
        fold_rows,
        lstm_acc=lstm_acc,
        nonkfold_hybrid_acc=nonkfold_hybrid_acc,
        path=res_dir / "manual_hybrid_kfold_vs_nonkfold_vs_lstm.csv",
    )
    _plot_summary(summary, fig_dir, low_snr_max=args.low_snr_max)

    overall = {}
    for key in [
        "lstm_baseline_accuracy",
        "nonkfold_manual_hybrid_accuracy",
        "kfold_manual_hybrid_mean",
    ]:
        vals = []
        for r in summary:
            if r[key] != "":
                vals.append(float(r[key]))
        overall[key] = float(np.mean(vals))

    with open(res_dir / "manual_hybrid_kfold_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "manual_snr_aware_hybrid_kfold_compare",
            "alpha_used": False,
            "selection_mode": args.selection_mode,
            "rule": (
                f"For SNR <= {args.low_snr_max} dB, choose normal/diff using fold "
                "validation accuracy. For cleaner SNR, use normal attention."
                if args.selection_mode == "validation_best"
                else f"SNR <= {args.low_snr_max} dB uses diff-attention, else normal-attention."
            ),
            "min_val_delta": args.min_val_delta,
            "n_folds": args.n_folds,
            "lstm_baseline_csv": str(args.lstm_baseline_csv),
            "nonkfold_hybrid_csv": str(args.nonkfold_hybrid_csv),
            "normal_kfold_dir": str(normal_dir),
            "diff_kfold_dir": str(diff_dir),
            "overall_mean_across_snr": overall,
        }, f, indent=2)

    print("\n[manual-kfold-hybrid] Saved:")
    print(f"  {res_dir / 'manual_hybrid_kfold_fold_acc_per_snr.csv'}")
    print(f"  {res_dir / 'manual_hybrid_kfold_vs_nonkfold_vs_lstm.csv'}")
    print(f"  {fig_dir / 'manual_hybrid_kfold_vs_nonkfold_vs_lstm_acc_vs_snr.png'}")
    print(f"  {fig_dir / 'manual_hybrid_kfold_delta_vs_comparisons.png'}")


if __name__ == "__main__":
    main()
