"""
evaluate_fourway_kfold_hybrid.py
================================

Four-way k-fold comparison for the 5-class AMR experiment.

Plots exactly the four curves requested:

1. LSTM/MCLDNN k-fold
2. Normal attention k-fold
3. Differential attention k-fold
4. Automated SNR-aware hybrid k-fold

Automated hybrid rule
---------------------
For every fold, the routing rule is learned from that fold's validation split:

    - for SNR <= low_snr_max, choose whichever model has higher validation
      accuracy at that SNR: normal attention or differential attention.
    - for cleaner SNRs, use normal attention.

No alpha blending, no confidence-interval smoothing, no monotonic trend forcing.
The final plot is mean ± std across the fold checkpoints.
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
        description="Four-way k-fold comparison: LSTM, normal attention, diff attention, automated hybrid."
    )
    p.add_argument("--datasetpath", required=True)
    p.add_argument(
        "--baseline-kfold-dir",
        default="experiments/5class_baseline_kfold/kfold",
    )
    p.add_argument(
        "--baseline-kfold-csv",
        default="experiments/5class_baseline_kfold/kfold/test/acc_per_snr.csv",
        help=(
            "Existing LSTM/MCLDNN k-fold acc_per_snr.csv. If present, the "
            "LSTM curve is loaded from this file and baseline fold weights "
            "are not required."
        ),
    )
    p.add_argument(
        "--normal-kfold-dir",
        default="experiments/5class_attention_kfold/kfold",
    )
    p.add_argument(
        "--diff-kfold-dir",
        default="experiments/5class_diffattention_kfold/kfold",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_fourway_kfold_hybrid",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--low-snr-max", type=int, default=2)
    p.add_argument("--min-val-delta", type=float, default=0.0)
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


def _require_fold_weights(kfold_dir: Path, fold: int) -> Path:
    path = kfold_dir / f"fold_{fold}" / "best_model.weights.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold weights: {path}\n"
            "Run src/train_kfold.py for the corresponding k-fold config first."
        )
    return path


def _load_acc_per_snr_csv(path: str | Path) -> dict[int, float] | None:
    path = Path(path)
    if not path.exists():
        return None
    out: dict[int, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(float(row["snr"]))] = float(row["accuracy"])
    print(f"[fourway-kfold] Loaded LSTM k-fold curve from CSV: {path}")
    return out


def _build_lstm_baseline_model(n_classes: int):
    from src.models.mcldnn import MCLDNN

    model = MCLDNN(
        classes=n_classes,
        dropout_rate=0.5,
        l2_dense=1.0e-3,
        l2_lstm=1.0e-4,
    )
    # MCLDNN() returns a compiled model in this codebase when used through
    # training, but load/predict does not require recompilation.  Return as-is.
    return model


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


def _accuracy_by_snr(y_true: np.ndarray,
                     pred: np.ndarray,
                     snr_labels: np.ndarray,
                     snrs: list[int]) -> dict[int, float]:
    return {
        int(snr): _accuracy(y_true[snr_labels == snr], pred[snr_labels == snr])
        for snr in snrs
    }


def _save_fold_rows(rows: list[dict], path: Path):
    fields = [
        "fold",
        "snr",
        "selected_model_for_hybrid",
        "lstm_kfold_accuracy",
        "normal_attention_kfold_accuracy",
        "diff_attention_kfold_accuracy",
        "automated_hybrid_kfold_accuracy",
        "normal_val_accuracy",
        "diff_val_accuracy",
        "diff_minus_normal_val",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_rows(rows: list[dict], path: Path) -> list[dict]:
    snrs = sorted({int(r["snr"]) for r in rows})
    metrics = [
        "lstm_kfold_accuracy",
        "normal_attention_kfold_accuracy",
        "diff_attention_kfold_accuracy",
        "automated_hybrid_kfold_accuracy",
    ]
    summary = []
    for snr in snrs:
        srows = [r for r in rows if int(r["snr"]) == snr]
        row = {"snr": snr, "n_folds": len(srows)}
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in srows], dtype=np.float32)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        row["hybrid_minus_normal_mean"] = (
            row["automated_hybrid_kfold_accuracy_mean"]
            - row["normal_attention_kfold_accuracy_mean"]
        )
        row["hybrid_minus_lstm_mean"] = (
            row["automated_hybrid_kfold_accuracy_mean"]
            - row["lstm_kfold_accuracy_mean"]
        )
        row["hybrid_minus_diff_mean"] = (
            row["automated_hybrid_kfold_accuracy_mean"]
            - row["diff_attention_kfold_accuracy_mean"]
        )
        summary.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    return summary


def _plot_summary(summary: list[dict], fig_dir: Path, low_snr_max: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([r["snr"] for r in summary], dtype=np.int32)
    series = [
        ("lstm_kfold_accuracy", "LSTM/MCLDNN k-fold", "#9467bd", "x"),
        ("normal_attention_kfold_accuracy", "Normal attention k-fold", "#1f77b4", "o"),
        ("diff_attention_kfold_accuracy", "Differential attention k-fold", "#d62728", "s"),
        ("automated_hybrid_kfold_accuracy", "Automated hybrid k-fold", "#ff7f0e", "D"),
    ]

    plt.figure(figsize=(12.5, 6.4))
    for key, label, color, marker in series:
        mean = np.array([100 * r[f"{key}_mean"] for r in summary])
        std = np.array([100 * r[f"{key}_std"] for r in summary])
        plt.plot(snrs, mean, marker=marker, linewidth=2.5, color=color, label=label)
        plt.fill_between(snrs, mean - std, mean + std, color=color, alpha=0.10)

    plt.axvline(low_snr_max, color="black", linestyle="--", alpha=0.55,
                label=f"hybrid validation-selection region: SNR <= {low_snr_max} dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Four-way K-fold Comparison: LSTM vs Attention vs Diff-Attention vs Hybrid")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "fourway_kfold_lstm_normal_diff_hybrid_acc_vs_snr.png",
                dpi=240, bbox_inches="tight")
    plt.close()

    hybrid = np.array([100 * r["automated_hybrid_kfold_accuracy_mean"] for r in summary])
    normal = np.array([100 * r["normal_attention_kfold_accuracy_mean"] for r in summary])
    diff = np.array([100 * r["diff_attention_kfold_accuracy_mean"] for r in summary])
    lstm = np.array([100 * r["lstm_kfold_accuracy_mean"] for r in summary])

    plt.figure(figsize=(12, 5))
    plt.plot(snrs, hybrid - normal, marker="D", linewidth=2.4,
             label="Hybrid - normal attention")
    plt.plot(snrs, hybrid - diff, marker="s", linewidth=2.4,
             label="Hybrid - differential attention")
    plt.plot(snrs, hybrid - lstm, marker="x", linewidth=2.4,
             label="Hybrid - LSTM baseline")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(low_snr_max, color="black", linestyle="--", alpha=0.55)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta accuracy (percentage points)")
    plt.title("Automated Hybrid K-fold Delta against Individual Models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "automated_hybrid_kfold_delta_vs_models.png",
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

    baseline_dir = Path(args.baseline_kfold_dir)
    normal_dir = Path(args.normal_kfold_dir)
    diff_dir = Path(args.diff_kfold_dir)
    baseline_csv_acc = _load_acc_per_snr_csv(args.baseline_kfold_csv)

    print("\n============================================================")
    print("  Experiment : fourway_kfold_hybrid")
    print("  Curves     : LSTM, normal-att, diff-att, automated hybrid")
    print("  Hybrid     : validation-selected per fold/SNR; no alpha")
    print(f"  Low SNR max: {args.low_snr_max} dB")
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
        print(f"\n[fourway-kfold] Fold {fold + 1}/{args.n_folds}")
        baseline_weights = (
            None if baseline_csv_acc is not None
            else _require_fold_weights(baseline_dir, fold)
        )
        normal_weights = _require_fold_weights(normal_dir, fold)
        diff_weights = _require_fold_weights(diff_dir, fold)

        val_idx = fold_idx[fold]
        X_val = normalize_samples(X_raw[val_idx])
        Y_val = _onehot([FIVE_CLASS.index(lbl[i][0]) for i in val_idx], len(FIVE_CLASS))
        val_snrs = np.array([lbl[i][1] for i in val_idx], dtype=np.int32)
        inp_val = prepare_inputs(X_val)

        if baseline_weights is not None:
            lstm_pred = _predict_fold_model(
                lambda: _build_lstm_baseline_model(len(FIVE_CLASS)),
                baseline_weights,
                inp_test,
                args.batch_size,
            )
        else:
            lstm_pred = None
        normal_val_pred = _predict_fold_model(
            lambda: _build_normal_attention_model(len(FIVE_CLASS)),
            normal_weights,
            inp_val,
            args.batch_size,
        )
        normal_test_pred = _predict_fold_model(
            lambda: _build_normal_attention_model(len(FIVE_CLASS)),
            normal_weights,
            inp_test,
            args.batch_size,
        )
        diff_val_pred = _predict_fold_model(
            lambda: _build_diff_attention_model(len(FIVE_CLASS)),
            diff_weights,
            inp_val,
            args.batch_size,
        )
        diff_test_pred = _predict_fold_model(
            lambda: _build_diff_attention_model(len(FIVE_CLASS)),
            diff_weights,
            inp_test,
            args.batch_size,
        )

        normal_val_acc = _accuracy_by_snr(Y_val, normal_val_pred, val_snrs, snrs)
        diff_val_acc = _accuracy_by_snr(Y_val, diff_val_pred, val_snrs, snrs)
        selected_by_snr = {}
        for snr in snrs:
            snr = int(snr)
            if (
                snr <= args.low_snr_max
                and diff_val_acc[snr] > normal_val_acc[snr] + args.min_val_delta
            ):
                selected_by_snr[snr] = "diff_attention"
            else:
                selected_by_snr[snr] = "normal_attention"

        use_diff = np.array([
            selected_by_snr[int(s)] == "diff_attention"
            for s in test_snrs
        ], dtype=bool)[:, None]
        hybrid_pred = np.where(use_diff, diff_test_pred, normal_test_pred)

        for snr in snrs:
            snr = int(snr)
            mask = test_snrs == snr
            fold_rows.append({
                "fold": fold,
                "snr": snr,
                "selected_model_for_hybrid": selected_by_snr[snr],
                "lstm_kfold_accuracy": (
                    baseline_csv_acc[snr]
                    if baseline_csv_acc is not None
                    else _accuracy(Y_test[mask], lstm_pred[mask])
                ),
                "normal_attention_kfold_accuracy": _accuracy(Y_test[mask], normal_test_pred[mask]),
                "diff_attention_kfold_accuracy": _accuracy(Y_test[mask], diff_test_pred[mask]),
                "automated_hybrid_kfold_accuracy": _accuracy(Y_test[mask], hybrid_pred[mask]),
                "normal_val_accuracy": normal_val_acc[snr],
                "diff_val_accuracy": diff_val_acc[snr],
                "diff_minus_normal_val": diff_val_acc[snr] - normal_val_acc[snr],
            })

    _save_fold_rows(fold_rows, res_dir / "fourway_kfold_fold_acc_per_snr.csv")
    summary = _summarize_rows(
        fold_rows,
        res_dir / "fourway_kfold_mean_std_acc_per_snr.csv",
    )
    _plot_summary(summary, fig_dir, low_snr_max=args.low_snr_max)

    overall = {}
    for key in [
        "lstm_kfold_accuracy_mean",
        "normal_attention_kfold_accuracy_mean",
        "diff_attention_kfold_accuracy_mean",
        "automated_hybrid_kfold_accuracy_mean",
    ]:
        overall[key] = float(np.mean([r[key] for r in summary]))

    metadata = {
        "method": "fourway_kfold_hybrid_comparison",
        "n_folds": args.n_folds,
        "low_snr_max": args.low_snr_max,
        "min_val_delta": args.min_val_delta,
        "hybrid_rule": (
            f"For each fold and SNR <= {args.low_snr_max} dB, choose diff-attention "
            "only if its validation accuracy is higher than normal attention. "
            "For cleaner SNRs, use normal attention."
        ),
        "alpha_used": False,
        "monotonic_smoothing_used": False,
        "baseline_kfold_dir": str(baseline_dir),
        "baseline_kfold_csv": str(args.baseline_kfold_csv),
        "baseline_curve_loaded_from_csv": baseline_csv_acc is not None,
        "normal_kfold_dir": str(normal_dir),
        "diff_kfold_dir": str(diff_dir),
        "overall_mean_across_snr": overall,
    }
    with open(res_dir / "fourway_kfold_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[fourway-kfold] Saved:")
    print(f"  {res_dir / 'fourway_kfold_fold_acc_per_snr.csv'}")
    print(f"  {res_dir / 'fourway_kfold_mean_std_acc_per_snr.csv'}")
    print(f"  {fig_dir / 'fourway_kfold_lstm_normal_diff_hybrid_acc_vs_snr.png'}")
    print(f"  {fig_dir / 'automated_hybrid_kfold_delta_vs_models.png'}")


if __name__ == "__main__":
    main()
