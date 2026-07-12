"""
evaluate_hybrid_confidence_intervals.py
=======================================

Confidence-interval analysis for the existing 5-class hybrid SNR-aware model.

This is not k-fold and not curve smoothing.  It quantifies uncertainty of the
existing hybrid result using:

1. Wilson 95% confidence intervals for per-SNR accuracy.
2. Paired bootstrap 95% confidence intervals for the per-SNR improvement:

       hybrid accuracy - normal attention accuracy

The paired bootstrap is useful because hybrid and normal predictions are made
on the same test samples, so the delta should preserve sample pairing.
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


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute confidence intervals for hybrid SNR-aware attention."
    )
    p.add_argument("--datasetpath", required=True)
    p.add_argument(
        "--normal-weights",
        default="experiments/5class_attention/checkpoints/best_model.weights.h5",
    )
    p.add_argument(
        "--diff-weights",
        default="experiments/5class_diffattention/checkpoints/best_model.weights.h5",
    )
    p.add_argument(
        "--lstm-baseline-csv",
        default="experiments/5class_baseline/results/acc_per_snr.csv",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_hybrid_confidence_intervals",
    )
    p.add_argument(
        "--selection-mode",
        choices=("validation_best", "fixed_band"),
        default="validation_best",
    )
    p.add_argument("--low-snr-max", type=int, default=2)
    p.add_argument("--diff-snr-min", type=int, default=-20)
    p.add_argument("--diff-snr-max", type=int, default=2)
    p.add_argument("--normal-at-upper-boundary", action="store_true")
    p.add_argument("--min-val-delta", type=float, default=0.0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--seed", type=int, default=2016)
    return p.parse_args()


def _make_iq_inputs(X: np.ndarray) -> list[np.ndarray]:
    return [
        np.expand_dims(X, axis=3).astype("float32"),
        np.expand_dims(X[:, 0, :], axis=2).astype("float32"),
        np.expand_dims(X[:, 1, :], axis=2).astype("float32"),
    ]


def _accuracy(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    return float(np.mean(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)))


def _accuracy_by_snr(Y: np.ndarray,
                     Y_hat: np.ndarray,
                     snr_labels: np.ndarray,
                     snrs: list[int]) -> dict[int, float]:
    return {
        int(snr): _accuracy(Y[snr_labels == snr], Y_hat[snr_labels == snr])
        for snr in snrs
    }


def _validation_best_selection(normal_val_pred: np.ndarray,
                               diff_val_pred: np.ndarray,
                               Y_val: np.ndarray,
                               val_snrs: np.ndarray,
                               snrs: list[int],
                               low_snr_max: int,
                               min_val_delta: float):
    normal_val_acc = _accuracy_by_snr(Y_val, normal_val_pred, val_snrs, snrs)
    diff_val_acc = _accuracy_by_snr(Y_val, diff_val_pred, val_snrs, snrs)
    selected_by_snr = {}
    for snr in snrs:
        snr = int(snr)
        if snr <= low_snr_max and diff_val_acc[snr] > normal_val_acc[snr] + min_val_delta:
            selected_by_snr[snr] = "diff_attention"
        else:
            selected_by_snr[snr] = "normal_attention"
    return selected_by_snr, normal_val_acc, diff_val_acc


def _hybrid_mask(test_snrs: np.ndarray,
                 diff_snr_min: int,
                 diff_snr_max: int,
                 normal_at_upper_boundary: bool) -> np.ndarray:
    if normal_at_upper_boundary:
        return (test_snrs >= diff_snr_min) & (test_snrs < diff_snr_max)
    return (test_snrs >= diff_snr_min) & (test_snrs <= diff_snr_max)


def _wilson_ci(correct: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    phat = correct / n
    denom = 1.0 + (z * z / n)
    centre = (phat + (z * z) / (2 * n)) / denom
    half = (
        z
        * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
        / denom
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_delta_ci(hybrid_correct: np.ndarray,
                        normal_correct: np.ndarray,
                        rng: np.random.Generator,
                        n_boot: int,
                        ci_level: float):
    """Paired bootstrap CI for mean(hybrid_correct - normal_correct)."""
    n = len(hybrid_correct)
    if n <= 1:
        delta = float(np.mean(hybrid_correct - normal_correct))
        return delta, np.nan, np.nan
    diffs = hybrid_correct.astype(np.float32) - normal_correct.astype(np.float32)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.mean(diffs[idx], axis=1)
    alpha = 1.0 - ci_level
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    return float(np.mean(diffs)), float(lo), float(hi)


def _load_lstm_acc(path: str | Path) -> dict[int, float]:
    path = Path(path)
    if not path.exists():
        print(f"[hybrid-ci] LSTM baseline CSV not found, skipping: {path}")
        return {}
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(float(row["snr"]))] = float(row["accuracy"])
    return out


def _save_ci_csv(rows: list[dict], path: Path):
    fields = [
        "snr",
        "n_test",
        "selected_model",
        "normal_attention_accuracy",
        "normal_attention_ci_low",
        "normal_attention_ci_high",
        "hybrid_accuracy",
        "hybrid_ci_low",
        "hybrid_ci_high",
        "delta_hybrid_minus_normal",
        "delta_bootstrap_ci_low",
        "delta_bootstrap_ci_high",
        "lstm_baseline_accuracy",
        "lstm_baseline_ci_low",
        "lstm_baseline_ci_high",
        "normal_val_accuracy",
        "diff_val_accuracy",
        "diff_minus_normal_val",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_ci(rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([r["snr"] for r in rows], dtype=np.int32)
    hybrid = np.array([100 * r["hybrid_accuracy"] for r in rows])
    hybrid_lo = np.array([100 * r["hybrid_ci_low"] for r in rows])
    hybrid_hi = np.array([100 * r["hybrid_ci_high"] for r in rows])
    normal = np.array([100 * r["normal_attention_accuracy"] for r in rows])
    normal_lo = np.array([100 * r["normal_attention_ci_low"] for r in rows])
    normal_hi = np.array([100 * r["normal_attention_ci_high"] for r in rows])

    plt.figure(figsize=(12, 6))
    plt.plot(snrs, normal, marker="o", linewidth=2.4, label="Normal attention")
    plt.fill_between(snrs, normal_lo, normal_hi, alpha=0.16)
    plt.plot(snrs, hybrid, marker="D", linewidth=2.6, label="Hybrid SNR-aware attention")
    plt.fill_between(snrs, hybrid_lo, hybrid_hi, alpha=0.18)
    plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="0 dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Normal vs Hybrid Accuracy with 95% Wilson Confidence Intervals")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "normal_vs_hybrid_accuracy_95ci_vs_snr.png",
                dpi=240, bbox_inches="tight")
    plt.close()

    has_lstm = all(r["lstm_baseline_accuracy"] != "" for r in rows)
    if has_lstm:
        lstm = np.array([100 * float(r["lstm_baseline_accuracy"]) for r in rows])
        lstm_lo = np.array([100 * float(r["lstm_baseline_ci_low"]) for r in rows])
        lstm_hi = np.array([100 * float(r["lstm_baseline_ci_high"]) for r in rows])
        plt.figure(figsize=(12, 6))
        plt.plot(snrs, lstm, marker="x", linewidth=2.2, label="LSTM baseline")
        plt.fill_between(snrs, lstm_lo, lstm_hi, alpha=0.12)
        plt.plot(snrs, normal, marker="o", linewidth=2.3, label="Normal attention")
        plt.fill_between(snrs, normal_lo, normal_hi, alpha=0.14)
        plt.plot(snrs, hybrid, marker="D", linewidth=2.6, label="Hybrid SNR-aware attention")
        plt.fill_between(snrs, hybrid_lo, hybrid_hi, alpha=0.18)
        plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="0 dB")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy (%)")
        plt.title("LSTM, Normal Attention, and Hybrid with 95% Confidence Intervals")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(snrs)
        plt.tight_layout()
        plt.savefig(fig_dir / "lstm_normal_hybrid_accuracy_95ci_vs_snr.png",
                    dpi=240, bbox_inches="tight")
        plt.close()

    delta = np.array([100 * r["delta_hybrid_minus_normal"] for r in rows])
    delta_lo = np.array([100 * r["delta_bootstrap_ci_low"] for r in rows])
    delta_hi = np.array([100 * r["delta_bootstrap_ci_high"] for r in rows])
    colors = ["#2ca02c" if lo > 0 else "#d62728" if hi < 0 else "#7f7f7f"
              for lo, hi in zip(delta_lo, delta_hi)]
    plt.figure(figsize=(12, 5))
    plt.bar([str(s) for s in snrs], delta, color=colors, alpha=0.78)
    x = np.arange(len(snrs))
    plt.errorbar(
        x,
        delta,
        yerr=[delta - delta_lo, delta_hi - delta],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.2,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Hybrid - normal accuracy (percentage points)")
    plt.title("Hybrid Improvement with 95% Paired Bootstrap Confidence Intervals")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "hybrid_minus_normal_delta_95ci_by_snr.png",
                dpi=240, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()

    import keras
    from src.dataset import FIVE_CLASS, load_data
    from src.models.mcldnn_attention import build_mcldnn_attention
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention

    keras.mixed_precision.set_global_policy("float32")

    normal_weights = Path(args.normal_weights)
    diff_weights = Path(args.diff_weights)
    if not normal_weights.exists():
        raise FileNotFoundError(f"Normal attention weights not found: {normal_weights}")
    if not diff_weights.exists():
        raise FileNotFoundError(f"Differential attention weights not found: {diff_weights}")

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("  Experiment : hybrid_confidence_intervals")
    print("  CI method  : Wilson accuracy CI + paired bootstrap delta CI")
    print(f"  Bootstrap  : {args.bootstrap_samples} samples")
    print(f"  Output dir : {out_dir}")
    print("============================================================\n")

    (mods, snrs, lbl), _, (X_val, Y_val), (X_test, Y_test), (_, val_idx, test_idx) = load_data(
        args.datasetpath,
        FIVE_CLASS,
        seed=args.seed,
        shuffle_split=False,
    )
    val_snrs = np.array([lbl[i][1] for i in val_idx], dtype=np.int32)
    test_snrs = np.array([lbl[i][1] for i in test_idx], dtype=np.int32)
    inputs_val = _make_iq_inputs(X_val)
    inputs_test = _make_iq_inputs(X_test)

    normal_model = build_mcldnn_attention(classes=len(FIVE_CLASS))
    normal_model.load_weights(normal_weights)
    print("[hybrid-ci] Predicting normal attention on validation...")
    normal_val_pred = normal_model.predict(inputs_val, batch_size=args.batch_size, verbose=1)
    print("[hybrid-ci] Predicting normal attention on test...")
    normal_test_pred = normal_model.predict(inputs_test, batch_size=args.batch_size, verbose=1)

    diff_model = build_mcldnn_diffattention(classes=len(FIVE_CLASS))
    diff_model.load_weights(diff_weights)
    print("[hybrid-ci] Predicting differential attention on validation...")
    diff_val_pred = diff_model.predict(inputs_val, batch_size=args.batch_size, verbose=1)
    print("[hybrid-ci] Predicting differential attention on test...")
    diff_test_pred = diff_model.predict(inputs_test, batch_size=args.batch_size, verbose=1)

    if args.selection_mode == "validation_best":
        selected_by_snr, normal_val_acc, diff_val_acc = _validation_best_selection(
            normal_val_pred=normal_val_pred,
            diff_val_pred=diff_val_pred,
            Y_val=Y_val,
            val_snrs=val_snrs,
            snrs=snrs,
            low_snr_max=args.low_snr_max,
            min_val_delta=args.min_val_delta,
        )
        use_diff = np.array(
            [selected_by_snr[int(s)] == "diff_attention" for s in test_snrs],
            dtype=bool,
        )
    else:
        use_diff = _hybrid_mask(
            test_snrs,
            diff_snr_min=args.diff_snr_min,
            diff_snr_max=args.diff_snr_max,
            normal_at_upper_boundary=args.normal_at_upper_boundary,
        )
        selected_by_snr = {
            int(s): "diff_attention" if bool(_hybrid_mask(
                np.array([s], dtype=np.int32),
                args.diff_snr_min,
                args.diff_snr_max,
                args.normal_at_upper_boundary,
            )[0]) else "normal_attention"
            for s in snrs
        }
        normal_val_acc = _accuracy_by_snr(Y_val, normal_val_pred, val_snrs, snrs)
        diff_val_acc = _accuracy_by_snr(Y_val, diff_val_pred, val_snrs, snrs)

    hybrid_test_pred = normal_test_pred.copy()
    hybrid_test_pred[use_diff] = diff_test_pred[use_diff]

    y_true = np.argmax(Y_test, axis=1)
    y_normal = np.argmax(normal_test_pred, axis=1)
    y_hybrid = np.argmax(hybrid_test_pred, axis=1)
    normal_correct_all = y_normal == y_true
    hybrid_correct_all = y_hybrid == y_true

    lstm_acc = _load_lstm_acc(args.lstm_baseline_csv)
    rng = np.random.default_rng(args.seed)
    z = 1.959963984540054

    rows = []
    for snr in snrs:
        snr = int(snr)
        mask = test_snrs == snr
        n = int(np.sum(mask))

        normal_correct = normal_correct_all[mask]
        hybrid_correct = hybrid_correct_all[mask]
        normal_c = int(np.sum(normal_correct))
        hybrid_c = int(np.sum(hybrid_correct))

        normal_lo, normal_hi = _wilson_ci(normal_c, n, z=z)
        hybrid_lo, hybrid_hi = _wilson_ci(hybrid_c, n, z=z)
        delta, delta_lo, delta_hi = _bootstrap_delta_ci(
            hybrid_correct=hybrid_correct,
            normal_correct=normal_correct,
            rng=rng,
            n_boot=args.bootstrap_samples,
            ci_level=args.ci_level,
        )

        lstm_value = lstm_acc.get(snr, "")
        if lstm_value != "":
            lstm_correct = int(round(float(lstm_value) * n))
            lstm_lo, lstm_hi = _wilson_ci(lstm_correct, n, z=z)
        else:
            lstm_lo, lstm_hi = "", ""

        rows.append({
            "snr": snr,
            "n_test": n,
            "selected_model": selected_by_snr[snr],
            "normal_attention_accuracy": normal_c / n,
            "normal_attention_ci_low": normal_lo,
            "normal_attention_ci_high": normal_hi,
            "hybrid_accuracy": hybrid_c / n,
            "hybrid_ci_low": hybrid_lo,
            "hybrid_ci_high": hybrid_hi,
            "delta_hybrid_minus_normal": delta,
            "delta_bootstrap_ci_low": delta_lo,
            "delta_bootstrap_ci_high": delta_hi,
            "lstm_baseline_accuracy": lstm_value,
            "lstm_baseline_ci_low": lstm_lo,
            "lstm_baseline_ci_high": lstm_hi,
            "normal_val_accuracy": normal_val_acc[snr],
            "diff_val_accuracy": diff_val_acc[snr],
            "diff_minus_normal_val": diff_val_acc[snr] - normal_val_acc[snr],
        })

    _save_ci_csv(rows, res_dir / "hybrid_confidence_intervals_per_snr.csv")
    _plot_ci(rows, fig_dir)

    overall_n = len(y_true)
    normal_overall = int(np.sum(normal_correct_all))
    hybrid_overall = int(np.sum(hybrid_correct_all))
    normal_overall_ci = _wilson_ci(normal_overall, overall_n, z=z)
    hybrid_overall_ci = _wilson_ci(hybrid_overall, overall_n, z=z)
    overall_delta = _bootstrap_delta_ci(
        hybrid_correct=hybrid_correct_all,
        normal_correct=normal_correct_all,
        rng=rng,
        n_boot=args.bootstrap_samples,
        ci_level=args.ci_level,
    )
    summary = {
        "n_test": overall_n,
        "normal_attention_accuracy": normal_overall / overall_n,
        "normal_attention_95ci": normal_overall_ci,
        "hybrid_accuracy": hybrid_overall / overall_n,
        "hybrid_95ci": hybrid_overall_ci,
        "delta_hybrid_minus_normal": overall_delta[0],
        "delta_95ci_paired_bootstrap": overall_delta[1:],
        "selection_mode": args.selection_mode,
        "low_snr_max": args.low_snr_max,
        "bootstrap_samples": args.bootstrap_samples,
    }
    with open(res_dir / "hybrid_confidence_interval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[hybrid-ci] Summary")
    print(f"  normal accuracy : {100 * summary['normal_attention_accuracy']:.2f}%")
    print(f"  hybrid accuracy : {100 * summary['hybrid_accuracy']:.2f}%")
    print(f"  delta           : {100 * summary['delta_hybrid_minus_normal']:.2f} pp")
    print(f"  Results saved   : {out_dir}/")


if __name__ == "__main__":
    main()
