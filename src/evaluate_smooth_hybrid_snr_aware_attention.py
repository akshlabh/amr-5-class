"""
evaluate_smooth_hybrid_snr_aware_attention.py
=============================================

Validation-calibrated smooth hybrid attention.

Problem addressed
-----------------
The earlier manual hybrid used a hard rule:

    low SNR -> differential attention
    high SNR -> normal attention

This can create a jagged SNR curve because differential attention is not better
at every low-SNR point.  For example, at some very low SNR values normal
attention can be slightly better simply due to model variance/noise.

This script uses a more stable and honest method:

1. Predict validation and test probabilities from both trained models.
2. On validation data only, learn a per-SNR mixing weight:

       p_hybrid = (1 - alpha) * p_normal + alpha * p_diff

   where alpha=0 means pure normal attention and alpha=1 means pure
   differential attention.
3. Smooth alpha across neighbouring SNRs using a Gaussian kernel.
4. Apply the smoothed alpha to the test set.
5. Save both raw curves and presentation-smoothed curves.

Important
---------
Test labels are never used to choose alpha.  Smoothing is applied to the
validation-derived routing weights, not to cheat using the test curve.
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
        description="Evaluate validation-calibrated smooth hybrid attention."
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
        "--lstm-baseline-acc-csv",
        default="experiments/5class_baseline/results/acc_per_snr.csv",
        help=(
            "Optional LSTM/MCLDNN baseline acc_per_snr.csv. If present, it is "
            "included in comparison plots and summary tables."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_smooth_hybrid_snr_aware_attention",
    )
    p.add_argument("--low-snr-max", type=int, default=2)
    p.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Grid step for validation search over alpha in [0, 1].",
    )
    p.add_argument(
        "--alpha-sigma-snrs",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma measured in SNR index steps.",
    )
    p.add_argument(
        "--presentation-smooth-window",
        type=int,
        default=3,
        help="Odd moving-average window used only for display curves.",
    )
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


def _categorical_crossentropy_np(Y: np.ndarray,
                                 Y_hat: np.ndarray,
                                 eps: float = 1e-7) -> float:
    Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
    return float(-np.mean(np.sum(Y * np.log(Y_hat), axis=1)))


def _blend(normal_pred: np.ndarray,
           diff_pred: np.ndarray,
           alpha: np.ndarray | float) -> np.ndarray:
    return (1.0 - alpha) * normal_pred + alpha * diff_pred


def _load_lstm_baseline_acc(path: str | Path | None) -> dict[int, float]:
    """
    Load the original MCLDNN/LSTM baseline per-SNR accuracy if available.

    The baseline is not part of the hybrid decision itself.  It is only used as
    an extra comparison curve in the plots/tables.
    """
    if path is None:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        print(f"[smooth-hybrid] LSTM baseline CSV not found, skipping: {csv_path}")
        return {}

    out: dict[int, float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "snr" not in row or "accuracy" not in row:
                continue
            out[int(float(row["snr"]))] = float(row["accuracy"])

    print(f"[smooth-hybrid] Loaded LSTM baseline curve from: {csv_path}")
    return out


def _best_alpha_on_validation(Y: np.ndarray,
                              normal_pred: np.ndarray,
                              diff_pred: np.ndarray,
                              alpha_grid: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_acc = -1.0
    for alpha in alpha_grid:
        pred = _blend(normal_pred, diff_pred, float(alpha))
        acc = _accuracy(Y, pred)
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)
    return best_alpha, best_acc


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if sigma <= 0:
        return values.copy()
    n = len(values)
    radius = max(1, int(np.ceil(3 * sigma)))
    out = np.zeros_like(values)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        x = np.arange(lo, hi) - i
        w = np.exp(-0.5 * (x / sigma) ** 2)
        w = w / np.sum(w)
        out[i] = float(np.sum(w * values[lo:hi]))
    return np.clip(out, 0.0, 1.0)


def _moving_average(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - radius)
        hi = min(len(arr), i + radius + 1)
        out[i] = float(np.mean(arr[lo:hi]))
    return out


def _isotonic_increasing(values: list[float] | np.ndarray) -> np.ndarray:
    """
    Pool-adjacent-violators isotonic regression for a non-decreasing trend.

    Accuracy vs SNR should be broadly non-decreasing physically: as SNR
    improves, the received signal becomes cleaner.  The raw per-SNR test curve
    can zig-zag because each SNR point is a finite sample estimate.  This
    function denoises the trend without using any future information beyond the
    sequence itself.
    """
    y = [float(v) for v in values]
    blocks = []
    for val in y:
        blocks.append({"sum": val, "count": 1, "avg": val})
        while len(blocks) >= 2 and blocks[-2]["avg"] > blocks[-1]["avg"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            merged = {
                "sum": b1["sum"] + b2["sum"],
                "count": b1["count"] + b2["count"],
            }
            merged["avg"] = merged["sum"] / merged["count"]
            blocks.append(merged)

    out = []
    for block in blocks:
        out.extend([block["avg"]] * block["count"])
    return np.asarray(out, dtype=np.float32)


def _regime_label(snr: int) -> str:
    """Stable SNR regimes for less noisy reporting than single-SNR points."""
    if snr <= -14:
        return "very_low_-20_to_-14"
    if snr <= -6:
        return "low_-12_to_-6"
    if snr <= 0:
        return "transition_-4_to_0"
    if snr <= 8:
        return "medium_2_to_8"
    return "high_10_to_18"


def _save_trend_and_regime_tables(rows: list[dict], res_dir: Path) -> tuple[list[dict], list[dict]]:
    snrs = [int(r["snr"]) for r in rows]
    has_lstm = all("lstm_baseline_accuracy" in r for r in rows)
    lstm = (
        np.array([r["lstm_baseline_accuracy"] for r in rows], dtype=np.float32)
        if has_lstm else None
    )
    normal = np.array([r["normal_attention_accuracy"] for r in rows], dtype=np.float32)
    diff = np.array([r["diff_attention_accuracy"] for r in rows], dtype=np.float32)
    hard = np.array([r["hard_validation_hybrid_accuracy"] for r in rows], dtype=np.float32)
    smooth = np.array([r["smooth_alpha_hybrid_accuracy"] for r in rows], dtype=np.float32)

    lstm_trend = _isotonic_increasing(lstm) if lstm is not None else None
    normal_trend = _isotonic_increasing(normal)
    diff_trend = _isotonic_increasing(diff)
    hard_trend = _isotonic_increasing(hard)
    smooth_trend = _isotonic_increasing(smooth)

    trend_rows = []
    with open(res_dir / "trend_smoothed_acc_per_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "snr",
            "lstm_baseline_raw",
            "normal_raw",
            "diff_raw",
            "hard_hybrid_raw",
            "smooth_hybrid_raw",
            "lstm_baseline_monotonic_trend",
            "normal_monotonic_trend",
            "diff_monotonic_trend",
            "hard_hybrid_monotonic_trend",
            "smooth_hybrid_monotonic_trend",
            "smooth_hybrid_trend_minus_normal_trend",
        ]
        writer.writerow(header)
        for i, snr in enumerate(snrs):
            row = {
                "snr": snr,
                "lstm_baseline_raw": float(lstm[i]) if lstm is not None else "",
                "normal_raw": float(normal[i]),
                "diff_raw": float(diff[i]),
                "hard_hybrid_raw": float(hard[i]),
                "smooth_hybrid_raw": float(smooth[i]),
                "lstm_baseline_monotonic_trend": float(lstm_trend[i]) if lstm_trend is not None else "",
                "normal_monotonic_trend": float(normal_trend[i]),
                "diff_monotonic_trend": float(diff_trend[i]),
                "hard_hybrid_monotonic_trend": float(hard_trend[i]),
                "smooth_hybrid_monotonic_trend": float(smooth_trend[i]),
                "smooth_hybrid_trend_minus_normal_trend": float(smooth_trend[i] - normal_trend[i]),
            }
            trend_rows.append(row)
            writer.writerow([
                row["snr"],
                row["lstm_baseline_raw"],
                row["normal_raw"],
                row["diff_raw"],
                row["hard_hybrid_raw"],
                row["smooth_hybrid_raw"],
                row["lstm_baseline_monotonic_trend"],
                row["normal_monotonic_trend"],
                row["diff_monotonic_trend"],
                row["hard_hybrid_monotonic_trend"],
                row["smooth_hybrid_monotonic_trend"],
                row["smooth_hybrid_trend_minus_normal_trend"],
            ])

    regimes = []
    for label in [
        "very_low_-20_to_-14",
        "low_-12_to_-6",
        "transition_-4_to_0",
        "medium_2_to_8",
        "high_10_to_18",
    ]:
        idx = [i for i, snr in enumerate(snrs) if _regime_label(snr) == label]
        if not idx:
            continue
        regime = {
            "regime": label,
            "snr_values": " ".join(str(snrs[i]) for i in idx),
            "lstm_baseline_accuracy": float(np.mean(lstm[idx])) if lstm is not None else "",
            "normal_attention_accuracy": float(np.mean(normal[idx])),
            "diff_attention_accuracy": float(np.mean(diff[idx])),
            "hard_validation_hybrid_accuracy": float(np.mean(hard[idx])),
            "smooth_alpha_hybrid_accuracy": float(np.mean(smooth[idx])),
        }
        regime["smooth_hybrid_minus_normal"] = (
            regime["smooth_alpha_hybrid_accuracy"] - regime["normal_attention_accuracy"]
        )
        regimes.append(regime)

    with open(res_dir / "regime_average_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "regime",
            "snr_values",
            "lstm_baseline_accuracy",
            "normal_attention_accuracy",
            "diff_attention_accuracy",
            "hard_validation_hybrid_accuracy",
            "smooth_alpha_hybrid_accuracy",
            "smooth_hybrid_minus_normal",
        ])
        for r in regimes:
            writer.writerow([
                r["regime"],
                r["snr_values"],
                r["lstm_baseline_accuracy"],
                r["normal_attention_accuracy"],
                r["diff_attention_accuracy"],
                r["hard_validation_hybrid_accuracy"],
                r["smooth_alpha_hybrid_accuracy"],
                r["smooth_hybrid_minus_normal"],
            ])

    return trend_rows, regimes


def _plot_curves(rows: list[dict], fig_dir: Path, presentation_window: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [r["snr"] for r in rows]
    has_lstm = all("lstm_baseline_accuracy" in r for r in rows)
    lstm = (
        np.array([100 * r["lstm_baseline_accuracy"] for r in rows])
        if has_lstm else None
    )
    normal = np.array([100 * r["normal_attention_accuracy"] for r in rows])
    diff = np.array([100 * r["diff_attention_accuracy"] for r in rows])
    hard = np.array([100 * r["hard_validation_hybrid_accuracy"] for r in rows])
    smooth = np.array([100 * r["smooth_alpha_hybrid_accuracy"] for r in rows])

    plt.figure(figsize=(12, 6))
    if lstm is not None:
        plt.plot(snrs, lstm, marker="x", linewidth=2.2,
                 color="#9467bd", label="LSTM baseline")
    plt.plot(snrs, hard, marker="^", linewidth=2.3,
             label="Hard validation-selected hybrid")
    plt.plot(snrs, smooth, marker="D", linewidth=2.6,
             label="Smooth validation-calibrated hybrid")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55, label="0 dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Smooth Hybrid SNR-aware Attention — Raw Test Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_raw_acc_vs_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    lstm_s = _moving_average(lstm, presentation_window) if lstm is not None else None
    hard_s = _moving_average(hard, presentation_window)
    smooth_s = _moving_average(smooth, presentation_window)

    plt.figure(figsize=(12, 6))
    if lstm is not None:
        plt.plot(snrs, lstm, color="#9467bd", alpha=0.22, linewidth=1.5)
    plt.plot(snrs, hard, color="#d62728", alpha=0.22, linewidth=1.5)
    plt.plot(snrs, smooth, color="#2ca02c", alpha=0.22, linewidth=1.5)
    if lstm_s is not None:
        plt.plot(snrs, lstm_s, marker="x", linewidth=2.8,
                 color="#9467bd", label=f"LSTM baseline ({presentation_window}-point smoothed)")
    plt.plot(snrs, hard_s, marker="^", linewidth=2.8,
             color="#d62728", label=f"Hard validation hybrid ({presentation_window}-point smoothed)")
    plt.plot(snrs, smooth_s, marker="D", linewidth=3.0,
             color="#2ca02c", label=f"Smooth validation hybrid ({presentation_window}-point smoothed)")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55, label="0 dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Presentation-smoothed Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_presentation_smoothed_acc_vs_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    delta = smooth - normal
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    plt.figure(figsize=(11, 4.8))
    plt.bar([str(s) for s in snrs], delta, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Smooth hybrid - normal (percentage points)")
    plt.title("Smooth Hybrid Improvement over Normal Attention")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_minus_normal_delta_by_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    raw_delta = smooth - normal
    smoothed_delta = _moving_average(raw_delta, presentation_window)
    plt.figure(figsize=(11, 4.8))
    plt.plot(snrs, raw_delta, marker="o", alpha=0.35, label="raw delta")
    plt.plot(snrs, smoothed_delta, marker="D", linewidth=2.8,
             label=f"{presentation_window}-point smoothed delta")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", alpha=0.55)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta accuracy (percentage points)")
    plt.title("Smoothed Delta: Smooth Hybrid minus Normal Attention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_smoothed_delta_by_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def _plot_trend_and_regime(trend_rows: list[dict], regime_rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [r["snr"] for r in trend_rows]
    has_lstm = all(r.get("lstm_baseline_raw", "") != "" for r in trend_rows)
    lstm_raw = (
        np.array([100 * r["lstm_baseline_raw"] for r in trend_rows])
        if has_lstm else None
    )
    normal_raw = np.array([100 * r["normal_raw"] for r in trend_rows])
    diff_raw = np.array([100 * r["diff_raw"] for r in trend_rows])
    hard_raw = np.array([100 * r["hard_hybrid_raw"] for r in trend_rows])
    hybrid_raw = np.array([100 * r["smooth_hybrid_raw"] for r in trend_rows])
    lstm_trend = (
        np.array([100 * r["lstm_baseline_monotonic_trend"] for r in trend_rows])
        if has_lstm else None
    )
    normal_trend = np.array([100 * r["normal_monotonic_trend"] for r in trend_rows])
    diff_trend = np.array([100 * r["diff_monotonic_trend"] for r in trend_rows])
    hard_trend = np.array([100 * r["hard_hybrid_monotonic_trend"] for r in trend_rows])
    hybrid_trend = np.array([100 * r["smooth_hybrid_monotonic_trend"] for r in trend_rows])

    plt.figure(figsize=(12, 6))
    if lstm_raw is not None:
        plt.scatter(snrs, lstm_raw, color="#9467bd", alpha=0.25, s=35)
    plt.scatter(snrs, hard_raw, color="#d62728", alpha=0.25, s=35)
    plt.scatter(snrs, hybrid_raw, color="#2ca02c", alpha=0.25, s=45)
    if lstm_trend is not None:
        plt.plot(snrs, lstm_trend, marker="x", linewidth=3.0,
                 color="#9467bd", label="LSTM baseline monotonic trend")
    plt.plot(snrs, hard_trend, marker="^", linewidth=3.0,
             color="#d62728", label="Hard validation hybrid monotonic trend")
    plt.plot(snrs, hybrid_trend, marker="D", linewidth=3.2,
             color="#2ca02c", label="Smooth validation hybrid monotonic trend")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55, label="0 dB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Monotonic Trend-smoothed Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_monotonic_trend_acc_vs_snr.png",
                dpi=220, bbox_inches="tight")
    plt.close()

    trend_delta = hybrid_trend - normal_trend
    plt.figure(figsize=(11, 4.8))
    plt.plot(snrs, trend_delta, marker="D", linewidth=2.8,
             color="#2ca02c", label="Hybrid trend - normal trend")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", alpha=0.55)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta accuracy (percentage points)")
    plt.title("Monotonic Trend Delta: Smooth Hybrid minus Normal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_monotonic_trend_delta.png",
                dpi=220, bbox_inches="tight")
    plt.close()

    labels = [r["regime"].replace("_", "\n") for r in regime_rows]
    has_lstm_regime = all(r.get("lstm_baseline_accuracy", "") != "" for r in regime_rows)
    lstm = (
        np.array([100 * r["lstm_baseline_accuracy"] for r in regime_rows])
        if has_lstm_regime else None
    )
    normal = np.array([100 * r["normal_attention_accuracy"] for r in regime_rows])
    diff = np.array([100 * r["diff_attention_accuracy"] for r in regime_rows])
    hard = np.array([100 * r["hard_validation_hybrid_accuracy"] for r in regime_rows])
    hybrid = np.array([100 * r["smooth_alpha_hybrid_accuracy"] for r in regime_rows])
    x = np.arange(len(labels))
    width = 0.25 if lstm is not None else 0.35

    plt.figure(figsize=(12, 5.8))
    if lstm is not None:
        plt.bar(x - width, lstm, width=width, label="LSTM baseline")
        plt.bar(x, hard, width=width, label="Hard validation hybrid")
        plt.bar(x + width, hybrid, width=width, label="Smooth validation hybrid")
    else:
        plt.bar(x - width / 2, hard, width=width, label="Hard validation hybrid")
        plt.bar(x + width / 2, hybrid, width=width, label="Smooth validation hybrid")
    plt.ylabel("Average accuracy (%)")
    plt.xlabel("SNR regime")
    plt.title("Stable SNR-regime Average Accuracy")
    plt.xticks(x, labels, fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "smooth_hybrid_regime_average_accuracy.png",
                dpi=220, bbox_inches="tight")
    plt.close()


def _plot_alpha(rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [r["snr"] for r in rows]
    raw_alpha = [r["validation_best_alpha"] for r in rows]
    smooth_alpha = [r["smoothed_alpha"] for r in rows]

    plt.figure(figsize=(11, 4.8))
    plt.plot(snrs, raw_alpha, marker="o", linewidth=2.2,
             label="Validation-best alpha")
    plt.plot(snrs, smooth_alpha, marker="D", linewidth=2.6,
             label="Smoothed alpha used on test")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Differential-attention mixing weight alpha")
    plt.title("Validation-derived Smooth Hybrid Mixing Weights")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "validation_alpha_vs_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()

    import keras
    from src.dataset import FIVE_CLASS, load_data
    from src.models.mcldnn_attention import build_mcldnn_attention
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention
    from src.utils.mltools import calculate_confusion_matrix, plot_confusion_matrix

    keras.mixed_precision.set_global_policy("float32")

    normal_weights = Path(args.normal_weights)
    diff_weights = Path(args.diff_weights)
    if not normal_weights.exists():
        raise FileNotFoundError(f"Normal attention weights not found: {normal_weights}")
    if not diff_weights.exists():
        raise FileNotFoundError(f"Diff attention weights not found: {diff_weights}")

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("  Experiment : 5class_smooth_hybrid_snr_aware_attention")
    print("  Method     : validation-calibrated smooth probability ensemble")
    print(f"  Low SNR max: {args.low_snr_max} dB")
    print(f"  Alpha sigma: {args.alpha_sigma_snrs}")
    print(f"  Output dir : {out_dir}")
    print("============================================================\n")

    lstm_baseline_acc_by_snr = _load_lstm_baseline_acc(args.lstm_baseline_acc_csv)

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
    print("[smooth-hybrid] Predicting normal attention on validation...")
    normal_val_pred = normal_model.predict(
        inputs_val, batch_size=args.batch_size, verbose=1
    )
    print("[smooth-hybrid] Predicting normal attention on test...")
    normal_test_pred = normal_model.predict(
        inputs_test, batch_size=args.batch_size, verbose=1
    )

    diff_model = build_mcldnn_diffattention(classes=len(FIVE_CLASS))
    diff_model.load_weights(diff_weights)
    print("[smooth-hybrid] Predicting diff attention on validation...")
    diff_val_pred = diff_model.predict(
        inputs_val, batch_size=args.batch_size, verbose=1
    )
    print("[smooth-hybrid] Predicting diff attention on test...")
    diff_test_pred = diff_model.predict(
        inputs_test, batch_size=args.batch_size, verbose=1
    )

    alpha_grid = np.round(
        np.arange(0.0, 1.0 + 0.5 * args.alpha_step, args.alpha_step),
        6,
    )
    snr_values = [int(s) for s in sorted(snrs)]
    raw_alpha = []
    val_best_acc = []
    val_normal_acc = []
    val_diff_acc = []

    for snr in snr_values:
        mask = val_snrs == snr
        n_acc = _accuracy(Y_val[mask], normal_val_pred[mask])
        d_acc = _accuracy(Y_val[mask], diff_val_pred[mask])
        best_alpha, best_acc = _best_alpha_on_validation(
            Y_val[mask],
            normal_val_pred[mask],
            diff_val_pred[mask],
            alpha_grid,
        )

        # For cleaner SNRs, force the robust prior that normal attention should
        # dominate.  This keeps the method focused on low/transition SNR.
        if snr > args.low_snr_max:
            best_alpha = 0.0
            best_acc = n_acc

        raw_alpha.append(best_alpha)
        val_best_acc.append(best_acc)
        val_normal_acc.append(n_acc)
        val_diff_acc.append(d_acc)

    smooth_alpha = _gaussian_smooth(
        np.array(raw_alpha, dtype=np.float32),
        sigma=args.alpha_sigma_snrs,
    )
    # Preserve the high-SNR prior after smoothing.
    smooth_alpha = np.array([
        float(a) if snr <= args.low_snr_max else 0.0
        for snr, a in zip(snr_values, smooth_alpha)
    ], dtype=np.float32)

    hard_alpha_by_snr = dict(zip(snr_values, raw_alpha))
    smooth_alpha_by_snr = dict(zip(snr_values, smooth_alpha))

    hard_alpha_test = np.array(
        [hard_alpha_by_snr[int(s)] for s in test_snrs],
        dtype=np.float32,
    )[:, None]
    smooth_alpha_test = np.array(
        [smooth_alpha_by_snr[int(s)] for s in test_snrs],
        dtype=np.float32,
    )[:, None]

    hard_hybrid_pred = _blend(normal_test_pred, diff_test_pred, hard_alpha_test)
    smooth_hybrid_pred = _blend(normal_test_pred, diff_test_pred, smooth_alpha_test)

    rows = []
    with open(res_dir / "smooth_hybrid_acc_per_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr",
            "n_test",
            "lstm_baseline_accuracy",
            "normal_attention_accuracy",
            "diff_attention_accuracy",
            "hard_validation_hybrid_accuracy",
            "smooth_alpha_hybrid_accuracy",
            "smooth_hybrid_minus_normal",
            "normal_val_accuracy",
            "diff_val_accuracy",
            "validation_best_alpha",
            "validation_best_alpha_accuracy",
            "smoothed_alpha",
        ])
        for i, snr in enumerate(snr_values):
            mask = test_snrs == snr
            row = {
                "snr": snr,
                "n_test": int(np.sum(mask)),
                "normal_attention_accuracy": _accuracy(Y_test[mask], normal_test_pred[mask]),
                "diff_attention_accuracy": _accuracy(Y_test[mask], diff_test_pred[mask]),
                "hard_validation_hybrid_accuracy": _accuracy(Y_test[mask], hard_hybrid_pred[mask]),
                "smooth_alpha_hybrid_accuracy": _accuracy(Y_test[mask], smooth_hybrid_pred[mask]),
                "normal_val_accuracy": val_normal_acc[i],
                "diff_val_accuracy": val_diff_acc[i],
                "validation_best_alpha": raw_alpha[i],
                "validation_best_alpha_accuracy": val_best_acc[i],
                "smoothed_alpha": float(smooth_alpha[i]),
            }
            if snr in lstm_baseline_acc_by_snr:
                row["lstm_baseline_accuracy"] = lstm_baseline_acc_by_snr[snr]
            row["smooth_hybrid_minus_normal"] = (
                row["smooth_alpha_hybrid_accuracy"] - row["normal_attention_accuracy"]
            )
            rows.append(row)
            writer.writerow([
                row["snr"],
                row["n_test"],
                row.get("lstm_baseline_accuracy", ""),
                row["normal_attention_accuracy"],
                row["diff_attention_accuracy"],
                row["hard_validation_hybrid_accuracy"],
                row["smooth_alpha_hybrid_accuracy"],
                row["smooth_hybrid_minus_normal"],
                row["normal_val_accuracy"],
                row["diff_val_accuracy"],
                row["validation_best_alpha"],
                row["validation_best_alpha_accuracy"],
                row["smoothed_alpha"],
            ])

    score_rows = [
        ("normal_attention", normal_test_pred),
        ("diff_attention", diff_test_pred),
        ("hard_validation_hybrid", hard_hybrid_pred),
        ("smooth_alpha_hybrid", smooth_hybrid_pred),
    ]
    with open(res_dir / "test_score.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "loss", "accuracy"])
        if all("lstm_baseline_accuracy" in r for r in rows):
            n = np.array([r["n_test"] for r in rows], dtype=np.float32)
            a = np.array([r["lstm_baseline_accuracy"] for r in rows], dtype=np.float32)
            writer.writerow(["lstm_baseline", "", float(np.sum(n * a) / np.sum(n))])
        for name, pred in score_rows:
            writer.writerow([
                name,
                _categorical_crossentropy_np(Y_test, pred),
                _accuracy(Y_test, pred),
            ])

    with open(res_dir / "validation_alpha_by_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr",
            "normal_val_accuracy",
            "diff_val_accuracy",
            "validation_best_alpha",
            "validation_best_alpha_accuracy",
            "smoothed_alpha_used_on_test",
        ])
        for i, snr in enumerate(snr_values):
            writer.writerow([
                snr,
                val_normal_acc[i],
                val_diff_acc[i],
                raw_alpha[i],
                val_best_acc[i],
                float(smooth_alpha[i]),
            ])

    conf_smooth, _, _ = calculate_confusion_matrix(Y_test, smooth_hybrid_pred, mods)
    conf_normal, _, _ = calculate_confusion_matrix(Y_test, normal_test_pred, mods)
    np.savetxt(res_dir / "confusion_smooth_hybrid_all_snrs.csv",
               conf_smooth, delimiter=",")
    np.savetxt(res_dir / "confusion_normal_attention_all_snrs.csv",
               conf_normal, delimiter=",")
    with open(res_dir / "confusion_smooth_hybrid_all_snrs.dat", "wb") as f:
        pickle.dump(conf_smooth, f)

    plot_confusion_matrix(
        conf_smooth,
        labels=mods,
        title="Smooth Hybrid Confusion — all SNRs",
        save_filename=str(fig_dir / "confusion_smooth_hybrid_all_snrs.png"),
    )
    plot_confusion_matrix(
        conf_normal,
        labels=mods,
        title="Normal Attention Confusion — all SNRs",
        save_filename=str(fig_dir / "confusion_normal_attention_all_snrs.png"),
    )

    _plot_curves(rows, fig_dir, args.presentation_smooth_window)
    trend_rows, regime_rows = _save_trend_and_regime_tables(rows, res_dir)
    _plot_trend_and_regime(trend_rows, regime_rows, fig_dir)
    _plot_alpha(rows, fig_dir)

    metadata = {
        "method": "validation_calibrated_smooth_probability_ensemble",
        "formula": "p_hybrid = (1 - alpha) * p_normal + alpha * p_diff",
        "alpha_source": "validation split only",
        "test_labels_used_for_alpha": False,
        "low_snr_max": args.low_snr_max,
        "alpha_step": args.alpha_step,
        "alpha_sigma_snrs": args.alpha_sigma_snrs,
        "presentation_smooth_window": args.presentation_smooth_window,
        "trend_smoothing": "monotonic isotonic regression using accuracy-vs-SNR physical prior",
        "regime_averaging": "very_low, low, transition, medium, high SNR bins",
        "lstm_baseline_acc_csv": str(args.lstm_baseline_acc_csv),
        "lstm_baseline_used_in_plots": all("lstm_baseline_accuracy" in r for r in rows),
        "normal_weights": str(normal_weights),
        "diff_weights": str(diff_weights),
    }
    with open(res_dir / "method_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(res_dir / "method_note.txt", "w", encoding="utf-8") as f:
        f.write("Smooth hybrid SNR-aware attention\n")
        f.write("=================================\n")
        f.write("This experiment does not simply smooth the plotted accuracy curve.\n")
        f.write("It learns a normal/diff probability mixing weight alpha using validation data only.\n")
        f.write("Then alpha is smoothed across neighbouring SNR values and applied to test predictions.\n")
        f.write("The presentation-smoothed plot is marked separately and is only for visual readability.\n")

    print("\n[smooth-hybrid] Summary")
    if all("lstm_baseline_accuracy" in r for r in rows):
        n = np.array([r["n_test"] for r in rows], dtype=np.float32)
        a = np.array([r["lstm_baseline_accuracy"] for r in rows], dtype=np.float32)
        print(f"  {'lstm_baseline':28s}: {100 * float(np.sum(n * a) / np.sum(n)):.2f}%")
    for name, pred in score_rows:
        print(f"  {name:28s}: {100 * _accuracy(Y_test, pred):.2f}%")
    print(f"  Results saved under           : {out_dir}/")


if __name__ == "__main__":
    main()
