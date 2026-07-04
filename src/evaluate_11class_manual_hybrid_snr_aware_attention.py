"""
evaluate_11class_manual_hybrid_snr_aware_attention.py
=====================================================

Manual SNR-aware hybrid for all 11 RadioML2016.10a classes.

This script compares:
    1. 11-class normal attention
    2. 11-class differential attention
    3. manual SNR-aware hybrid

Manual hybrid rule:
    true SNR <= threshold_db  -> use differential attention prediction
    true SNR >  threshold_db  -> use normal attention prediction

No gated/SNR-prediction model is used here.
"""

from __future__ import annotations

import argparse
import csv
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
        description="Evaluate 11-class normal, differential, and manual hybrid attention."
    )
    p.add_argument("--datasetpath", required=True)
    p.add_argument(
        "--normal-weights",
        default="experiments/11class_attention/checkpoints/best_model.weights.h5",
    )
    p.add_argument(
        "--diff-weights",
        default="experiments/11class_diffattention/checkpoints/best_model.weights.h5",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/11class_manual_hybrid_snr_aware_attention",
    )
    p.add_argument("--threshold-db", type=float, default=0.0)
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


def _plot_threeway_acc(per_snr_rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [r["snr"] for r in per_snr_rows]
    normal = [100 * r["normal_attention_accuracy"] for r in per_snr_rows]
    diff = [100 * r["diff_attention_accuracy"] for r in per_snr_rows]
    hybrid = [100 * r["manual_hybrid_accuracy"] for r in per_snr_rows]

    plt.figure(figsize=(12, 6))
    plt.plot(snrs, normal, marker="o", linewidth=2.4, label="Normal attention")
    plt.plot(snrs, diff, marker="s", linewidth=2.4, label="Differential attention")
    plt.plot(snrs, hybrid, marker="^", linewidth=2.6,
             label="Manual hybrid SNR-aware")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55,
                label="0 dB route boundary")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("11-class Normal vs Differential vs Manual Hybrid SNR-aware Attention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "normal_vs_diff_vs_manual_hybrid_acc_vs_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    delta = [h - n for h, n in zip(hybrid, normal)]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    plt.figure(figsize=(11, 4.8))
    plt.bar([str(s) for s in snrs], delta, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Hybrid - normal accuracy (percentage points)")
    plt.title("11-class Manual Hybrid minus Normal Attention by SNR")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "manual_hybrid_minus_normal_delta_by_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()

    import keras
    from src.dataset import ALL_CLASSES, load_data
    from src.models.mcldnn_attention import build_mcldnn_attention
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention
    from src.utils.mltools import calculate_confusion_matrix, plot_confusion_matrix

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
    print("  Experiment : 11class_manual_hybrid_snr_aware_attention")
    print("  Classes    : all 11 RadioML2016.10a classes")
    print(f"  Rule       : SNR <= {args.threshold_db:g} dB -> diff, else normal")
    print(f"  Normal ckpt: {normal_weights}")
    print(f"  Diff ckpt  : {diff_weights}")
    print(f"  Output dir : {out_dir}")
    print("============================================================\n")

    (mods, snrs, lbl), _, _, (X_test, Y_test), (_, _, test_idx) = load_data(
        args.datasetpath,
        ALL_CLASSES,
        seed=args.seed,
        shuffle_split=False,
    )
    test_snrs = np.array([lbl[i][1] for i in test_idx], dtype=np.int32)
    inputs_test = _make_iq_inputs(X_test)

    normal_model = build_mcldnn_attention(classes=len(ALL_CLASSES))
    normal_model.load_weights(normal_weights)
    print("[11class-hybrid] Predicting normal attention...")
    normal_pred = normal_model.predict(
        inputs_test, batch_size=args.batch_size, verbose=1
    )

    diff_model = build_mcldnn_diffattention(classes=len(ALL_CLASSES))
    diff_model.load_weights(diff_weights)
    print("[11class-hybrid] Predicting differential attention...")
    diff_pred = diff_model.predict(
        inputs_test, batch_size=args.batch_size, verbose=1
    )

    use_diff = test_snrs <= args.threshold_db
    hybrid_pred = normal_pred.copy()
    hybrid_pred[use_diff] = diff_pred[use_diff]

    print(f"[11class-hybrid] Test samples using diff attention  : {int(use_diff.sum())}")
    print(f"[11class-hybrid] Test samples using normal attention: {int((~use_diff).sum())}")

    rows = [
        ("normal_attention", normal_pred),
        ("diff_attention", diff_pred),
        ("manual_hybrid_snr_aware", hybrid_pred),
    ]

    with open(res_dir / "test_score.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "loss", "accuracy"])
        for name, pred in rows:
            writer.writerow([
                name,
                _categorical_crossentropy_np(Y_test, pred),
                _accuracy(Y_test, pred),
            ])

    with open(res_dir / "manual_hybrid_rule.txt", "w", encoding="utf-8") as f:
        f.write("11-class manual hybrid SNR-aware attention rule\n")
        f.write("===============================================\n")
        f.write(f"SNR <= {args.threshold_db:g} dB -> differential attention\n")
        f.write(f"SNR >  {args.threshold_db:g} dB -> normal attention\n")
        f.write("No gated SNR prediction model is used.\n")
        f.write(f"Normal checkpoint: {normal_weights}\n")
        f.write(f"Diff checkpoint  : {diff_weights}\n")

    # Overall confusion matrices.
    for name, pred in rows:
        conf, _, _ = calculate_confusion_matrix(Y_test, pred, mods)
        np.savetxt(res_dir / f"confusion_{name}_all_snrs.csv",
                   conf, delimiter=",")
        with open(res_dir / f"confusion_{name}_all_snrs.dat", "wb") as f:
            pickle.dump(conf, f)
        plot_confusion_matrix(
            conf,
            labels=mods,
            title=f"11-class Confusion — {name}",
            save_filename=str(fig_dir / f"confusion_{name}_all_snrs.png"),
        )

    per_snr_rows = []
    with open(res_dir / "normal_diff_manual_hybrid_acc_per_snr.csv",
              "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr",
            "n_test",
            "normal_attention_accuracy",
            "diff_attention_accuracy",
            "manual_hybrid_accuracy",
            "manual_hybrid_minus_normal",
            "manual_hybrid_selected_model",
        ])
        for snr in sorted(snrs):
            mask = test_snrs == snr
            if not np.any(mask):
                continue
            selected = "diff_attention" if snr <= args.threshold_db else "normal_attention"
            row = {
                "snr": int(snr),
                "n_test": int(np.sum(mask)),
                "normal_attention_accuracy": _accuracy(Y_test[mask], normal_pred[mask]),
                "diff_attention_accuracy": _accuracy(Y_test[mask], diff_pred[mask]),
                "manual_hybrid_accuracy": _accuracy(Y_test[mask], hybrid_pred[mask]),
                "manual_hybrid_selected_model": selected,
            }
            row["manual_hybrid_minus_normal"] = (
                row["manual_hybrid_accuracy"] - row["normal_attention_accuracy"]
            )
            per_snr_rows.append(row)
            writer.writerow([
                row["snr"],
                row["n_test"],
                row["normal_attention_accuracy"],
                row["diff_attention_accuracy"],
                row["manual_hybrid_accuracy"],
                row["manual_hybrid_minus_normal"],
                row["manual_hybrid_selected_model"],
            ])

    _plot_threeway_acc(per_snr_rows, fig_dir)

    acc_hybrid = {r["snr"]: r["manual_hybrid_accuracy"] for r in per_snr_rows}
    acc_normal = {r["snr"]: r["normal_attention_accuracy"] for r in per_snr_rows}
    acc_diff = {r["snr"]: r["diff_attention_accuracy"] for r in per_snr_rows}
    with open(res_dir / "acc_manual_hybrid.dat", "wb") as f:
        pickle.dump(acc_hybrid, f)
    with open(res_dir / "acc_normal_attention.dat", "wb") as f:
        pickle.dump(acc_normal, f)
    with open(res_dir / "acc_diff_attention.dat", "wb") as f:
        pickle.dump(acc_diff, f)

    print("\n[11class-hybrid] Summary")
    for name, pred in rows:
        print(f"  {name:28s}: {100 * _accuracy(Y_test, pred):.2f}%")
    print(f"  Results saved under           : {out_dir}/")


if __name__ == "__main__":
    main()
