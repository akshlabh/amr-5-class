"""
evaluate_gated_hybrid_snr_aware_attention.py
============================================

Automatic gated hybrid attention evaluation.

Why this exists
---------------
The manual hybrid needs the true SNR label:

    SNR <= 0 dB -> differential attention
    SNR >  0 dB -> normal attention

In a real receiver we may not have that SNR label.  This script replaces the
manual SNR rule with the trained lightweight SNR gate:

    IQ signal -> SNR gate -> low/high route -> diff/normal attention

It compares:
    1. normal attention baseline
    2. differential attention baseline
    3. manual hybrid using true SNR threshold
    4. gated hybrid using predicted SNR region
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
        description="Evaluate gated automatic hybrid attention vs manual hybrid."
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
        "--gate-weights",
        default="experiments/snr_gate_lightweight/checkpoints/best_model.weights.h5",
    )
    p.add_argument(
        "--gate-scaler",
        default="experiments/snr_gate_lightweight/results/snr_gate_feature_scaler.npz",
    )
    p.add_argument(
        "--gate-metadata",
        default="experiments/snr_gate_lightweight/results/snr_gate_metadata.json",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_gated_hybrid_snr_aware_attention",
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


def _binary_confusion(y_true: np.ndarray,
                      y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    conf_norm = np.zeros_like(conf, dtype=np.float32)
    row_sum = conf.sum(axis=1, keepdims=True)
    np.divide(conf, row_sum, out=conf_norm, where=row_sum > 0)
    return conf, conf_norm


def _plot_binary_confusion(conf_norm: np.ndarray, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["low route\n(diff)", "high route\n(normal)"]
    plt.figure(figsize=(5.4, 4.4), dpi=200)
    plt.imshow(conf_norm * 100, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(label="Percent (%)")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted by SNR gate")
    plt.ylabel("Manual true-SNR route")
    plt.title("SNR Gate Routing Confusion")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{conf_norm[i, j] * 100:.1f}%",
                     ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_fourway_acc(per_snr_rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [r["snr"] for r in per_snr_rows]
    normal = [100 * r["normal_attention_accuracy"] for r in per_snr_rows]
    diff = [100 * r["diff_attention_accuracy"] for r in per_snr_rows]
    manual = [100 * r["manual_hybrid_accuracy"] for r in per_snr_rows]
    gated = [100 * r["gated_hybrid_accuracy"] for r in per_snr_rows]

    plt.figure(figsize=(12, 6))
    plt.plot(snrs, normal, marker="o", linewidth=2.2, label="Normal attention")
    plt.plot(snrs, diff, marker="s", linewidth=2.2, label="Differential attention")
    plt.plot(snrs, manual, marker="^", linewidth=2.4, label="Manual hybrid (true SNR)")
    plt.plot(snrs, gated, marker="D", linewidth=2.4, label="Gated hybrid (predicted SNR)")
    plt.axvline(0, color="black", linestyle="--", alpha=0.55, label="0 dB route boundary")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Manual Hybrid vs Gated Hybrid SNR-aware Attention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig(fig_dir / "manual_vs_gated_hybrid_acc_vs_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    delta = [g - m for g, m in zip(gated, manual)]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    plt.figure(figsize=(11, 4.8))
    plt.bar([str(s) for s in snrs], delta, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Gated - manual accuracy (percentage points)")
    plt.title("Impact of Replacing True SNR with Predicted SNR Gate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "gated_minus_manual_delta_by_snr.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()

    import keras
    from src.dataset import FIVE_CLASS, load_data
    from src.features.signal_features import extract_snr_gate_features
    from src.models.mcldnn_attention import build_mcldnn_attention
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention
    from src.models.snr_gate import build_snr_gate
    from src.utils.mltools import calculate_confusion_matrix, plot_confusion_matrix

    keras.mixed_precision.set_global_policy("float32")

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "normal weights": Path(args.normal_weights),
        "diff weights": Path(args.diff_weights),
        "gate weights": Path(args.gate_weights),
        "gate scaler": Path(args.gate_scaler),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    gate_meta_path = Path(args.gate_metadata)
    gate_meta = {}
    if gate_meta_path.exists():
        gate_meta = json.loads(gate_meta_path.read_text(encoding="utf-8"))

    print("\n============================================================")
    print("  Experiment : gated_hybrid_snr_aware_attention")
    print("  Manual     : true SNR <= 0 dB -> diff, > 0 dB -> normal")
    print("  Gated      : SNR-gate prediction chooses diff/normal")
    print(f"  Threshold  : {args.threshold_db:g} dB")
    print(f"  Output dir : {out_dir}")
    print("============================================================\n")

    (mods, snrs, lbl), _, _, (X_test, Y_test), (_, _, test_idx) = load_data(
        args.datasetpath,
        FIVE_CLASS,
        seed=args.seed,
        shuffle_split=False,
    )
    test_snrs = np.array([lbl[i][1] for i in test_idx], dtype=np.int32)
    inputs_test = _make_iq_inputs(X_test)

    normal_model = build_mcldnn_attention(classes=len(FIVE_CLASS))
    normal_model.load_weights(args.normal_weights)
    print("[gated-hybrid] Predicting normal attention...")
    normal_pred = normal_model.predict(inputs_test,
                                       batch_size=args.batch_size,
                                       verbose=1)

    diff_model = build_mcldnn_diffattention(classes=len(FIVE_CLASS))
    diff_model.load_weights(args.diff_weights)
    print("[gated-hybrid] Predicting differential attention...")
    diff_pred = diff_model.predict(inputs_test,
                                   batch_size=args.batch_size,
                                   verbose=1)

    # Manual route uses true SNR labels.
    manual_use_diff = test_snrs <= args.threshold_db

    # Gated route uses learned SNR-region prediction.
    scaler = np.load(args.gate_scaler)
    gate_features, gate_feature_names = extract_snr_gate_features(X_test)
    gate_features = ((gate_features - scaler["mean"]) / scaler["std"]).astype("float32")
    gate_model = build_snr_gate(input_dim=gate_features.shape[1])
    gate_model.load_weights(args.gate_weights)
    print("[gated-hybrid] Predicting SNR gate route...")
    gate_prob = gate_model.predict(gate_features,
                                   batch_size=args.batch_size,
                                   verbose=1)
    gate_route = np.argmax(gate_prob, axis=1)
    gated_use_diff = gate_route == 0

    manual_pred = normal_pred.copy()
    manual_pred[manual_use_diff] = diff_pred[manual_use_diff]

    gated_pred = normal_pred.copy()
    gated_pred[gated_use_diff] = diff_pred[gated_use_diff]

    y_route_true = np.where(manual_use_diff, 0, 1)
    route_conf, route_conf_norm = _binary_confusion(y_route_true, gate_route)
    route_acc = float(np.mean(y_route_true == gate_route))

    print(f"[gated-hybrid] Manual diff-route samples : {int(manual_use_diff.sum())}")
    print(f"[gated-hybrid] Gated diff-route samples  : {int(gated_use_diff.sum())}")
    print(f"[gated-hybrid] SNR gate route accuracy   : {100 * route_acc:.2f}%")

    rows = [
        ("normal_attention", normal_pred),
        ("diff_attention", diff_pred),
        ("manual_hybrid_true_snr", manual_pred),
        ("gated_hybrid_predicted_snr", gated_pred),
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

    with open(res_dir / "routing_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["threshold_db", args.threshold_db])
        writer.writerow(["snr_gate_route_accuracy", route_acc])
        writer.writerow(["manual_diff_route_count", int(manual_use_diff.sum())])
        writer.writerow(["manual_normal_route_count", int((~manual_use_diff).sum())])
        writer.writerow(["gated_diff_route_count", int(gated_use_diff.sum())])
        writer.writerow(["gated_normal_route_count", int((~gated_use_diff).sum())])

    with open(res_dir / "snr_gate_route_confusion_raw.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", "pred_low_diff", "pred_high_normal"])
        writer.writerow(["true_low_diff", int(route_conf[0, 0]), int(route_conf[0, 1])])
        writer.writerow(["true_high_normal", int(route_conf[1, 0]), int(route_conf[1, 1])])

    with open(res_dir / "snr_gate_route_confusion_normalized.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", "pred_low_diff", "pred_high_normal"])
        writer.writerow(["true_low_diff", float(route_conf_norm[0, 0]), float(route_conf_norm[0, 1])])
        writer.writerow(["true_high_normal", float(route_conf_norm[1, 0]), float(route_conf_norm[1, 1])])

    _plot_binary_confusion(route_conf_norm, fig_dir / "snr_gate_route_confusion.png")

    # Modulation confusion matrices for manual and gated hybrids.
    conf_manual, _, _ = calculate_confusion_matrix(Y_test, manual_pred, mods)
    conf_gated, _, _ = calculate_confusion_matrix(Y_test, gated_pred, mods)
    conf_delta = conf_gated - conf_manual
    np.savetxt(res_dir / "confusion_manual_hybrid_all_snrs.csv", conf_manual, delimiter=",")
    np.savetxt(res_dir / "confusion_gated_hybrid_all_snrs.csv", conf_gated, delimiter=",")
    np.savetxt(res_dir / "confusion_gated_minus_manual_all_snrs.csv", conf_delta, delimiter=",")
    with open(res_dir / "confusion_manual_hybrid_all_snrs.dat", "wb") as f:
        pickle.dump(conf_manual, f)
    with open(res_dir / "confusion_gated_hybrid_all_snrs.dat", "wb") as f:
        pickle.dump(conf_gated, f)

    plot_confusion_matrix(
        conf_manual,
        labels=mods,
        title="Manual Hybrid Confusion — true SNR route",
        save_filename=str(fig_dir / "confusion_manual_hybrid_all_snrs.png"),
    )
    plot_confusion_matrix(
        conf_gated,
        labels=mods,
        title="Gated Hybrid Confusion — predicted SNR route",
        save_filename=str(fig_dir / "confusion_gated_hybrid_all_snrs.png"),
    )

    per_snr_rows = []
    with open(res_dir / "manual_vs_gated_acc_per_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr",
            "n_test",
            "normal_attention_accuracy",
            "diff_attention_accuracy",
            "manual_hybrid_accuracy",
            "gated_hybrid_accuracy",
            "gated_minus_manual",
            "snr_gate_route_accuracy",
            "manual_diff_route_fraction",
            "gated_diff_route_fraction",
            "mean_gate_p_low",
        ])
        for snr in sorted(snrs):
            mask = test_snrs == snr
            if not np.any(mask):
                continue
            row = {
                "snr": int(snr),
                "n_test": int(np.sum(mask)),
                "normal_attention_accuracy": _accuracy(Y_test[mask], normal_pred[mask]),
                "diff_attention_accuracy": _accuracy(Y_test[mask], diff_pred[mask]),
                "manual_hybrid_accuracy": _accuracy(Y_test[mask], manual_pred[mask]),
                "gated_hybrid_accuracy": _accuracy(Y_test[mask], gated_pred[mask]),
                "snr_gate_route_accuracy": float(np.mean(y_route_true[mask] == gate_route[mask])),
                "manual_diff_route_fraction": float(np.mean(manual_use_diff[mask])),
                "gated_diff_route_fraction": float(np.mean(gated_use_diff[mask])),
                "mean_gate_p_low": float(np.mean(gate_prob[mask, 0])),
            }
            row["gated_minus_manual"] = (
                row["gated_hybrid_accuracy"] - row["manual_hybrid_accuracy"]
            )
            per_snr_rows.append(row)
            writer.writerow([
                row["snr"],
                row["n_test"],
                row["normal_attention_accuracy"],
                row["diff_attention_accuracy"],
                row["manual_hybrid_accuracy"],
                row["gated_hybrid_accuracy"],
                row["gated_minus_manual"],
                row["snr_gate_route_accuracy"],
                row["manual_diff_route_fraction"],
                row["gated_diff_route_fraction"],
                row["mean_gate_p_low"],
            ])

    _plot_fourway_acc(per_snr_rows, fig_dir)

    with open(res_dir / "gated_hybrid_rule.txt", "w", encoding="utf-8") as f:
        f.write("Gated hybrid SNR-aware attention rule\n")
        f.write("=====================================\n")
        f.write(f"Manual reference: true SNR <= {args.threshold_db:g} dB -> diff attention; otherwise normal.\n")
        f.write("Gated hybrid: trained SNR gate predicts route 0/1 from IQ features.\n")
        f.write("Route 0 -> diff attention; route 1 -> normal attention.\n")
        f.write(f"Normal checkpoint: {args.normal_weights}\n")
        f.write(f"Diff checkpoint  : {args.diff_weights}\n")
        f.write(f"Gate checkpoint  : {args.gate_weights}\n")
        f.write(f"Gate scaler      : {args.gate_scaler}\n")
        if gate_meta:
            f.write("\nGate metadata:\n")
            f.write(json.dumps(gate_meta, indent=2))

    print("\n[gated-hybrid] Summary")
    for name, pred in rows:
        print(f"  {name:28s}: {100 * _accuracy(Y_test, pred):.2f}%")
    print(f"  SNR gate route accuracy       : {100 * route_acc:.2f}%")
    print(f"  Results saved under           : {out_dir}/")


if __name__ == "__main__":
    main()
