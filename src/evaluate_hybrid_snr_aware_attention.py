"""
evaluate_hybrid_snr_aware_attention.py
======================================

Hybrid SNR-aware attention evaluation.

Research idea
-------------
The plain differential-attention model was not better everywhere, but it gave
useful gains in the low/transition SNR region.  The normal attention model was
stronger at cleaner SNRs.

So this script builds an inference-time hybrid:

    if -8 dB <= SNR <= +2 dB:
        use plain differential-attention prediction
    else:
        use normal-attention prediction

Then it compares the hybrid against the normal-attention baseline on the same
test split.

This is not a new trained model.  It is a deterministic SNR-aware ensemble
using two already-trained checkpoints:

    experiments/5class_attention/checkpoints/best_model.weights.h5
    experiments/5class_diffattention/checkpoints/best_model.weights.h5
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np


# When invoked as `python src/evaluate_hybrid_snr_aware_attention.py`, make
# repo-root imports work.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["KERAS_BACKEND"] = "tensorflow"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Hybrid SNR-aware Attention vs normal attention."
    )
    p.add_argument(
        "--datasetpath",
        required=True,
        help="Path to RML2016.10a_dict.dat / .pkl dataset.",
    )
    p.add_argument(
        "--normal-weights",
        default="experiments/5class_attention/checkpoints/best_model.weights.h5",
        help="Normal attention checkpoint.",
    )
    p.add_argument(
        "--diff-weights",
        default="experiments/5class_diffattention/checkpoints/best_model.weights.h5",
        help="Plain differential attention checkpoint.",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/5class_hybrid_snr_aware_attention",
        help="Output experiment directory.",
    )
    p.add_argument(
        "--diff-snr-min",
        type=int,
        default=-8,
        help="Minimum SNR using differential attention.",
    )
    p.add_argument(
        "--diff-snr-max",
        type=int,
        default=2,
        help="Maximum SNR using differential attention.",
    )
    p.add_argument(
        "--normal-at-upper-boundary",
        action="store_true",
        help=(
            "Use normal attention at diff_snr_max itself. "
            "Default is inclusive: diff_snr_min <= SNR <= diff_snr_max."
        ),
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


def _categorical_crossentropy_np(Y: np.ndarray,
                                 Y_hat: np.ndarray,
                                 eps: float = 1e-7) -> float:
    Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
    return float(-np.mean(np.sum(Y * np.log(Y_hat), axis=1)))


def _accuracy(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    return float(np.mean(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)))


def _hybrid_mask(test_snrs: np.ndarray,
                 diff_snr_min: int,
                 diff_snr_max: int,
                 normal_at_upper_boundary: bool) -> np.ndarray:
    if normal_at_upper_boundary:
        return (test_snrs >= diff_snr_min) & (test_snrs < diff_snr_max)
    return (test_snrs >= diff_snr_min) & (test_snrs <= diff_snr_max)


def main():
    args = _parse_args()

    import keras
    from src.dataset import FIVE_CLASS, load_data
    from src.models.mcldnn_attention import build_mcldnn_attention
    from src.models.mcldnn_diffattention import build_mcldnn_diffattention
    from src.utils.mltools import (
        calculate_confusion_matrix,
        plot_acc_per_class_vs_snr,
        plot_acc_vs_snr,
        plot_confusion_matrix,
    )

    keras.mixed_precision.set_global_policy("float32")

    out_dir = Path(args.output_dir)
    ckpt_normal = Path(args.normal_weights)
    ckpt_diff = Path(args.diff_weights)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_normal.exists():
        raise FileNotFoundError(f"Normal attention checkpoint not found: {ckpt_normal}")
    if not ckpt_diff.exists():
        raise FileNotFoundError(f"Diff-attention checkpoint not found: {ckpt_diff}")

    print("\n============================================================")
    print("  Experiment : 5class_hybrid_snr_aware_attention")
    print("  Rule       : use diff-attention in selected low-SNR band")
    print(f"  Diff band  : {args.diff_snr_min} dB to {args.diff_snr_max} dB "
          f"({'upper exclusive' if args.normal_at_upper_boundary else 'inclusive'})")
    print(f"  Normal ckpt: {ckpt_normal}")
    print(f"  Diff ckpt  : {ckpt_diff}")
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
    normal_model.load_weights(ckpt_normal)
    print("[hybrid] Predicting normal attention...")
    normal_pred = normal_model.predict(inputs_test,
                                       batch_size=args.batch_size,
                                       verbose=1)

    diff_model = build_mcldnn_diffattention(classes=len(FIVE_CLASS))
    diff_model.load_weights(ckpt_diff)
    print("[hybrid] Predicting differential attention...")
    diff_pred = diff_model.predict(inputs_test,
                                   batch_size=args.batch_size,
                                   verbose=1)

    use_diff = _hybrid_mask(
        test_snrs,
        diff_snr_min=args.diff_snr_min,
        diff_snr_max=args.diff_snr_max,
        normal_at_upper_boundary=args.normal_at_upper_boundary,
    )
    hybrid_pred = normal_pred.copy()
    hybrid_pred[use_diff] = diff_pred[use_diff]

    print(f"[hybrid] Test samples using diff attention  : {int(use_diff.sum())}")
    print(f"[hybrid] Test samples using normal attention: {int((~use_diff).sum())}")

    normal_loss = _categorical_crossentropy_np(Y_test, normal_pred)
    hybrid_loss = _categorical_crossentropy_np(Y_test, hybrid_pred)
    normal_acc = _accuracy(Y_test, normal_pred)
    hybrid_acc = _accuracy(Y_test, hybrid_pred)

    with open(res_dir / "test_score.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "loss", "accuracy"])
        writer.writerow(["normal_attention", normal_loss, normal_acc])
        writer.writerow(["hybrid_snr_aware_attention", hybrid_loss, hybrid_acc])

    with open(res_dir / "hybrid_rule.txt", "w", encoding="utf-8") as f:
        f.write("Hybrid SNR-aware attention rule\n")
        f.write("================================\n")
        f.write(f"Use differential attention when SNR >= {args.diff_snr_min} and ")
        if args.normal_at_upper_boundary:
            f.write(f"SNR < {args.diff_snr_max}.\n")
        else:
            f.write(f"SNR <= {args.diff_snr_max}.\n")
        f.write("Use normal attention otherwise.\n")
        f.write(f"Normal checkpoint: {ckpt_normal}\n")
        f.write(f"Diff checkpoint  : {ckpt_diff}\n")

    conf_normal, _, _ = calculate_confusion_matrix(Y_test, normal_pred, mods)
    conf_hybrid, _, _ = calculate_confusion_matrix(Y_test, hybrid_pred, mods)
    conf_delta = conf_hybrid - conf_normal

    with open(res_dir / "confusion_normal_all_snrs.dat", "wb") as f:
        pickle.dump(conf_normal, f)
    with open(res_dir / "confusion_hybrid_all_snrs.dat", "wb") as f:
        pickle.dump(conf_hybrid, f)
    np.savetxt(res_dir / "confusion_normal_all_snrs.csv",
               conf_normal, delimiter=",")
    np.savetxt(res_dir / "confusion_hybrid_all_snrs.csv",
               conf_hybrid, delimiter=",")
    np.savetxt(res_dir / "confusion_hybrid_minus_normal_all_snrs.csv",
               conf_delta, delimiter=",")

    plot_confusion_matrix(
        conf_normal,
        labels=mods,
        title="Confusion Matrix — Normal Attention baseline (all SNRs)",
        save_filename=str(fig_dir / "confusion_normal_all_snrs.png"),
    )
    plot_confusion_matrix(
        conf_hybrid,
        labels=mods,
        title="Confusion Matrix — Hybrid SNR-aware Attention (all SNRs)",
        save_filename=str(fig_dir / "confusion_all_snrs.png"),
    )

    acc_normal = {}
    acc_hybrid = {}
    acc_mod_snr = np.zeros((len(mods), len(snrs)))

    with open(res_dir / "acc_per_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr",
            "normal_attention_accuracy",
            "hybrid_accuracy",
            "delta_hybrid_minus_normal",
            "selected_model",
        ])

        for i, snr in enumerate(snrs):
            mask = test_snrs == snr
            selected = "diff_attention" if bool(use_diff[mask][0]) else "normal_attention"

            Yn = Y_test[mask]
            normal_snr_pred = normal_pred[mask]
            hybrid_snr_pred = hybrid_pred[mask]

            conf_i, cor, ncor = calculate_confusion_matrix(Yn, hybrid_snr_pred, mods)
            acc_hybrid[snr] = cor / (cor + ncor)
            acc_normal[snr] = _accuracy(Yn, normal_snr_pred)
            acc_mod_snr[:, i] = np.round(
                np.diag(conf_i) / np.sum(conf_i, axis=1), 4
            )

            writer.writerow([
                snr,
                acc_normal[snr],
                acc_hybrid[snr],
                acc_hybrid[snr] - acc_normal[snr],
                selected,
            ])

            plot_confusion_matrix(
                conf_i,
                labels=mods,
                title=(
                    f"Hybrid Confusion (SNR={snr} dB, "
                    f"acc={100 * acc_hybrid[snr]:.1f}%, source={selected})"
                ),
                save_filename=str(fig_dir / f"confusion_snr{snr:+03d}.png"),
            )

    with open(res_dir / "acc.dat", "wb") as f:
        pickle.dump(acc_hybrid, f)
    with open(res_dir / "acc_normal_baseline.dat", "wb") as f:
        pickle.dump(acc_normal, f)
    with open(res_dir / "acc_for_mod.dat", "wb") as f:
        pickle.dump(acc_mod_snr, f)

    plot_acc_vs_snr(
        acc_hybrid,
        title="Overall Accuracy vs SNR — Hybrid SNR-aware Attention",
        save_filename=str(fig_dir / "acc_vs_snr.png"),
    )
    plot_acc_per_class_vs_snr(acc_mod_snr, mods, snrs, save_dir=str(fig_dir))

    # Comparison plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_values = sorted(snrs)
    normal_vals = [100 * acc_normal[s] for s in snr_values]
    hybrid_vals = [100 * acc_hybrid[s] for s in snr_values]
    delta_vals = [h - n for h, n in zip(hybrid_vals, normal_vals)]

    plt.figure(figsize=(11, 6))
    plt.plot(snr_values, normal_vals, marker="o", linewidth=2.4,
             label="Normal attention baseline")
    plt.plot(snr_values, hybrid_vals, marker="s", linewidth=2.4,
             label="Hybrid SNR-aware attention")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Hybrid SNR-aware Attention vs Normal Attention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(snr_values)
    plt.tight_layout()
    plt.savefig(fig_dir / "hybrid_vs_normal_acc_vs_snr.png",
                dpi=180, bbox_inches="tight")
    plt.close()

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_vals]
    plt.figure(figsize=(11, 4.8))
    plt.bar([str(s) for s in snr_values], delta_vals, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta accuracy (percentage points)")
    plt.title("Hybrid minus Normal Attention by SNR")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "hybrid_minus_normal_delta_by_snr.png",
                dpi=180, bbox_inches="tight")
    plt.close()

    print("\n[hybrid] Summary")
    print(f"  Normal attention accuracy : {100 * normal_acc:.2f}%")
    print(f"  Hybrid attention accuracy : {100 * hybrid_acc:.2f}%")
    print(f"  Delta                     : {100 * (hybrid_acc - normal_acc):+.2f} p.p.")
    print(f"  Results saved under       : {out_dir}/")


if __name__ == "__main__":
    main()
