"""
train_snr_gate.py — Train a lightweight SNR-aware routing model
================================================================

Purpose
-------
Train a binary classifier that decides which attention model should be used:

    low SNR  (SNR <= threshold, default 0 dB) -> differential attention
    high SNR (SNR >  threshold)               -> normal attention

The model uses compact IQ-derived analytic features, not modulation labels.
It preserves the same dataset split logic used elsewhere in the repository.

Usage
-----
    python src/train_snr_gate.py --config configs/exp_snr_gate.yaml

Kaggle dataset override:
    python src/train_snr_gate.py --config configs/exp_snr_gate.yaml \
        --datasetpath /kaggle/input/rml201610a-dict/RML2016.10a_dict.dat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import yaml


def _parse_args():
    p = argparse.ArgumentParser(description="Train binary SNR-region gate")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--datasetpath", default=None, help="Override dataset path")
    p.add_argument("--resume", default=None, help="Optional weights path to resume")
    return p.parse_args()


def _onehot_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((len(y), 2), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    conf_norm = np.zeros_like(conf, dtype=np.float32)
    row_sum = conf.sum(axis=1, keepdims=True)
    np.divide(conf, row_sum, out=conf_norm, where=row_sum > 0)
    return conf, conf_norm


def _plot_history(history, log_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history.history["loss"], label="train loss")
    plt.plot(epochs, history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SNR Gate Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "snr_gate_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history.history["accuracy"], label="train accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SNR Gate Training Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "snr_gate_accuracy.png", dpi=200)
    plt.close()


def _plot_confusion(conf_norm: np.ndarray, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["low SNR\n<=0 dB", "high SNR\n>0 dB"]
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.2), dpi=200)
    plt.imshow(conf_norm * 100, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(label="Percent (%)")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted route")
    plt.ylabel("True SNR region")
    plt.title("SNR Gate Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{conf_norm[i, j] * 100:.1f}%",
                     ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_per_snr(per_snr_rows: list[dict], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    snrs = [r["snr"] for r in per_snr_rows]
    acc = [r["accuracy"] for r in per_snr_rows]
    p_low = [r["mean_p_low"] for r in per_snr_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(snrs, np.array(acc) * 100, marker="o", linewidth=2)
    plt.axvline(0, linestyle="--", color="black", alpha=0.6, label="routing threshold")
    plt.xlabel("True SNR (dB)")
    plt.ylabel("Binary route accuracy (%)")
    plt.title("SNR Gate Accuracy vs True SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "snr_gate_accuracy_vs_snr.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(snrs, p_low, marker="o", linewidth=2, label="mean P(low SNR)")
    plt.axvline(0, linestyle="--", color="black", alpha=0.6, label="0 dB threshold")
    plt.axhline(0.5, linestyle=":", color="gray", alpha=0.8, label="decision threshold")
    plt.xlabel("True SNR (dB)")
    plt.ylabel("Mean predicted probability")
    plt.title("SNR Gate Mean Low-SNR Probability vs True SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "snr_gate_p_low_vs_snr.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = _parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"].get("seed", 2016))
    from src.utils.seed import set_all_seeds
    set_all_seeds(seed)

    import keras
    from src.dataset import load_data
    from src.features.signal_features import extract_snr_gate_features
    from src.models.snr_gate import build_snr_gate

    exp_name = cfg["experiment"]["name"]
    data_path = args.datasetpath if args.datasetpath else cfg["dataset"]["path"]
    selected_classes = cfg["dataset"]["classes"]
    threshold_db = float(cfg.get("threshold_db", 0.0))
    shuffle_split = bool(cfg.get("shuffle_split", False))
    split_file = cfg.get("split_file", None)
    snr_range = cfg.get("snr_range", None)
    if snr_range is not None:
        snr_range = tuple(snr_range)

    tr = cfg["training"]
    epochs = int(tr["epochs"])
    batch_size = int(tr["batch_size"])
    dropout_rate = float(tr.get("dropout_rate", 0.15))
    initial_lr = float(tr.get("initial_lr", 1e-3))
    es_patience = int(tr.get("early_stopping_patience", 20))
    rlr_patience = int(tr.get("reduce_lr_patience", 8))
    rlr_factor = float(tr.get("reduce_lr_factor", 0.5))
    min_lr = float(tr.get("min_lr", 1e-7))

    out = cfg["output"]
    base_dir = Path(out["base_dir"])
    ckpt_path = Path(out["checkpoint_path"])
    log_dir = Path(out["log_dir"])
    fig_dir = Path(out["figure_dir"])
    res_dir = Path(out["result_dir"])
    for d in [ckpt_path.parent, log_dir, fig_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Experiment : {exp_name}")
    print(f"Task       : binary SNR routing")
    print(f"Rule       : class 0 low <= {threshold_db:g} dB, class 1 high > {threshold_db:g} dB")
    print(f"Dataset    : {data_path}")
    print(f"Classes    : {selected_classes}")
    print(f"Split      : shuffle_split={shuffle_split}, split_file={split_file}")
    print(f"Checkpoint : {ckpt_path}")
    print("=" * 70)

    (mods, snrs, lbl), \
    (X_train, _), \
    (X_val, _), \
    (X_test, _), \
    (train_idx, val_idx, test_idx) = load_data(
        data_path,
        selected_classes,
        seed=seed,
        split_file=split_file,
        snr_range=snr_range,
        shuffle_split=shuffle_split,
    )

    y_train_raw = np.array([0 if lbl[i][1] <= threshold_db else 1 for i in train_idx], dtype=np.int64)
    y_val_raw = np.array([0 if lbl[i][1] <= threshold_db else 1 for i in val_idx], dtype=np.int64)
    y_test_raw = np.array([0 if lbl[i][1] <= threshold_db else 1 for i in test_idx], dtype=np.int64)
    Y_train = _onehot_binary(y_train_raw)
    Y_val = _onehot_binary(y_val_raw)
    Y_test = _onehot_binary(y_test_raw)

    Xf_train, feature_names = extract_snr_gate_features(X_train)
    Xf_val, _ = extract_snr_gate_features(X_val)
    Xf_test, _ = extract_snr_gate_features(X_test)

    mean, std = _standardize_fit(Xf_train)
    Xf_train = _standardize_apply(Xf_train, mean, std)
    Xf_val = _standardize_apply(Xf_val, mean, std)
    Xf_test = _standardize_apply(Xf_test, mean, std)

    np.savez(res_dir / "snr_gate_feature_scaler.npz", mean=mean, std=std)
    with open(res_dir / "snr_gate_feature_names.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "feature_name"])
        for i, name in enumerate(feature_names):
            w.writerow([i, name])

    model = build_snr_gate(
        input_dim=Xf_train.shape[1],
        classes=2,
        dropout_rate=dropout_rate,
        learning_rate=initial_lr,
    )
    if args.resume:
        model.load_weights(args.resume)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=rlr_factor,
            patience=rlr_patience,
            min_lr=min_lr,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=es_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_dir / "training_log.csv"), append=bool(args.resume)),
    ]

    history = model.fit(
        Xf_train,
        Y_train,
        validation_data=(Xf_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    _plot_history(history, log_dir)

    score = model.evaluate(Xf_test, Y_test, batch_size=batch_size, verbose=1)
    print(f"[snr_gate] Test loss={score[0]:.5f}, test accuracy={score[1]:.5f}")

    prob_test = model.predict(Xf_test, batch_size=batch_size, verbose=1)
    pred_test = np.argmax(prob_test, axis=1)
    conf, conf_norm = _confusion(y_test_raw, pred_test)
    _plot_confusion(conf_norm, fig_dir / "snr_gate_confusion.png")

    with open(res_dir / "test_score.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loss", "accuracy"])
        w.writerow([float(score[0]), float(score[1])])

    with open(res_dir / "confusion_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred", "pred_low_snr", "pred_high_snr"])
        w.writerow(["true_low_snr", int(conf[0, 0]), int(conf[0, 1])])
        w.writerow(["true_high_snr", int(conf[1, 0]), int(conf[1, 1])])

    with open(res_dir / "confusion_normalized.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred", "pred_low_snr", "pred_high_snr"])
        w.writerow(["true_low_snr", float(conf_norm[0, 0]), float(conf_norm[0, 1])])
        w.writerow(["true_high_snr", float(conf_norm[1, 0]), float(conf_norm[1, 1])])

    test_snrs = np.array([lbl[i][1] for i in test_idx])
    per_snr_rows = []
    with open(res_dir / "snr_gate_acc_per_snr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr", "n_test", "accuracy", "mean_p_low", "mean_p_high", "low_route_fraction"])
        for snr in sorted(snrs):
            mask = test_snrs == snr
            if not np.any(mask):
                continue
            acc = float(np.mean(pred_test[mask] == y_test_raw[mask]))
            mean_p_low = float(np.mean(prob_test[mask, 0]))
            mean_p_high = float(np.mean(prob_test[mask, 1]))
            low_route_fraction = float(np.mean(pred_test[mask] == 0))
            row = {
                "snr": int(snr),
                "n_test": int(np.sum(mask)),
                "accuracy": acc,
                "mean_p_low": mean_p_low,
                "mean_p_high": mean_p_high,
                "low_route_fraction": low_route_fraction,
            }
            per_snr_rows.append(row)
            w.writerow([
                row["snr"],
                row["n_test"],
                row["accuracy"],
                row["mean_p_low"],
                row["mean_p_high"],
                row["low_route_fraction"],
            ])

    _plot_per_snr(per_snr_rows, fig_dir)

    metadata = {
        "experiment": exp_name,
        "threshold_db": threshold_db,
        "class_mapping": {
            "0": f"low_snr <= {threshold_db:g} dB -> differential attention",
            "1": f"high_snr > {threshold_db:g} dB -> normal attention",
        },
        "feature_dim": int(Xf_train.shape[1]),
        "feature_names": feature_names,
        "test_accuracy": float(score[1]),
        "checkpoint_path": str(ckpt_path),
        "scaler_path": str(res_dir / "snr_gate_feature_scaler.npz"),
    }
    with open(res_dir / "snr_gate_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[snr_gate] Outputs saved under {base_dir}/")
    print("[snr_gate] Routing labels:")
    print(f"  0 -> SNR <= {threshold_db:g} dB -> differential attention")
    print(f"  1 -> SNR >  {threshold_db:g} dB -> normal attention")


if __name__ == "__main__":
    main()
