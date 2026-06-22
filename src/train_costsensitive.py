"""
train_costsensitive.py — Attention model training with cost-sensitive cross-entropy
====================================================================================
Usage
-----
    python src/train_costsensitive.py \\
        --config configs/exp_5class_attention_costsensitive.yaml

    # Kaggle / override dataset path:
    python src/train_costsensitive.py \\
        --config configs/exp_5class_attention_costsensitive.yaml \\
        --datasetpath /kaggle/input/.../RML2016.10a_5class.pkl

What this script does differently from train.py
------------------------------------------------
1. Reads a `cost_matrix` block from the YAML config.
2. Builds the cost matrix using build_cost_matrix() from src/losses/cost_sensitive.py.
3. Builds the model exactly as before via build_mcldnn_attention().
4. Recompiles the model with CostSensitiveCrossEntropy (overrides the default CE
   compiled inside build_mcldnn_attention — Keras fully supports recompilation).
5. Saves a cost_matrix_config.json to the results directory for reproducibility.
6. Everything else (data loading, callbacks, evaluation, plots) is IDENTICAL to train.py.

Note: Accuracy metric is unaffected by the custom loss — it always uses argmax,
      so val_accuracy in callbacks and logs is directly comparable to other runs.
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
import json
import pickle
import numpy as np
import yaml


def _parse_args():
    p = argparse.ArgumentParser(
        description='Train MCLDNN-Attention with cost-sensitive cross-entropy')
    p.add_argument('--config', required=True,
                   help='Path to YAML experiment config file')
    p.add_argument('--datasetpath', default=None,
                   help='Override dataset path in YAML (e.g. Kaggle input path)')
    p.add_argument('--resume', default=None,
                   help='Path to .weights.h5 checkpoint to resume from')
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = _parse_args()
    cfg  = _load_config(args.config)
    seed = cfg['experiment'].get('seed', 2016)

    # ── Seed everything before TF/Keras imports ───────────────────────────────
    from src.utils.seed import set_all_seeds
    set_all_seeds(seed)

    # ── TF / Keras (after seed) ───────────────────────────────────────────────
    import keras
    from src.dataset       import load_data
    from src.utils.mltools import (show_history, calculate_confusion_matrix,
                                   plot_confusion_matrix, plot_acc_vs_snr,
                                   plot_acc_per_class_vs_snr)
    from src.losses.cost_sensitive import (build_cost_matrix,
                                           print_cost_matrix,
                                           CostSensitiveCrossEntropy)
    from src.models.mcldnn_attention import build_mcldnn_attention

    # ── Read config values ────────────────────────────────────────────────────
    exp_name         = cfg['experiment']['name']
    selected_classes = cfg['dataset']['classes']
    data_path        = args.datasetpath or cfg['dataset']['path']

    tr_cfg       = cfg['training']
    nb_epoch     = tr_cfg['epochs']
    batch_size   = tr_cfg['batch_size']
    es_patience  = tr_cfg['early_stopping_patience']
    rlr_patience = tr_cfg['reduce_lr_patience']
    rlr_factor   = tr_cfg['reduce_lr_factor']
    min_lr       = tr_cfg['min_lr']
    dropout_rate = tr_cfg.get('dropout_rate', 0.6)
    initial_lr   = tr_cfg.get('initial_lr', 5e-4)
    class_weights_cfg = tr_cfg.get('class_weights', None)

    # ── Cost matrix config ────────────────────────────────────────────────────
    cm_cfg = cfg.get('cost_matrix', {})
    alpha  = float(cm_cfg.get('alpha', 2.0))
    # Parse high-cost pairs: list of [true_class, pred_class] from YAML
    raw_pairs = cm_cfg.get('high_cost_pairs',
                           [['QAM16', 'QAM64'], ['QAM64', 'QAM16']])
    high_cost_pairs = [tuple(p) for p in raw_pairs]
    label_smoothing = float(cm_cfg.get('label_smoothing', 0.1))

    out_cfg   = cfg['output']
    ckpt_path = out_cfg['checkpoint_path']
    log_dir   = out_cfg['log_dir']
    fig_dir   = out_cfg['figure_dir']
    res_dir   = out_cfg['result_dir']

    n_classes = len(selected_classes)

    # Create output directories
    for d in [os.path.dirname(ckpt_path), log_dir, fig_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Build cost matrix ─────────────────────────────────────────────────────
    C = build_cost_matrix(
        classes=selected_classes,
        high_cost_pairs=high_cost_pairs,
        alpha=alpha,
    )

    print(f"\n{'='*60}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Model       : mcldnn_attention  (cost-sensitive loss)")
    print(f"  Dropout     : {dropout_rate}   Initial LR : {initial_lr}")
    print(f"  ES Patience : {es_patience}    RLR Patience: {rlr_patience}")
    print(f"  Alpha (QAM16↔QAM64 penalty) : {alpha}")
    print(f"  Label smoothing : {label_smoothing}")
    print(f"  Classes     : {selected_classes}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"{'='*60}\n")

    print("[cost] Cost matrix:")
    print_cost_matrix(C, selected_classes)
    print()

    # Save cost matrix config for reproducibility
    cm_record = {
        'alpha':            alpha,
        'high_cost_pairs':  [list(p) for p in high_cost_pairs],
        'label_smoothing':  label_smoothing,
        'cost_matrix':      C.tolist(),
    }
    with open(os.path.join(res_dir, 'cost_matrix_config.json'), 'w') as f:
        json.dump(cm_record, f, indent=2)

    # ── Load data ─────────────────────────────────────────────────────────────
    (mods, snrs, lbl), \
    (X_train, Y_train), \
    (X_val,   Y_val),   \
    (X_test,  Y_test),  \
    (train_idx, val_idx, test_idx) = load_data(
        data_path, selected_classes, seed)

    # Prepare 3-input format
    def prepare_inputs(X):
        return [
            np.expand_dims(X, axis=3),           # (N, 2, 128, 1)
            np.expand_dims(X[:, 0, :], axis=2),  # (N, 128, 1) — I channel
            np.expand_dims(X[:, 1, :], axis=2),  # (N, 128, 1) — Q channel
        ]

    inp_train = prepare_inputs(X_train)
    inp_val   = prepare_inputs(X_val)
    inp_test  = prepare_inputs(X_test)

    print(f"[train] Input shapes: "
          f"IQ={inp_train[0].shape}  I={inp_train[1].shape}  Q={inp_train[2].shape}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_mcldnn_attention(
        classes=n_classes,
        dropout_rate=dropout_rate,
        learning_rate=initial_lr,
    )

    # ── Recompile with cost-sensitive loss ────────────────────────────────────
    # build_mcldnn_attention() compiles with standard CategoricalCrossentropy.
    # Calling compile() again replaces the loss — Keras fully supports this.
    cost_loss = CostSensitiveCrossEntropy(C, label_smoothing=label_smoothing)
    model.compile(
        loss=cost_loss,
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),
        metrics=['accuracy'],
    )
    print(f"[train] Recompiled with CostSensitiveCrossEntropy (alpha={alpha})\n")

    if args.resume:
        model.load_weights(args.resume)
        print(f"[train] Resumed from checkpoint: {args.resume}")

    model.summary()

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weight_dict = None
    if class_weights_cfg is not None:
        class_weight_dict = {
            i: float(class_weights_cfg.get(cls, 1.0))
            for i, cls in enumerate(selected_classes)
        }
        print(f"[train] Class weights: {class_weight_dict}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode='max',
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=rlr_factor,
            patience=rlr_patience,
            verbose=1,
            mode='max',
            min_lr=min_lr,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=es_patience,
            verbose=1,
            mode='max',
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            append=bool(args.resume),
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        inp_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=2,
        validation_data=(inp_val, Y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    # ── Save training curves ──────────────────────────────────────────────────
    show_history(history, save_dir=log_dir)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    score = model.evaluate(inp_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"\n[train] Test loss={score[0]:.4f}  Test accuracy={score[1]:.4f}")

    with open(os.path.join(res_dir, 'test_score.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'accuracy'])
        writer.writerow([score[0], score[1]])

    # ── Per-SNR evaluation ────────────────────────────────────────────────────
    print("\n[train] Running per-SNR evaluation ...")
    acc         = {}
    acc_mod_snr = np.zeros((len(mods), len(snrs)))
    test_SNRs   = [lbl[x][1] for x in test_idx]

    # Overall confusion matrix
    Y_hat_all = model.predict(inp_test, batch_size=batch_size)
    confnorm_all, _, _ = calculate_confusion_matrix(Y_test, Y_hat_all, mods)
    plot_confusion_matrix(
        confnorm_all, labels=mods,
        title=f'Confusion Matrix — {exp_name} (all SNRs)',
        save_filename=os.path.join(fig_dir, 'confusion_all_snrs.png'),
    )

    # Per-class accuracy
    per_class_acc = np.diag(confnorm_all)
    print("\n[train] Per-class test accuracy:")
    for cls, acc_val in zip(mods, per_class_acc):
        print(f"  {cls:8s}: {acc_val*100:.1f}%")

    with open(os.path.join(res_dir, 'per_class_acc.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'accuracy'])
        for cls, acc_val in zip(mods, per_class_acc):
            writer.writerow([cls, f'{acc_val:.4f}'])

    # Per-SNR
    per_snr_csv = open(os.path.join(res_dir, 'acc_per_snr.csv'), 'w', newline='')
    snr_writer  = csv.writer(per_snr_csv)
    snr_writer.writerow(['snr', 'accuracy'])

    for i, snr in enumerate(snrs):
        mask      = np.array(test_SNRs) == snr
        X_snr     = [br[mask] for br in inp_test]
        Y_snr     = Y_test[mask]
        Y_hat_snr = model.predict(X_snr, batch_size=batch_size)
        confnorm_i, cor, ncor = calculate_confusion_matrix(Y_snr, Y_hat_snr, mods)
        acc[snr]  = cor / (cor + ncor)
        snr_writer.writerow([snr, acc[snr]])
        acc_mod_snr[:, i] = np.round(
            np.diag(confnorm_i) / np.maximum(np.sum(confnorm_i, axis=1), 1e-9), 4)
        plot_confusion_matrix(
            confnorm_i, labels=mods,
            title=f'Confusion (SNR={snr} dB, acc={100*acc[snr]:.1f}%)',
            save_filename=os.path.join(fig_dir, f'confusion_snr{snr:+03d}.png'),
        )
    per_snr_csv.close()

    # ── Save result pickles ───────────────────────────────────────────────────
    with open(os.path.join(res_dir, 'acc.dat'), 'wb') as f:
        pickle.dump(acc, f)
    with open(os.path.join(res_dir, 'acc_for_mod.dat'), 'wb') as f:
        pickle.dump(acc_mod_snr, f)

    # ── Accuracy plots ────────────────────────────────────────────────────────
    plot_acc_vs_snr(
        acc,
        title=f'Overall Accuracy vs SNR — {exp_name}',
        save_filename=os.path.join(fig_dir, 'acc_vs_snr.png'),
    )
    plot_acc_per_class_vs_snr(acc_mod_snr, mods, snrs, save_dir=fig_dir)

    print(f"\n[train] Done. All outputs saved under {out_cfg['base_dir']}/")
    print(f"[train] Peak overall accuracy: "
          f"{max(acc.values()):.4f} at SNR={max(acc, key=acc.get)} dB")
    print(f"\n[cost] Cost matrix used: alpha={alpha}  "
          f"pairs={[list(p) for p in high_cost_pairs]}")


if __name__ == '__main__':
    main()
