"""
train_kfold.py — 5-Fold Cross-Validation for MCLDNN and MCLDNN-Attention
=========================================================================
Usage:
    python src/train_kfold.py --config configs/exp_5class_baseline_kfold.yaml
    python src/train_kfold.py --config configs/exp_5class_attention_kfold.yaml

    # Override dataset path (Kaggle):
    python src/train_kfold.py --config configs/exp_5class_baseline_kfold.yaml \\
                              --datasetpath /kaggle/input/.../RML2016.10a_5class.pkl

Splitting strategy (block-stratified, zero data leakage):
──────────────────────────────────────────────────────────
  For each (modulation, SNR) block of 1000 samples (seeded permutation):
    ├── 200 samples → final held-out test set  (never touched during training)
    └── 800 samples → divided into 5 equal folds of 160 each

  Fold k (k = 0..4):
    ├── val   = fold k                         (160 × n_blocks)
    └── train = folds 0..4 excluding fold k    (640 × n_blocks)

  Test evaluation: performed EXACTLY ONCE after all 5 folds complete,
  using the single best-fold model (selected by peak val_accuracy).

Output structure:
──────────────────
  experiments/<name>/kfold/
    fold_0/
      best_model.weights.h5   — best weights from this fold's training
      history.csv             — per-epoch train/val loss and accuracy
      val_confusion.png       — confusion matrix on fold's validation set
      metrics.json            — {val_acc, val_loss, best_epoch, n_epochs}
    fold_1/ ... fold_4/       — same structure
    test/
      confusion_all_snrs.png
      acc_vs_snr.png
      acc_per_class_vs_snr.png
      test_score.csv          — overall accuracy + loss
      per_class_acc.csv
      acc_per_snr.csv
    summary.json              — mean±std val_acc, best_fold, all fold metrics
"""

import os
import sys
import csv
import json
import pickle
import argparse
import numpy as np

# ── Repo-root path fix ────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ['KERAS_BACKEND'] = 'tensorflow'

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='5-Fold Cross-Validation for MCLDNN / MCLDNN-Attention')
    p.add_argument('--config', required=True,
                   help='Path to YAML experiment config file')
    p.add_argument('--datasetpath', default=None,
                   help='Override dataset path from config (Kaggle use)')
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Block-stratified k-fold splitter
# ─────────────────────────────────────────────────────────────────────────────

def make_kfold_splits(n_blocks: int,
                      block_size: int = 1000,
                      n_folds: int = 5,
                      test_per_block: int = 200,
                      seed: int = 2016):
    """
    Build block-stratified train / val / test index lists.

    Within each (mod, SNR) block of `block_size` samples a seeded permutation
    is drawn.  The first `test_per_block` positions become the test set; the
    remaining (block_size - test_per_block) are divided evenly into `n_folds`
    validation folds.

    Parameters
    ----------
    n_blocks       : int  total number of (mod, SNR) blocks
    block_size     : int  samples per block (default 1000 for RML2016.10a)
    n_folds        : int  number of CV folds (default 5)
    test_per_block : int  samples reserved as final test per block (default 200)
    seed           : int  RNG seed for reproducibility

    Returns
    -------
    test_idx  : list[int]         indices of the held-out test set
    fold_idx  : list[list[int]]   fold_idx[k] = indices of samples in fold k
    """
    trainval_per_block = block_size - test_per_block          # 800
    assert trainval_per_block % n_folds == 0, (
        f"trainval_per_block={trainval_per_block} must be divisible by n_folds={n_folds}"
    )
    fold_size = trainval_per_block // n_folds                  # 160

    rng      = np.random.RandomState(seed)
    test_idx = []
    fold_idx = [[] for _ in range(n_folds)]

    for b in range(n_blocks):
        base          = b * block_size
        perm          = rng.permutation(block_size)
        block_indices = base + perm

        test_idx.extend(block_indices[:test_per_block].tolist())

        trainval = block_indices[test_per_block:]              # 800 indices
        for k in range(n_folds):
            fold_idx[k].extend(
                trainval[k * fold_size: (k + 1) * fold_size].tolist()
            )

    return test_idx, fold_idx


# ─────────────────────────────────────────────────────────────────────────────
# Input preparation (mirrors train.py)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inputs(X: np.ndarray):
    """
    Return the three-input list expected by MCLDNN / MCLDNN-Attention.

    Shape: [IQ_4D (N,2,128,1), I_1D (N,128,1), Q_1D (N,128,1)]
    """
    return [
        np.expand_dims(X, axis=3),            # (N, 2, 128, 1) — IQ 2D
        np.expand_dims(X[:, 0, :], axis=2),   # (N, 128, 1)    — I channel
        np.expand_dims(X[:, 1, :], axis=2),   # (N, 128, 1)    — Q channel
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model builder (fresh model every fold)
# ─────────────────────────────────────────────────────────────────────────────

def build_fresh_model(model_type: str,
                      n_classes: int,
                      dropout_rate: float,
                      l2_dense: float,
                      l2_lstm: float,
                      initial_lr: float,
                      keras):
    """
    Build and compile a fresh (randomly initialised) model.

    Parameters
    ----------
    model_type   : 'mcldnn', 'lstm64', 'mcldnn_attention', or 'mcldnn_diffattention'
    n_classes    : number of output classes
    dropout_rate : dropout probability
    l2_dense     : L2 weight on Dense layers (MCLDNN only)
    l2_lstm      : L2 weight on LSTM layers  (MCLDNN only)
    initial_lr   : Adam learning rate
    keras        : the keras module (imported after seed is set)

    Returns
    -------
    keras.Model  compiled training model
    """
    if model_type == 'mcldnn_attention':
        from src.models.mcldnn_attention import build_mcldnn_attention
        model = build_mcldnn_attention(
            classes=n_classes,
            dropout_rate=dropout_rate,
            learning_rate=initial_lr,
        )
    elif model_type == 'mcldnn_diffattention':
        from src.models.mcldnn_diffattention import build_mcldnn_diffattention
        model = build_mcldnn_diffattention(
            classes=n_classes,
            dropout_rate=dropout_rate,
            learning_rate=initial_lr,
        )
    elif model_type == 'lstm64':
        from src.models.mcldnn_lstm64 import MCLDNN_LSTM64
        model = MCLDNN_LSTM64(
            classes=n_classes,
            dropout_rate=dropout_rate,
            l2_dense=l2_dense,
            l2_lstm=l2_lstm,
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(
                learning_rate=initial_lr, clipnorm=1.0),
            metrics=['accuracy'],
        )
    else:  # 'mcldnn' (baseline)
        from src.models.mcldnn import MCLDNN
        model = MCLDNN(
            classes=n_classes,
            dropout_rate=dropout_rate,
            l2_dense=l2_dense,
            l2_lstm=l2_lstm,
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(
                learning_rate=initial_lr, clipnorm=1.0),
            metrics=['accuracy'],
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg  = _load_config(args.config)
    seed = cfg['experiment'].get('seed', 2016)

    # ── Seed everything (before any TF/Keras import) ──────────────────────────
    from src.utils.seed import set_all_seeds
    set_all_seeds(seed)

    # ── TF / Keras (after seed) ───────────────────────────────────────────────
    import keras
    from src.dataset import normalize_samples
    from src.utils.mltools import (
        show_history, calculate_confusion_matrix,
        plot_confusion_matrix, plot_acc_vs_snr,
        plot_acc_per_class_vs_snr,
    )

    # ── Read config ───────────────────────────────────────────────────────────
    exp_name         = cfg['experiment']['name']
    model_type       = cfg['experiment'].get('model', 'mcldnn')
    selected_classes = cfg['dataset']['classes']
    data_path        = args.datasetpath or cfg['dataset']['path']

    tr_cfg       = cfg['training']
    nb_epoch     = tr_cfg['epochs']
    batch_size   = tr_cfg['batch_size']
    es_patience  = tr_cfg['early_stopping_patience']
    rlr_patience = tr_cfg['reduce_lr_patience']
    rlr_factor   = tr_cfg['reduce_lr_factor']
    min_lr       = tr_cfg['min_lr']
    dropout_rate = tr_cfg.get('dropout_rate', 0.5)
    l2_dense     = tr_cfg.get('l2_dense', 1e-3)
    l2_lstm      = tr_cfg.get('l2_lstm',  1e-4)
    initial_lr   = tr_cfg.get('initial_lr', 1e-3)

    class_weights_cfg = tr_cfg.get('class_weights', None)

    kf_cfg   = cfg.get('kfold', {})
    n_folds  = kf_cfg.get('n_folds', 5)
    kf_seed  = kf_cfg.get('seed', seed)

    out_cfg  = cfg['output']
    base_dir = out_cfg['base_dir']
    kfold_dir = os.path.join(base_dir, 'kfold')
    n_classes = len(selected_classes)

    os.makedirs(kfold_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  K-Fold CV : {exp_name}")
    print(f"  Model     : {model_type}")
    print(f"  Folds     : {n_folds}")
    print(f"  Dropout   : {dropout_rate}  L2 Dense: {l2_dense}  L2 LSTM: {l2_lstm}")
    print(f"  LR        : {initial_lr}    ES patience: {es_patience}")
    print(f"  Classes   : {selected_classes}")
    print(f"  Output    : {kfold_dir}")
    print(f"{'='*60}\n")

    # ── Load raw pickle (do NOT use load_data — we manage splits ourselves) ───
    print("[kfold] Loading raw dataset ...")
    Xd = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')

    all_snrs = sorted(set(k[1] for k in Xd.keys()))
    mods     = selected_classes          # ordered — determines one-hot index
    snrs     = all_snrs

    # Build flat X array and label list (identical order to dataset.py)
    X_raw, lbl = [], []
    for mod in mods:
        for snr in snrs:
            block = Xd[(mod, snr)]       # (1000, 2, 128)
            X_raw.append(block)
            for _ in range(block.shape[0]):
                lbl.append((mod, snr))
    X_raw = np.vstack(X_raw)             # (N, 2, 128)

    n_blocks   = len(mods) * len(snrs)
    n_examples = X_raw.shape[0]
    print(f"[kfold] {len(mods)} classes × {len(snrs)} SNRs × 1000 = {n_examples} samples")

    # ── Build block-stratified splits ─────────────────────────────────────────
    test_idx, fold_idx = make_kfold_splits(
        n_blocks=n_blocks,
        block_size=1000,
        n_folds=n_folds,
        test_per_block=200,
        seed=kf_seed,
    )

    # Sanity checks
    all_trainval = [i for f in fold_idx for i in f]
    assert len(set(test_idx) & set(all_trainval)) == 0, \
        "Data leakage: test and trainval sets overlap!"
    assert len(test_idx) + len(all_trainval) == n_examples, \
        "Split sizes don't sum to total samples!"
    print(f"[kfold] Test set   : {len(test_idx):,} samples ({len(test_idx)/n_examples*100:.0f}%)")
    print(f"[kfold] Train+Val  : {len(all_trainval):,} samples per fold rotation")
    print(f"[kfold] Per fold   : ~{len(fold_idx[0]):,} val  /  ~{len(all_trainval)-len(fold_idx[0]):,} train\n")

    # Normalise and prepare test inputs once (test set is FIXED across all folds)
    X_test_raw = normalize_samples(X_raw[test_idx])
    Y_test     = _to_onehot([mods.index(lbl[i][0]) for i in test_idx], n_classes)
    inp_test   = prepare_inputs(X_test_raw)
    test_SNRs  = [lbl[i][1] for i in test_idx]

    # Build class_weight_dict if needed
    class_weight_dict = None
    if class_weights_cfg is not None:
        class_weight_dict = {
            i: float(class_weights_cfg.get(cls, 1.0))
            for i, cls in enumerate(mods)
        }
        print(f"[kfold] Class weights: {class_weight_dict}")

    # ── 5-Fold training loop ──────────────────────────────────────────────────
    fold_results = []   # list of dicts with val_acc, val_loss, best_epoch

    for fold in range(n_folds):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold + 1}/{n_folds}")
        print(f"{'─'*60}")

        fold_dir = os.path.join(kfold_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        ckpt_path = os.path.join(fold_dir, 'best_model.weights.h5')

        # Build train / val index lists for this fold
        val_idx   = fold_idx[fold]
        train_idx = [i for k, f in enumerate(fold_idx) if k != fold for i in f]

        # Normalise inputs
        X_train_raw = normalize_samples(X_raw[train_idx])
        X_val_raw   = normalize_samples(X_raw[val_idx])
        Y_train     = _to_onehot([mods.index(lbl[i][0]) for i in train_idx], n_classes)
        Y_val       = _to_onehot([mods.index(lbl[i][0]) for i in val_idx],   n_classes)

        inp_train = prepare_inputs(X_train_raw)
        inp_val   = prepare_inputs(X_val_raw)

        print(f"[fold {fold}] train={len(train_idx):,}  val={len(val_idx):,}")

        # Re-seed before each fold so weight initialisation is deterministic
        # but differs per fold (fold index shifts the seed)
        set_all_seeds(seed + fold)

        # Build FRESH model from scratch
        model = build_fresh_model(
            model_type=model_type,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            l2_dense=l2_dense,
            l2_lstm=l2_lstm,
            initial_lr=initial_lr,
            keras=keras,
        )

        # Callbacks
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
                os.path.join(fold_dir, 'history.csv'),
            ),
        ]

        # Train
        history = model.fit(
            inp_train, Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=2,
            validation_data=(inp_val, Y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
        )

        # Collect best-epoch metrics
        best_epoch   = int(np.argmax(history.history['val_accuracy']))
        best_val_acc = float(np.max(history.history['val_accuracy']))
        best_val_loss= float(history.history['val_loss'][best_epoch])
        n_epochs_run = len(history.history['val_accuracy'])

        print(f"[fold {fold}] Best epoch={best_epoch+1}  "
              f"val_acc={best_val_acc:.4f}  val_loss={best_val_loss:.4f}")

        # Confusion matrix on validation set (using best restored weights)
        Y_hat_val = model.predict(inp_val, batch_size=batch_size)
        confnorm_val, _, _ = calculate_confusion_matrix(Y_val, Y_hat_val, mods)
        plot_confusion_matrix(
            confnorm_val, labels=mods,
            title=f'Fold {fold} Validation Confusion (val_acc={best_val_acc:.3f})',
            save_filename=os.path.join(fold_dir, 'val_confusion.png'),
        )

        # Save history plots
        show_history(history, save_dir=fold_dir)

        # Save metrics JSON
        metrics = {
            'fold':        fold,
            'val_acc':     best_val_acc,
            'val_loss':    best_val_loss,
            'best_epoch':  best_epoch + 1,
            'n_epochs':    n_epochs_run,
        }
        with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        fold_results.append(metrics)

        # Free GPU memory before next fold
        del model
        keras.backend.clear_session()

    # ── Summary across folds ──────────────────────────────────────────────────
    val_accs  = [r['val_acc']  for r in fold_results]
    val_losses= [r['val_loss'] for r in fold_results]
    mean_acc  = float(np.mean(val_accs))
    std_acc   = float(np.std(val_accs))
    best_fold = int(np.argmax(val_accs))

    print(f"\n{'='*60}")
    print(f"  K-FOLD SUMMARY  ({exp_name})")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']} | val_acc={r['val_acc']:.4f} "
              f"| val_loss={r['val_loss']:.4f} | best_epoch={r['best_epoch']}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Mean val_acc : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best fold    : fold_{best_fold} "
          f"(val_acc={fold_results[best_fold]['val_acc']:.4f})")
    print(f"{'='*60}\n")

    summary = {
        'exp_name':   exp_name,
        'n_folds':    n_folds,
        'fold_results': fold_results,
        'mean_val_acc': mean_acc,
        'std_val_acc':  std_acc,
        'best_fold':    best_fold,
    }
    with open(os.path.join(kfold_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[kfold] Summary saved → {os.path.join(kfold_dir, 'summary.json')}")

    # ── Final test evaluation (ONCE, using best-fold model) ───────────────────
    print(f"\n[kfold] Loading best fold model (fold_{best_fold}) for test evaluation ...")
    best_ckpt = os.path.join(kfold_dir, f'fold_{best_fold}', 'best_model.weights.h5')

    set_all_seeds(seed + best_fold)
    best_model = build_fresh_model(
        model_type=model_type,
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        l2_dense=l2_dense,
        l2_lstm=l2_lstm,
        initial_lr=initial_lr,
        keras=keras,
    )
    best_model.load_weights(best_ckpt)
    print(f"[kfold] Loaded weights from {best_ckpt}")

    test_dir = os.path.join(kfold_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # Overall test score
    score = best_model.evaluate(inp_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"\n[kfold] TEST  loss={score[0]:.4f}  accuracy={score[1]:.4f}")

    with open(os.path.join(test_dir, 'test_score.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'accuracy'])
        writer.writerow([score[0], score[1]])

    # Overall confusion matrix
    Y_hat_test = best_model.predict(inp_test, batch_size=batch_size)
    confnorm_all, _, _ = calculate_confusion_matrix(Y_test, Y_hat_test, mods)
    plot_confusion_matrix(
        confnorm_all, labels=mods,
        title=f'Test Confusion Matrix — {exp_name} (acc={score[1]:.3f})',
        save_filename=os.path.join(test_dir, 'confusion_all_snrs.png'),
    )

    # Per-class accuracy CSV
    per_class = np.diag(confnorm_all)
    with open(os.path.join(test_dir, 'per_class_acc.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'accuracy'])
        for cls, acc_val in zip(mods, per_class):
            writer.writerow([cls, f'{acc_val:.4f}'])
            print(f"  {cls:8s}: {acc_val*100:.1f}%")

    # Per-SNR evaluation
    acc_snr      = {}
    acc_mod_snr  = np.zeros((len(mods), len(snrs)))
    per_snr_rows = []

    for i, snr in enumerate(snrs):
        mask      = np.array(test_SNRs) == snr
        X_snr     = [br[mask] for br in inp_test]
        Y_snr     = Y_test[mask]
        Y_hat_snr = best_model.predict(X_snr, batch_size=batch_size)
        confnorm_i, cor, ncor = calculate_confusion_matrix(Y_snr, Y_hat_snr, mods)
        acc_snr[snr]    = cor / (cor + ncor)
        per_snr_rows.append([snr, acc_snr[snr]])
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) /
                                      np.maximum(np.sum(confnorm_i, axis=1), 1e-9), 4)

    with open(os.path.join(test_dir, 'acc_per_snr.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr', 'accuracy'])
        writer.writerows(per_snr_rows)

    plot_acc_vs_snr(
        acc_snr,
        title=f'Test Accuracy vs SNR — {exp_name}',
        save_filename=os.path.join(test_dir, 'acc_vs_snr.png'),
    )
    plot_acc_per_class_vs_snr(acc_mod_snr, mods, snrs, save_dir=test_dir)

    print(f"\n[kfold] Peak SNR accuracy: "
          f"{max(acc_snr.values()):.4f} at SNR={max(acc_snr, key=acc_snr.get)} dB")
    print(f"[kfold] All outputs saved under: {kfold_dir}/")
    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"  Val  (5-fold): {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Test (best model fold_{best_fold}): {score[1]:.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_onehot(class_indices: list, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(class_indices), n_classes), dtype=np.float32)
    oh[np.arange(len(class_indices)), class_indices] = 1.0
    return oh


if __name__ == '__main__':
    main()
