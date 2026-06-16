"""
train.py — Config-driven MCLDNN training entry point
=====================================================
Usage
-----
    python src/train.py --config configs/exp_5class_baseline.yaml
    python src/train.py --config configs/exp_4class_ablation.yaml

    # Override dataset path (e.g. on Kaggle where the full .dat file is the input)
    python src/train.py --config configs/exp_5class_baseline.yaml \
                        --datasetpath /kaggle/input/rml201610a-dict/RML2016.10a_dict.dat

Resume from checkpoint
----------------------
    python src/train.py --config configs/exp_5class_baseline.yaml \
                        --resume experiments/5class_baseline/checkpoints/best_model.weights.h5

Branch ablation
---------------
    python src/train.py --config configs/branch_ablation/5class_I.yaml
    python src/train.py --config configs/branch_ablation/5class_Q.yaml
    python src/train.py --config configs/branch_ablation/5class_IQ.yaml

    The YAML key experiment.branch selects the model variant:
      'full' (default) — original 3-input MCLDNN
      'I'              — I-channel Conv1D branch only
      'Q'              — Q-channel Conv1D branch only
      'IQ'             — combined 2D-CNN branch only
"""

import os
import sys

# ── Repo-root path fix ───────────────────────────────────────────────────────
# When this script is invoked as `python src/train.py`, Python puts src/ on
# sys.path, making `from src.utils...` fail. Insert the repo root explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import csv
import pickle
import numpy as np
import yaml

# ── Seed must be set before any TF/Keras import ──────────────────────────────
# We defer the actual seed call until after we read the config


def _parse_args():
    p = argparse.ArgumentParser(description='Train MCLDNN on RML2016.10a')
    p.add_argument('--config', required=True,
                   help='Path to YAML experiment config file')
    p.add_argument('--datasetpath', default=None,
                   help='Override the dataset path in the YAML config. '
                        'Use this on Kaggle to point to the .dat input file. '
                        'Example: /kaggle/input/rml201610a-dict/RML2016.10a_dict.dat')
    p.add_argument('--resume', default=None,
                   help='Path to .weights.h5 checkpoint to resume from')
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args   = _parse_args()
    cfg    = _load_config(args.config)
    seed   = cfg['experiment'].get('seed', 2016)

    # ── Seed everything ───────────────────────────────────────────────────────
    from src.utils.seed import set_all_seeds
    set_all_seeds(seed)

    # ── TF / Keras imports (after seed) ───────────────────────────────────────
    import keras
    from src.dataset       import load_data
    from src.utils.mltools import (show_history, calculate_confusion_matrix,
                                   plot_confusion_matrix, plot_acc_vs_snr,
                                   plot_acc_per_class_vs_snr)

    # ── Read config values ────────────────────────────────────────────────────
    exp_name         = cfg['experiment']['name']
    branch           = cfg['experiment'].get('branch', 'full')
    selected_classes = cfg['dataset']['classes']
    data_path        = args.datasetpath if args.datasetpath else cfg['dataset']['path']

    tr_cfg           = cfg['training']
    nb_epoch         = tr_cfg['epochs']
    batch_size       = tr_cfg['batch_size']
    es_patience      = tr_cfg['early_stopping_patience']
    rlr_patience     = tr_cfg['reduce_lr_patience']
    rlr_factor       = tr_cfg['reduce_lr_factor']
    min_lr           = tr_cfg['min_lr']
    dropout_rate     = tr_cfg.get('dropout_rate', 0.5)   # default 0.5 matches original paper
    l2_dense         = tr_cfg.get('l2_dense', 1e-3)      # default 1e-3 matches original paper
    l2_lstm          = tr_cfg.get('l2_lstm',  1e-4)      # default 1e-4 matches original paper
    snr_weighted     = tr_cfg.get('snr_weighted', False)  # SNR-proportional sample weights
    initial_lr       = tr_cfg.get('initial_lr', 1e-3)    # default 1e-3 (Adam, original paper)
    class_weights_cfg= tr_cfg.get('class_weights', None) # class-specific penalty weights

    # Transfer-learning config (4-class only)
    transfer_from           = cfg.get('transfer_from', None)
    transfer_source_classes = cfg.get('transfer_source_classes', 5)

    out_cfg          = cfg['output']
    ckpt_path        = out_cfg['checkpoint_path']
    log_dir          = out_cfg['log_dir']
    fig_dir          = out_cfg['figure_dir']
    res_dir          = out_cfg['result_dir']

    n_classes        = len(selected_classes)

    # Optional fixed-split file (decouples seed from test-set definition)
    split_file    = cfg.get('split_file', None)

    # SNR filtering + global shuffle split (attention experiment)
    snr_range     = cfg.get('snr_range', None)
    if snr_range is not None:
        snr_range = tuple(snr_range)   # YAML loads lists; convert to tuple
    shuffle_split = cfg.get('shuffle_split', False)

    # Model type selection
    model_type    = cfg['experiment'].get('model', 'mcldnn')

    # ── Create output directories ─────────────────────────────────────────────
    for d in [os.path.dirname(ckpt_path), log_dir, fig_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Model      : {model_type}")
    print(f"  Branch     : {branch}")
    print(f"  Dropout    : {dropout_rate}")
    print(f"  L2 Dense   : {l2_dense}   L2 LSTM : {l2_lstm}")
    print(f"  SNR Weighted: {snr_weighted}")
    print(f"  Initial LR : {initial_lr}")
    print(f"  SNR Range  : {snr_range}")
    print(f"  Shuffle Split: {shuffle_split}")
    if transfer_from:
        print(f"  Transfer   : {transfer_from}")
    print(f"  Classes    : {selected_classes}")
    print(f"  n_classes  : {n_classes}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    (mods, snrs, lbl), \
    (X_train, Y_train), \
    (X_val,   Y_val),   \
    (X_test,  Y_test),  \
    (train_idx, val_idx, test_idx) = load_data(
        data_path, selected_classes, seed,
        split_file=split_file,
        snr_range=snr_range,
        shuffle_split=shuffle_split)

    # ── Prepare inputs (branch-aware) ─────────────────────────────────────────
    def prepare_inputs(X):
        """
        Return the model input list/array for the selected branch.

        Branch inputs
        -------------
        'full': [IQ_4D (N,2,128,1), I_1D (N,128,1), Q_1D (N,128,1)]
        'I'   : [I_1D  (N,128,1)]
        'Q'   : [Q_1D  (N,128,1)]
        'IQ'  : [IQ_4D (N,2,128,1)]
        """
        if branch == 'full':
            return [
                np.expand_dims(X, axis=3),           # (N, 2, 128, 1) — IQ 2D
                np.expand_dims(X[:, 0, :], axis=2),  # (N, 128, 1)    — I channel
                np.expand_dims(X[:, 1, :], axis=2),  # (N, 128, 1)    — Q channel
            ]
        elif branch == 'I':
            return [np.expand_dims(X[:, 0, :], axis=2)]   # (N, 128, 1)
        elif branch == 'Q':
            return [np.expand_dims(X[:, 1, :], axis=2)]   # (N, 128, 1)
        else:  # 'IQ'
            return [np.expand_dims(X, axis=3)]             # (N, 2, 128, 1)

    inp_train = prepare_inputs(X_train)
    inp_val   = prepare_inputs(X_val)
    inp_test  = prepare_inputs(X_test)

    # For single-input branch variants Keras 3 expects a plain array, not a
    # 1-element list.  Unwrap here so model.fit / evaluate / predict don't emit:
    #   "The structure of `inputs` doesn't match the expected structure."
    if branch != 'full':
        inp_train = inp_train[0]   # (N, 128, 1)  or  (N, 2, 128, 1)
        inp_val   = inp_val[0]
        inp_test  = inp_test[0]

    # Print input shapes (handle single-input variants gracefully)
    if branch == 'full':
        print(f"[train] Input shapes: "
              f"IQ={inp_train[0].shape}  I={inp_train[1].shape}  Q={inp_train[2].shape}")
    else:
        print(f"[train] Input shapes ({branch}): {inp_train.shape}")

    # ── Build model ────────────────────────────────────────────────────────────
    resume_weights = args.resume
    is_attention   = (model_type == 'mcldnn_attention')

    if is_attention:
        from src.models.mcldnn_attention import build_mcldnn_attention
        model = build_mcldnn_attention(classes=n_classes,
                                       dropout_rate=dropout_rate)
        if resume_weights:
            model.load_weights(resume_weights)
    elif branch == 'full':
        from src.models.mcldnn import MCLDNN
        model = MCLDNN(weights=resume_weights, classes=n_classes,
                       dropout_rate=dropout_rate, l2_dense=l2_dense,
                       l2_lstm=l2_lstm)
    else:
        from src.models.mcldnn_ablation import build_mcldnn_branch
        model = build_mcldnn_branch(classes=n_classes, branch=branch,
                                    dropout_rate=dropout_rate,
                                    l2_dense=l2_dense,
                                    l2_lstm=l2_lstm,
                                    weights=resume_weights)

    # ── Transfer learning (4-class: copy 5-class CNN+LSTM, skip output layer) ───────
    # The 4-class problem suffers from gradient cancellation at initialization:
    # balanced classes + random CNN features → batch gradients cancel over epoch
    # → model never escapes the uniform-output attractor.
    # The 5-class model already solved this via BPSK's distinctive signal structure.
    # Transferring its CNN+LSTM weights gives the 4-class model meaningful initial
    # feature representations, bypassing the gradient cancellation problem entirely.
    if transfer_from:
        if not os.path.exists(transfer_from):
            print(f"[train] WARNING: transfer_from path not found: {transfer_from}")
            print("[train] Falling back to random initialisation.")
        else:
            print(f"[train] Loading source model ({transfer_source_classes}-class) "
                  f"for weight transfer...")
            if branch == 'full':
                from src.models.mcldnn import MCLDNN as _SrcModel
                src = _SrcModel(classes=transfer_source_classes,
                                l2_dense=l2_dense, l2_lstm=l2_lstm)
            else:
                from src.models.mcldnn_ablation import build_mcldnn_branch as _SrcBuild
                src = _SrcBuild(classes=transfer_source_classes, branch=branch,
                                l2_dense=l2_dense, l2_lstm=l2_lstm)
            src.load_weights(transfer_from)

            n_ok, n_skip = 0, 0
            for src_layer in src.layers:
                try:
                    dst_layer = model.get_layer(src_layer.name)
                    src_w = src_layer.get_weights()
                    dst_w = dst_layer.get_weights()
                    if src_w and all(s.shape == d.shape
                                     for s, d in zip(src_w, dst_w)):
                        dst_layer.set_weights(src_w)
                        n_ok += 1
                        print(f"  \u2713 {src_layer.name}")
                    elif src_w:
                        n_skip += 1
                        print(f"  \u2717 {src_layer.name}  "
                              f"(shape mismatch — output layer, random init)")
                except ValueError:
                    pass
            print(f"[train] Transfer complete: {n_ok} layers copied, "
                  f"{n_skip} skipped (output layer).")
            del src

    # Attention model is already compiled inside build_mcldnn_attention().
    # For MCLDNN/ablation, compile with standard single-output settings.
    if not is_attention:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),
            metrics=['accuracy']
        )

    # ── SNR-weighted sample weighting ─────────────────────────────────────────
    # Root-cause fix for 4-class training instability:
    # 55% of training samples (SNR ≤ 0 dB) carry near-zero discriminative gradient
    # for similar modulations (QPSK/8PSK/QAM16/QAM64).  Weighting by SNR ensures
    # the effective loss gradient is dominated by high-SNR samples that produce
    # meaningful class separation, providing enough signal to survive 124-step BPTT
    # through the LSTM and reach the CNN feature extractor.
    #
    # Weight formula: w(SNR) = 0.1 + 0.9 × (SNR − SNR_min) / (SNR_max − SNR_min)
    #   → SNR = -20 dB → w = 0.10  (low-SNR still contributes, not discarded)
    #   → SNR =  18 dB → w = 1.00  (high-SNR gets full weight)
    # Weights are mean-normalised so overall loss magnitude is preserved.
    if snr_weighted:
        _train_snrs    = np.array([lbl[i][1] for i in train_idx], dtype=np.float32)
        _snr_min, _snr_max = _train_snrs.min(), _train_snrs.max()
        sample_weights = 0.1 + 0.9 * (_train_snrs - _snr_min) / (_snr_max - _snr_min)
        sample_weights = sample_weights / sample_weights.mean()  # mean-normalise
        print(f"[train] SNR-weighted: w_min={sample_weights.min():.3f}  "
              f"w_max={sample_weights.max():.3f}  w_mean={sample_weights.mean():.3f}")
    else:
        sample_weights = None
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=rlr_factor,
            patience=rlr_patience,
            verbose=1,
            mode='max',
            min_lr=min_lr
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=es_patience,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            append=bool(resume_weights)
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    # The attention training model has a SINGLE softmax output (just like the
    # regular MCLDNN), so Y_train / Y_val / Y_test are passed directly.
    # Attention weight extraction happens only in evaluate_attention.py via
    # the separate build_mcldnn_attention_extractor() model.
    
    class_weight_dict = None
    if class_weights_cfg is not None:
        class_weight_dict = {}
        for i, cls in enumerate(selected_classes):
            class_weight_dict[i] = float(class_weights_cfg.get(cls, 1.0))
        print(f"[train] Using class weights: {class_weight_dict}")

    history = model.fit(
        inp_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=2,
        validation_data=(inp_val, Y_val),
        callbacks=callbacks,
        sample_weight=sample_weights,
        class_weight=class_weight_dict
    )

    # ── Save training curves ──────────────────────────────────────────────────
    show_history(history, save_dir=log_dir)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    score = model.evaluate(inp_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"\n[train] Test loss={score[0]:.4f}  Test accuracy={score[1]:.4f}")

    # ── Save test score ───────────────────────────────────────────────────────
    with open(os.path.join(res_dir, 'test_score.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'accuracy'])
        writer.writerow([score[0], score[1]])

    # ── Per-SNR evaluation ────────────────────────────────────────────────────
    print("\n[train] Running per-SNR evaluation ...")
    acc            = {}
    acc_mod_snr    = np.zeros((len(mods), len(snrs)))
    test_SNRs      = [lbl[x][1] for x in test_idx]

    # Overall confusion matrix
    # Both training model variants return a plain ndarray (single output)
    Y_hat_all = model.predict(inp_test, batch_size=batch_size)
    confnorm_all, _, _ = calculate_confusion_matrix(Y_test, Y_hat_all, mods)
    plot_confusion_matrix(
        confnorm_all, labels=mods,
        title=f'Confusion Matrix — {exp_name} (all SNRs)',
        save_filename=os.path.join(fig_dir, 'confusion_all_snrs.png')
    )

    # Per-SNR
    per_snr_csv = open(os.path.join(res_dir, 'acc_per_snr.csv'), 'w', newline='')
    snr_writer  = csv.writer(per_snr_csv)
    snr_writer.writerow(['snr', 'accuracy'])

    for i, snr in enumerate(snrs):
        mask  = np.array(test_SNRs) == snr
        # inp_test is a list of arrays for 'full'/'attention', array for single-branch
        if branch == 'full' or is_attention:
            X_snr = [br[mask] for br in inp_test]
        else:
            X_snr = inp_test[mask]
        Y_snr      = Y_test[mask]
        Y_hat_snr  = model.predict(X_snr, batch_size=batch_size)
        confnorm_i, cor, ncor = calculate_confusion_matrix(Y_snr, Y_hat_snr, mods)
        acc[snr]   = cor / (cor + ncor)
        snr_writer.writerow([snr, acc[snr]])
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) /
                                      np.sum(confnorm_i, axis=1), 4)
        plot_confusion_matrix(
            confnorm_i, labels=mods,
            title=f'Confusion (SNR={snr} dB, acc={100*acc[snr]:.1f}%)',
            save_filename=os.path.join(
                fig_dir, f'confusion_snr{snr:+03d}.png')
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
        save_filename=os.path.join(fig_dir, 'acc_vs_snr.png')
    )
    plot_acc_per_class_vs_snr(acc_mod_snr, mods, snrs, save_dir=fig_dir)

    print(f"\n[train] Done. All outputs saved under {out_cfg['base_dir']}/")
    print(f"[train] Peak overall accuracy: "
          f"{max(acc.values()):.4f} at SNR={max(acc, key=acc.get)} dB")


if __name__ == '__main__':
    main()
