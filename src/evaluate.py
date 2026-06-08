"""
evaluate.py — Standalone evaluation / comparison script
========================================================
Usage (single experiment)
--------------------------
    python src/evaluate.py --config configs/exp_5class_baseline.yaml \
                           --weights experiments/5class_baseline/checkpoints/best_model.weights.h5

Usage (compare two experiments — 5-class vs 4-class)
-----------------------------------------------------
    python src/evaluate.py \
        --compare \
        --configs  configs/exp_5class_baseline.yaml configs/exp_4class_ablation.yaml \
        --weights  experiments/5class_baseline/checkpoints/best_model.weights.h5 \
                   experiments/4class_ablation/checkpoints/best_model.weights.h5 \
        --labels   "5-class" "4-class ablation" \
        --save_dir experiments/comparison_5vs4
"""

import os
import sys

# ── Repo-root path fix ───────────────────────────────────────────────────────
# When this script is invoked as `python src/evaluate.py`, Python puts src/ on
# sys.path, making `from src.utils...` fail. Insert the repo root explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import pickle
import numpy as np
import yaml

os.environ['KERAS_BACKEND'] = 'tensorflow'


def _parse_args():
    p = argparse.ArgumentParser(description='Evaluate / compare MCLDNN experiments')
    p.add_argument('--compare',  action='store_true',
                   help='Compare multiple experiment configs side-by-side')

    # Single-experiment mode
    p.add_argument('--config',   default=None)
    p.add_argument('--weights',  nargs='+', default=None)

    # Multi-experiment mode
    p.add_argument('--configs',  nargs='+', default=None)
    p.add_argument('--labels',   nargs='+', default=None,
                   help='Display names for each experiment')
    p.add_argument('--save_dir', default='experiments/comparison')
    p.add_argument('--datasetpath', default=None,
                   help='Override dataset path from YAML config. '
                        'Applied to all experiments when using --compare. '
                        'Example: /kaggle/input/rml201610a-dict/RML2016.10a_dict.dat')
    return p.parse_args()


def _load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _prepare_inputs(X):
    X1 = np.expand_dims(X[:, 0, :], axis=2)
    X2 = np.expand_dims(X[:, 1, :], axis=2)
    X4 = np.expand_dims(X, axis=3)
    return [X4, X1, X2]


def evaluate_single(cfg, weights_path, datasetpath_override=None):
    """Evaluate one experiment config + weights. Returns (acc_dict, acc_mod_snr, mods, snrs)."""
    from src.dataset       import load_data
    from src.models.mcldnn import MCLDNN
    from src.utils.mltools import (calculate_confusion_matrix,
                                    plot_confusion_matrix,
                                    plot_acc_vs_snr,
                                    plot_acc_per_class_vs_snr)
    import keras, csv

    selected_classes = cfg['dataset']['classes']
    data_path        = datasetpath_override if datasetpath_override else cfg['dataset']['path']
    seed             = cfg['experiment'].get('seed', 2016)
    exp_name         = cfg['experiment']['name']
    fig_dir          = cfg['output']['figure_dir']
    res_dir          = cfg['output']['result_dir']
    batch_size       = cfg['training']['batch_size']
    n_classes        = len(selected_classes)

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    (mods, snrs, lbl), _, _, (X_test, Y_test), (_, _, test_idx) = \
        load_data(data_path, selected_classes, seed)

    inp_test  = _prepare_inputs(X_test)
    test_SNRs = [lbl[x][1] for x in test_idx]

    model = MCLDNN(weights=weights_path, classes=n_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(clipnorm=1.0),
                  metrics=['accuracy'])

    acc         = {}
    acc_mod_snr = np.zeros((len(mods), len(snrs)))

    # Full confusion matrix
    Y_hat_all = model.predict(inp_test, batch_size=batch_size)
    confnorm_all, _, _ = calculate_confusion_matrix(Y_test, Y_hat_all, mods)
    plot_confusion_matrix(
        confnorm_all, labels=mods,
        title=f'{exp_name} — All SNRs',
        save_filename=os.path.join(fig_dir, 'confusion_all_snrs.png')
    )

    with open(os.path.join(res_dir, 'acc_per_snr.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr', 'accuracy'])
        for i, snr in enumerate(snrs):
            mask      = np.array(test_SNRs) == snr
            X_snr     = [b[mask] for b in inp_test]
            Y_snr     = Y_test[mask]
            Y_hat_snr = model.predict(X_snr, batch_size=batch_size)
            cnorm, cor, ncor = calculate_confusion_matrix(Y_snr, Y_hat_snr, mods)
            acc[snr]  = cor / (cor + ncor)
            w.writerow([snr, acc[snr]])
            acc_mod_snr[:, i] = np.round(
                np.diag(cnorm) / np.sum(cnorm, axis=1), 4)
            plot_confusion_matrix(
                cnorm, labels=mods,
                title=f'{exp_name} — SNR={snr} dB  ({100*acc[snr]:.1f}%)',
                save_filename=os.path.join(
                    fig_dir, f'confusion_snr{snr:+03d}.png')
            )

    with open(os.path.join(res_dir, 'acc.dat'), 'wb') as f:
        pickle.dump(acc, f)
    with open(os.path.join(res_dir, 'acc_for_mod.dat'), 'wb') as f:
        pickle.dump(acc_mod_snr, f)

    plot_acc_vs_snr(acc,
                    title=f'Overall Accuracy vs SNR — {exp_name}',
                    save_filename=os.path.join(fig_dir, 'acc_vs_snr.png'))
    plot_acc_per_class_vs_snr(acc_mod_snr, mods, snrs, save_dir=fig_dir)

    return acc, acc_mod_snr, mods, snrs


def compare_experiments(configs, weights_list, labels, save_dir, datasetpath_override=None):
    """
    Evaluate multiple experiments and plot them together on one accuracy curve.
    Intended for 5-class vs 4-class comparison (and any future variants).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for cfg_path, wpath, label in zip(configs, weights_list, labels):
        cfg = _load_config(cfg_path)
        print(f"\n{'─'*50}\nEvaluating: {label}")
        acc, acc_mod_snr, mods, snrs = evaluate_single(cfg, wpath, datasetpath_override)
        all_results[label] = {'acc': acc, 'acc_mod_snr': acc_mod_snr,
                               'mods': mods, 'snrs': snrs}

    # ── Overall accuracy comparison plot ──────────────────────────────────────
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    for idx, (label, res) in enumerate(all_results.items()):
        snrs_sorted = sorted(res['snrs'])
        vals = [res['acc'][s] for s in snrs_sorted]
        plt.plot(snrs_sorted, vals,
                 label=label, marker=markers[idx % len(markers)],
                 linewidth=2, markersize=6)

    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Classification Accuracy', fontsize=13)
    plt.title('MCLDNN: Experiment Comparison — Overall Accuracy vs SNR', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_acc_vs_snr.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[evaluate] Comparison figure saved to {save_dir}/comparison_acc_vs_snr.png")

    # ── Save comparison summary CSV ───────────────────────────────────────────
    import csv
    summary_path = os.path.join(save_dir, 'comparison_summary.csv')
    first_snrs = sorted(list(all_results.values())[0]['snrs'])
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr'] + list(all_results.keys()))
        for snr in first_snrs:
            row = [snr]
            for res in all_results.values():
                row.append(res['acc'].get(snr, ''))
            writer.writerow(row)
    print(f"[evaluate] Comparison summary saved to {summary_path}")


def main():
    args = _parse_args()

    from src.utils.seed import set_all_seeds
    set_all_seeds(2016)

    if args.compare:
        if not args.configs or not args.weights:
            print("Error: --compare requires --configs and --weights")
            sys.exit(1)
        labels = args.labels or [os.path.basename(c) for c in args.configs]
        compare_experiments(args.configs, args.weights, labels, args.save_dir,
                            args.datasetpath)
    else:
        if not args.config or not args.weights:
            print("Error: provide --config and --weights")
            sys.exit(1)
        cfg = _load_config(args.config)
        evaluate_single(cfg, args.weights[0], args.datasetpath)


if __name__ == '__main__':
    main()
