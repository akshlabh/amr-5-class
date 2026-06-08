"""
analyze_branch_ablation.py — Branch ablation comparison plots and summary table
================================================================================
Reads acc_per_snr.csv produced by train.py for each model variant and generates:
  - Accuracy vs SNR curves (4 overlaid lines per figure)
  - Summary table (peak accuracy, peak SNR, overall accuracy)
  - Combined confusion matrix figure grid (if confusion_all_snrs.png exists)

Usage
-----
    python src/analyze_branch_ablation.py

Outputs
-------
    experiments/branch_ablation_comparison_5class.png
    experiments/branch_ablation_comparison_4class.png
    experiments/branch_ablation_comparison_confmat_5class.png   (if confusion PNGs exist)
    experiments/branch_ablation_comparison_confmat_4class.png   (if confusion PNGs exist)
"""

import os
import sys
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')            # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each entry: (label, results_dir, figure_dir, line_color, line_style)
MODELS_5CLASS = [
    ('Full MCLDNN',
     'experiments/5class_baseline/results',
     'experiments/5class_baseline/figures',
     'royalblue', '-'),
    ('I-only',
     'experiments/branch_5class/I_only/results',
     'experiments/branch_5class/I_only/figures',
     'crimson', '--'),
    ('Q-only',
     'experiments/branch_5class/Q_only/results',
     'experiments/branch_5class/Q_only/figures',
     'forestgreen', '--'),
    ('2D-only (IQ)',
     'experiments/branch_5class/IQ_only/results',
     'experiments/branch_5class/IQ_only/figures',
     'darkorange', ':'),
]

MODELS_4CLASS = [
    ('Full MCLDNN',
     'experiments/4class_ablation/results',
     'experiments/4class_ablation/figures',
     'royalblue', '-'),
    ('I-only',
     'experiments/branch_4class/I_only/results',
     'experiments/branch_4class/I_only/figures',
     'crimson', '--'),
    ('Q-only',
     'experiments/branch_4class/Q_only/results',
     'experiments/branch_4class/Q_only/figures',
     'forestgreen', '--'),
    ('2D-only (IQ)',
     'experiments/branch_4class/IQ_only/results',
     'experiments/branch_4class/IQ_only/figures',
     'darkorange', ':'),
]

OUTPUT_5CLASS_PNG     = 'experiments/branch_ablation_comparison_5class.png'
OUTPUT_4CLASS_PNG     = 'experiments/branch_ablation_comparison_4class.png'
OUTPUT_5CLASS_CONF    = 'experiments/branch_ablation_comparison_confmat_5class.png'
OUTPUT_4CLASS_CONF    = 'experiments/branch_ablation_comparison_confmat_4class.png'


# ---------------------------------------------------------------------------
# Helper: load acc_per_snr.csv
# ---------------------------------------------------------------------------

def load_acc_per_snr(results_dir: str):
    """
    Load accuracy vs SNR from {results_dir}/acc_per_snr.csv.

    Returns
    -------
    snrs : list[int]
    accs : list[float]
    or (None, None) if the file does not exist.
    """
    csv_path = os.path.join(results_dir, 'acc_per_snr.csv')
    if not os.path.exists(csv_path):
        warnings.warn(
            f"[analyze] acc_per_snr.csv not found — skipping: {csv_path}",
            UserWarning, stacklevel=2
        )
        return None, None

    snrs, accs = [], []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snrs.append(int(float(row['snr'])))
            accs.append(float(row['accuracy']))
    return snrs, accs


# ---------------------------------------------------------------------------
# Helper: compute summary stats
# ---------------------------------------------------------------------------

def compute_summary(snrs, accs):
    """
    Parameters
    ----------
    snrs : list[int]
    accs : list[float]

    Returns
    -------
    dict with keys: peak_accuracy, peak_snr_db, overall_accuracy
    """
    peak_idx      = int(np.argmax(accs))
    peak_accuracy = accs[peak_idx]
    peak_snr_db   = snrs[peak_idx]
    overall_acc   = float(np.mean(accs))
    return {
        'peak_accuracy':  peak_accuracy,
        'peak_snr_db':    peak_snr_db,
        'overall_accuracy': overall_acc,
    }


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_branch_comparison(model_specs, title: str, save_path: str):
    """
    Generate an Accuracy vs SNR figure with one curve per model variant.

    Parameters
    ----------
    model_specs : list of (label, results_dir, figure_dir, color, linestyle)
    title       : figure title string
    save_path   : path to save the PNG

    Returns
    -------
    summary_rows : list of dicts (one per available model)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_rows = []

    snrs_reference = None
    for (label, res_dir, _fig_dir, color, ls) in model_specs:
        snrs, accs = load_acc_per_snr(res_dir)
        if snrs is None:
            continue

        # ── SNR grid consistency check ────────────────────────────────────────
        if snrs_reference is None:
            snrs_reference = snrs
        elif snrs != snrs_reference:
            warnings.warn(
                f"[analyze] SNR grid mismatch for '{label}': "
                f"expected {snrs_reference}, got {snrs}. "
                "Results may not be comparable (different seeds → different test sets?).",
                UserWarning, stacklevel=2
            )

        ax.plot(snrs, [a * 100 for a in accs],
                color=color, linestyle=ls, linewidth=2.0,
                marker='o', markersize=4, label=label)

        stats = compute_summary(snrs, accs)
        stats['model'] = label
        summary_rows.append(stats)

    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] Saved: {save_path}")

    return summary_rows


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(rows, title: str):
    """
    Pretty-print a summary table to stdout.

    Parameters
    ----------
    rows  : list of dicts with keys: model, peak_accuracy, peak_snr_db, overall_accuracy
    title : heading string
    """
    if not rows:
        print(f"\n[{title}] No data available yet.\n")
        return

    col_w = [24, 14, 12, 16]
    header = (f"{'Model':<{col_w[0]}} "
              f"{'Peak Acc (%)':>{col_w[1]}} "
              f"{'Peak SNR':>{col_w[2]}} "
              f"{'Overall Acc (%)':>{col_w[3]}}")
    sep    = '-' * (sum(col_w) + len(col_w))

    print(f"\n{'='*len(sep)}")
    print(f"  {title}")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['model']:<{col_w[0]}} "
              f"{r['peak_accuracy']*100:>{col_w[1]}.2f} "
              f"{r['peak_snr_db']:>{col_w[2]}d} dB "
              f"{r['overall_accuracy']*100:>{col_w[3]}.2f}")
    print(sep)


# ---------------------------------------------------------------------------
# Confusion matrix grid
# ---------------------------------------------------------------------------

def plot_confmat_grid(model_specs, title: str, save_path: str):
    """
    If confusion_all_snrs.png exists in each model's figure directory, assemble
    them into a 1×N grid figure and save.

    Parameters
    ----------
    model_specs : list of (label, results_dir, figure_dir, color, linestyle)
    title       : suptitle string
    save_path   : path to save the combined PNG
    """
    available = []
    for (label, _res_dir, fig_dir, _color, _ls) in model_specs:
        img_path = os.path.join(fig_dir, 'confusion_all_snrs.png')
        if os.path.exists(img_path):
            available.append((label, img_path))

    if not available:
        print(f"[analyze] No confusion_all_snrs.png files found — skipping confmat grid.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, img_path) in zip(axes, available):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label, fontsize=12, fontweight='bold', pad=8)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[analyze] Saved confmat grid: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MCLDNN Branch Ablation Analysis")
    print("=" * 60)

    # ── 5-class comparison ────────────────────────────────────────────────────
    rows_5 = plot_branch_comparison(
        model_specs=MODELS_5CLASS,
        title='Branch Ablation — 5-class (BPSK / QPSK / 8PSK / QAM16 / QAM64)',
        save_path=OUTPUT_5CLASS_PNG,
    )
    print_summary_table(rows_5, title='5-class Branch Ablation Summary')

    # ── 4-class comparison ────────────────────────────────────────────────────
    rows_4 = plot_branch_comparison(
        model_specs=MODELS_4CLASS,
        title='Branch Ablation — 4-class (QPSK / 8PSK / QAM16 / QAM64)',
        save_path=OUTPUT_4CLASS_PNG,
    )
    print_summary_table(rows_4, title='4-class Branch Ablation Summary')

    # ── Confusion matrix grids (optional) ────────────────────────────────────
    plot_confmat_grid(
        model_specs=MODELS_5CLASS,
        title='Confusion Matrices (All SNRs) — 5-class Branch Ablation',
        save_path=OUTPUT_5CLASS_CONF,
    )
    plot_confmat_grid(
        model_specs=MODELS_4CLASS,
        title='Confusion Matrices (All SNRs) — 4-class Branch Ablation',
        save_path=OUTPUT_4CLASS_CONF,
    )

    print("\n[analyze] Done.\n")


if __name__ == '__main__':
    # Ensure repo root is on sys.path when run as a script
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    main()
