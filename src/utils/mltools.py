"""
mltools.py — Visualisation and evaluation utilities
====================================================
TF2 / Keras 3 compatible.  Key changes from original mltools.py:
  - history.history['acc']     → history.history['accuracy']
  - history.history['val_acc'] → history.history['val_accuracy']
  - All figures are saved to caller-specified directories (no hardcoded paths)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend (works on Kaggle / servers)
import matplotlib.pyplot as plt


# ── Training history plots ────────────────────────────────────────────────────

def show_history(history, save_dir: str = 'figures') -> None:
    """
    Save loss and accuracy curves and their raw values as .txt files.

    Parameters
    ----------
    history  : Keras History object returned by model.fit()
    save_dir : directory where figures and .txt files are written
    """
    os.makedirs(save_dir, exist_ok=True)

    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    epochs     = history.epoch

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure()
    plt.title('Training Loss')
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, val_loss,   label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'total_loss.png'), dpi=150)
    plt.close()

    # ── Accuracy curve ────────────────────────────────────────────────────────
    plt.figure()
    plt.title('Training Accuracy')
    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, val_acc,   label='val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'total_acc.png'), dpi=150)
    plt.close()

    # ── Save raw numbers ──────────────────────────────────────────────────────
    np.savetxt(os.path.join(save_dir, 'train_acc.txt'),  np.array(train_acc))
    np.savetxt(os.path.join(save_dir, 'val_acc.txt'),    np.array(val_acc))
    np.savetxt(os.path.join(save_dir, 'train_loss.txt'), np.array(train_loss))
    np.savetxt(os.path.join(save_dir, 'val_loss.txt'),   np.array(val_loss))

    print(f"[mltools] History saved to {save_dir}/")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def calculate_confusion_matrix(Y: np.ndarray,
                                Y_hat: np.ndarray,
                                classes: list):
    """
    Compute a normalised confusion matrix.

    Parameters
    ----------
    Y       : one-hot ground truth   shape (N, num_classes)
    Y_hat   : softmax predictions    shape (N, num_classes)
    classes : list of class name strings

    Returns
    -------
    confnorm : normalised confusion matrix  shape (num_classes, num_classes)
    correct  : int  number of correct predictions
    incorrect: int  number of incorrect predictions
    """
    n = len(classes)
    conf = np.zeros((n, n))

    for k in range(Y.shape[0]):
        true_cls = int(np.argmax(Y[k]))
        pred_cls = int(np.argmax(Y_hat[k]))
        conf[true_cls, pred_cls] += 1

    confnorm = np.zeros_like(conf)
    for i in range(n):
        row_sum = np.sum(conf[i])
        if row_sum > 0:
            confnorm[i] = conf[i] / row_sum

    correct   = int(np.sum(np.diag(conf)))
    incorrect = int(np.sum(conf) - correct)
    return confnorm, correct, incorrect


def plot_confusion_matrix(cm: np.ndarray,
                          labels: list,
                          title: str = 'Confusion Matrix',
                          save_filename: str = None,
                          cmap=None) -> None:
    """
    Plot and optionally save a normalised confusion matrix (values × 100).

    Parameters
    ----------
    cm            : normalised confusion matrix (output of calculate_confusion_matrix)
    labels        : list of class name strings
    title         : figure title
    save_filename : full path to save the figure; if None the figure is shown
    cmap          : matplotlib colormap (default: Blues)
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    n = len(labels)
    fig_size = max(4, n)
    plt.figure(figsize=(fig_size, int(fig_size * 0.85)), dpi=200)
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap,
               vmin=0, vmax=100)
    plt.colorbar(label='Accuracy (%)')
    plt.title(title, fontsize=11)

    tick_marks = np.arange(n)
    plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = int(round(cm[i, j] * 100))
            color = 'darkorange' if i == j else 'black'
            fsize = 7 if val == 100 else 9
            plt.text(j, i, val, ha='center', va='center',
                     fontsize=fsize, color=color)

    plt.tight_layout()
    if save_filename is not None:
        parent = os.path.dirname(save_filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close()


# ── Per-SNR accuracy curves ───────────────────────────────────────────────────

def plot_acc_vs_snr(acc: dict,
                    title: str = 'Overall Accuracy vs SNR',
                    save_filename: str = None) -> None:
    """
    Plot overall accuracy (averaged across classes) vs. SNR.

    Parameters
    ----------
    acc           : dict  {snr_value: accuracy_float}
    title         : figure title
    save_filename : full path to save the figure
    """
    snrs = sorted(acc.keys())
    vals = [acc[s] for s in snrs]

    plt.figure(figsize=(8, 5))
    plt.plot(snrs, vals, marker='o', linewidth=2, markersize=6)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_filename is not None:
        parent = os.path.dirname(save_filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close()


def plot_acc_per_class_vs_snr(acc_mod_snr: np.ndarray,
                               classes: list,
                               snrs: list,
                               save_dir: str = 'figures') -> None:
    """
    Plot per-class accuracy vs. SNR (one line per modulation type).

    Parameters
    ----------
    acc_mod_snr : array  shape (num_classes, num_snrs)
    classes     : list of class name strings
    snrs        : list of SNR values
    save_dir    : directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(classes):
        plt.plot(snrs, acc_mod_snr[i], label=cls, marker='o', markersize=4)
        for x, y in zip(snrs, acc_mod_snr[i]):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=6)

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Per-Class Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy vs SNR', fontsize=12)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_per_class_vs_snr.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[mltools] Per-class accuracy figure saved to {save_dir}/")
