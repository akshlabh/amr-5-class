"""
cost_sensitive.py — Cost-sensitive cross-entropy loss for AMR classification
=============================================================================

Motivation
----------
Standard cross-entropy treats every misclassification equally.  In a real
communication system, confusing QAM16 with QAM64 (and vice versa) is a more
costly error than, say, confusing BPSK with QPSK — because QAM16 and QAM64
are spectrally similar high-order constellations and the downstream demodulator
failure mode is more severe.

This module implements a differentiable cost-weighted cross-entropy that
penalises high-cost misclassification pairs more heavily during training.

Loss formula
------------
For sample i with one-hot true label y_i and predicted softmax p_i:

    w_i = Σ_j  C[true_class_i, j] × p_i[j]      ← expected cost (soft, differentiable)

    L_i = (1 + w_i) × CE(y_i, p_i)

The (1 + w_i) floor ensures a minimum gradient signal even for samples the
model already predicts correctly (where w_i → 0 because most probability mass
sits on the diagonal C[i,i]=0 entry).

Gradient behaviour
------------------
When the model places probability mass on a costly wrong class (e.g. p[QAM64]
is high when the true class is QAM16), w_i becomes large (≈α × p[QAM64]),
multiplying the CE gradient by (1 + α × p[QAM64]).  This forces the optimiser
to take a proportionally larger step for those confusing samples.

Label smoothing
---------------
Label smoothing is applied to the CE component independently and is fully
compatible with cost-weighting.  Set label_smoothing=0.0 to disable.

Usage
-----
    from src.losses.cost_sensitive import CostSensitiveCrossEntropy, build_cost_matrix

    cost_matrix = build_cost_matrix(
        classes=['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64'],
        high_cost_pairs=[('QAM16', 'QAM64'), ('QAM64', 'QAM16')],
        alpha=2.0,
    )
    loss_fn = CostSensitiveCrossEntropy(cost_matrix, label_smoothing=0.1)
    model.compile(loss=loss_fn, optimizer=..., metrics=['accuracy'])
"""

import numpy as np
import keras


# ─────────────────────────────────────────────────────────────────────────────
# Cost matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_matrix(classes: list,
                      high_cost_pairs: list,
                      alpha: float = 2.0) -> np.ndarray:
    """
    Build a (n_classes × n_classes) cost matrix.

    Parameters
    ----------
    classes         : ordered list of class name strings
                      (must match the one-hot encoding order used during training)
    high_cost_pairs : list of (true_class_name, pred_class_name) tuples that
                      should receive penalty α instead of 1.
                      Example: [('QAM16', 'QAM64'), ('QAM64', 'QAM16')]
    alpha           : float  penalty multiplier for high-cost pairs (default 2.0)

    Returns
    -------
    C : np.ndarray  shape (n_classes, n_classes)  float32
        C[i, i] = 0   (no cost for correct prediction)
        C[i, j] = 1   (default cost for wrong prediction)
        C[i, j] = α   (elevated cost for specified pairs)
    """
    n = len(classes)
    C = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(C, 0.0)

    idx = {cls: i for i, cls in enumerate(classes)}
    for true_cls, pred_cls in high_cost_pairs:
        if true_cls not in idx:
            raise ValueError(f"Class '{true_cls}' not in classes list: {classes}")
        if pred_cls not in idx:
            raise ValueError(f"Class '{pred_cls}' not in classes list: {classes}")
        i, j = idx[true_cls], idx[pred_cls]
        if i == j:
            raise ValueError(f"High-cost pair ({true_cls}, {pred_cls}) is a diagonal "
                             "entry — cost must be 0 for correct predictions.")
        C[i, j] = float(alpha)

    return C


def print_cost_matrix(C: np.ndarray, classes: list) -> None:
    """Pretty-print the cost matrix to stdout."""
    n = len(classes)
    col_w = max(len(c) for c in classes) + 2
    header = f"{'True \\ Pred':<{col_w}}" + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(classes):
        row = f"{cls:<{col_w}}"
        for j in range(n):
            val = C[i, j]
            cell = "0" if val == 0 else (f"α={val:.1f}" if val != 1.0 else "1")
            row += f"{cell:>{col_w}}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras loss
# ─────────────────────────────────────────────────────────────────────────────

class CostSensitiveCrossEntropy(keras.losses.Loss):
    """
    Cost-sensitive cross-entropy loss.

    For each sample, the standard cross-entropy is multiplied by
    (1 + expected_cost), where expected_cost is the dot product of the
    predicted probability distribution with the cost row corresponding to
    the sample's true class.

    This is fully differentiable: expected_cost = Σ_j C[y_i, j] × p_i[j]
    uses only standard tensor operations.

    Parameters
    ----------
    cost_matrix     : np.ndarray  (n_classes, n_classes)  float32
                      Built with build_cost_matrix().
    label_smoothing : float  applied to CE component (default 0.1)
    name            : str    loss name shown in Keras logs
    """

    def __init__(self,
                 cost_matrix: np.ndarray,
                 label_smoothing: float = 0.1,
                 name: str = 'cost_sensitive_ce',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._cost_matrix_np = cost_matrix.astype(np.float32)
        self.label_smoothing = label_smoothing
        # Store as a non-trainable weight so it moves to GPU automatically
        self._cost_matrix = keras.ops.convert_to_tensor(
            self._cost_matrix_np, dtype='float32')

    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Tensor  shape (batch, n_classes)  one-hot ground truth
        y_pred : Tensor  shape (batch, n_classes)  softmax probabilities

        Returns
        -------
        Tensor  scalar mean loss over the batch
        """
        import keras.ops as ops

        # ── Standard cross-entropy (with optional label smoothing) ────────────
        ce = keras.losses.categorical_crossentropy(
            y_true, y_pred,
            label_smoothing=self.label_smoothing,
        )
        # ce shape: (batch,)

        # ── Expected cost per sample ──────────────────────────────────────────
        # cost_row[i] = C[true_class_i, :] — the cost of each possible prediction
        # given the true class of sample i.
        # Computed as: y_true (batch, n_classes) @ C (n_classes, n_classes)
        # = (batch, n_classes)  where row i contains C[true_class_i, :]
        cost_row = ops.matmul(y_true, self._cost_matrix)  # (batch, n_classes)

        # expected_cost[i] = Σ_j cost_row[i, j] × p_i[j]
        # = dot(C[y_i, :], p_i)  — smooth proxy for C[y_i, argmax(p_i)]
        expected_cost = ops.sum(cost_row * y_pred, axis=-1)  # (batch,)

        # ── Weighted loss ─────────────────────────────────────────────────────
        # Floor at 1 so correct-prediction samples still receive gradient signal
        weighted_ce = (1.0 + expected_cost) * ce             # (batch,)

        return ops.mean(weighted_ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'cost_matrix':     self._cost_matrix_np.tolist(),
            'label_smoothing': self.label_smoothing,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        config['cost_matrix'] = np.array(config['cost_matrix'], dtype=np.float32)
        return cls(**config)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    C = build_cost_matrix(
        classes=classes,
        high_cost_pairs=[('QAM16', 'QAM64'), ('QAM64', 'QAM16')],
        alpha=2.0,
    )

    print("Cost matrix (alpha=2.0):")
    print_cost_matrix(C, classes)
    print()

    loss_fn = CostSensitiveCrossEntropy(C, label_smoothing=0.1)

    # Batch of 3 samples: BPSK correct, QAM16 confused as QAM64, QAM64 confused as QAM16
    y_true = np.array([
        [1, 0, 0, 0, 0],   # BPSK  (correct)
        [0, 0, 0, 1, 0],   # QAM16 (predicted QAM64 → high cost)
        [0, 0, 0, 0, 1],   # QAM64 (predicted QAM16 → high cost)
    ], dtype=np.float32)

    y_pred_correct  = np.array([[0.9, 0.025, 0.025, 0.025, 0.025],
                                [0.025, 0.025, 0.025, 0.9, 0.025],
                                [0.025, 0.025, 0.025, 0.025, 0.9]], dtype=np.float32)

    y_pred_confused = np.array([[0.9, 0.025, 0.025, 0.025, 0.025],   # BPSK → correct
                                [0.025, 0.025, 0.025, 0.025, 0.9],   # QAM16 → QAM64
                                [0.025, 0.025, 0.025, 0.9, 0.025]], dtype=np.float32)  # QAM64 → QAM16

    l_correct  = float(loss_fn(y_true, y_pred_correct))
    l_confused = float(loss_fn(y_true, y_pred_confused))

    print(f"Loss (all correct predictions) : {l_correct:.4f}")
    print(f"Loss (QAM16↔QAM64 confused)    : {l_confused:.4f}")
    print(f"Cost amplification ratio       : {l_confused / l_correct:.2f}×")
    assert l_confused > l_correct, "Cost-sensitive loss must be higher for confused predictions!"
    print("\n✓ Smoke test passed — cost-sensitive loss is higher for costly confusion pairs.")
