"""
mcldnn_ablation.py — Configurable single-branch MCLDNN variants for ablation study
====================================================================================
Implements four variants of the MCLDNN architecture, selectable via the ``branch``
parameter of :func:`build_mcldnn_branch`.

Variants
--------
'full'
    Delegates to the full three-input :func:`~src.models.mcldnn.MCLDNN` model
    (IQ 2D branch + I 1D branch + Q 1D branch combined).  This is the original
    paper architecture and serves as the ceiling baseline.

'I'
    Uses only the in-phase (I) channel.
    Input:  (N, 128, 1)
    Path:   Conv1D(50, 8, causal, relu, L2=1e-4)
            → Reshape(1, 128, 50)
            → Conv2D(100, (1, 5), valid, relu, L2=1e-4)   # kernel shape fix: (1,5) not (2,5)
            → Reshape(124, 100)
            → LSTM stack → Dense head
    Output: (N, classes) softmax

'Q'
    Identical to 'I' but operates on the quadrature (Q) channel.
    Input:  (N, 128, 1)
    Same conv/LSTM/Dense structure as 'I'.

'IQ'
    Uses both channels stacked as a 2D image; no 1D branch.
    Input:  (N, 2, 128, 1)
    Path:   Conv2D(50, (2, 8), same, relu, L2=1e-4)
            → Conv2D(100, (2, 5), valid, relu, L2=1e-4)
            → Reshape(124, 100)
            → LSTM stack → Dense head
    Output: (N, classes) softmax

Kernel-shape fix (important)
----------------------------
In the single-branch 'I' and 'Q' variants, after Reshape((1, 128, 50)) the
spatial height is 1, so the Conv2D height kernel must also be 1.  Using (2, 5)
would produce an output height of 0 (invalid).  The correct kernel is (1, 5),
yielding output shape (batch, 1, 124, 100) which Reshape safely maps to
(batch, 124, 100).

Shared head (all variants except 'full')
-----------------------------------------
After the branch-specific feature extraction:
    LSTM(128, return_sequences=True,  kernel_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4))
    LSTM(128, return_sequences=False, kernel_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-4))
    Dense(128, selu, L2=1e-3) → Dropout(dr)
    Dense(128, selu, L2=1e-3) → Dropout(dr)
    Dense(classes, softmax)

Regularization schedule
-----------------------
Conv layers  : kernel_regularizer = L2(1e-4)
LSTM layers  : kernel_regularizer = recurrent_regularizer = L2(1e-4)
Dense fc1/fc2: kernel_regularizer = L2(1e-3)   (heavier — these are the widest layers)
"""

import os
import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, LSTM
)


# ---------------------------------------------------------------------------
# Shared LSTM + classifier head
# ---------------------------------------------------------------------------

def _lstm_head(x, classes: int, dr: float, l2_dense: float = 1e-3,
               l2_lstm: float = 1e-4):
    """
    Apply the shared LSTM stack and Dense classifier on top of a
    feature tensor of shape (batch, 124, 100).

    Parameters
    ----------
    x        : tensor of shape (batch, 124, 100)
    classes  : number of output classes
    dr       : dropout rate
    l2_dense : L2 penalty on Dense fc1/fc2 layers.
               Default 1e-3 (original paper value, works for 5-class).
               Use 1e-4 for 4-class: the weaker 4-class gradient signal
               is overwhelmed by the 1e-3 penalty, causing the model to
               collapse to uniform output and never escape.
    l2_lstm  : L2 on LSTM kernel + recurrent weights (default 1e-4).
               Set to 0.0 for 4-class: the recurrent_regularizer shrinks
               spectral_radius(W_h), worsening vanishing gradients through
               the 124 BPTT time steps of the harder 4-class task.

    Returns
    -------
    output tensor (batch, classes)
    """
    _l2_lstm_reg = keras.regularizers.L2(l2_lstm) if l2_lstm > 0 else None
    x = LSTM(units=128, return_sequences=True, name='lstm_1',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)
    x = LSTM(units=128, return_sequences=False, name='lstm_2',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)
    x = Dense(128, activation='selu', name='fc1',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop1')(x)
    x = Dense(128, activation='selu', name='fc2',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop2')(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)
    return x


# ---------------------------------------------------------------------------
# Branch builders
# ---------------------------------------------------------------------------

def _build_I_branch(classes: int, dr: float, l2_dense: float = 1e-3,
                    l2_lstm: float = 1e-4) -> Model:
    """
    I-only branch.

    Input shape : (128, 1)   — in-phase channel
    Architecture:
        Conv1D(50, 8, causal, relu, L2=1e-4)  → (128, 50)
        Reshape(1, 128, 50)                    → (1, 128, 50)
        Conv2D(100, (1,5), valid, relu, L2=1e-4) → (1, 124, 100)
        Reshape(124, 100)                      → (124, 100)
        LSTM stack + Dense head
    """
    inp = Input(shape=(128, 1), name='input_I')

    x = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1_I',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(inp)
    # shape: (batch, 128, 50)

    x = Reshape((1, 128, 50), name='reshape_I')(x)
    # shape: (batch, 1, 128, 50)

    # kernel height must be 1 (spatial height = 1 after reshape)
    x = Conv2D(100, (1, 5), padding='valid', activation='relu',
               name='conv2_I',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # shape: (batch, 1, 124, 100)

    x = Reshape((124, 100), name='reshape_lstm')(x)
    # shape: (batch, 124, 100)

    out = _lstm_head(x, classes=classes, dr=dr, l2_dense=l2_dense,
                     l2_lstm=l2_lstm)

    return Model(inputs=inp, outputs=out, name='MCLDNN_I')


def _build_Q_branch(classes: int, dr: float, l2_dense: float = 1e-3,
                    l2_lstm: float = 1e-4) -> Model:
    """
    Q-only branch.

    Input shape : (128, 1)   — quadrature channel
    Architecture: identical to I-branch but named 'Q'.
    """
    inp = Input(shape=(128, 1), name='input_Q')

    x = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1_Q',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(inp)
    # shape: (batch, 128, 50)

    x = Reshape((1, 128, 50), name='reshape_Q')(x)
    # shape: (batch, 1, 128, 50)

    # kernel height must be 1 (spatial height = 1 after reshape)
    x = Conv2D(100, (1, 5), padding='valid', activation='relu',
               name='conv2_Q',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # shape: (batch, 1, 124, 100)

    x = Reshape((124, 100), name='reshape_lstm')(x)
    # shape: (batch, 124, 100)

    out = _lstm_head(x, classes=classes, dr=dr, l2_dense=l2_dense,
                     l2_lstm=l2_lstm)

    return Model(inputs=inp, outputs=out, name='MCLDNN_Q')


def _build_IQ_branch(classes: int, dr: float, l2_dense: float = 1e-3,
                     l2_lstm: float = 1e-4) -> Model:
    """
    IQ-combined branch (2D CNN only, no 1D sub-branches).

    Input shape : (2, 128, 1)  — stacked I and Q channels
    Architecture:
        Conv2D(50,  (2, 8), same,  relu, L2=1e-4)   → (2, 128, 50)
        Conv2D(100, (2, 5), valid, relu, L2=1e-4)   → (1, 124, 100)
        Reshape(124, 100)                            → (124, 100)
        LSTM stack + Dense head
    """
    inp = Input(shape=(2, 128, 1), name='input_IQ')

    x = Conv2D(50, (2, 8), padding='same', activation='relu',
               name='conv1_IQ',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(inp)
    # shape: (batch, 2, 128, 50)

    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv2_IQ',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # shape: (batch, 1, 124, 100)

    x = Reshape((124, 100), name='reshape_lstm')(x)
    # shape: (batch, 124, 100)

    out = _lstm_head(x, classes=classes, dr=dr, l2_dense=l2_dense,
                     l2_lstm=l2_lstm)

    return Model(inputs=inp, outputs=out, name='MCLDNN_IQ')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mcldnn_branch(classes: int = 5,
                        branch: str = 'full',
                        dropout_rate: float = 0.5,
                        l2_dense: float = 1e-3,
                        l2_lstm: float = 1e-4,
                        weights: str = None) -> keras.Model:
    """
    Build an MCLDNN model for the specified branch configuration.

    Parameters
    ----------
    classes      : int   — number of modulation classes (default 5)
    branch       : str   — one of {'full', 'I', 'Q', 'IQ'}
                          'full' → delegates to the original three-input MCLDNN
                          'I'    → I-channel only (Conv1D → Conv2D(1,5) → LSTM)
                          'Q'    → Q-channel only (same architecture as 'I')
                          'IQ'   → both channels, 2D CNN only, no 1D sub-branches
    dropout_rate : float — dropout probability applied in Dense head (default 0.5)
    weights      : str or None — path to a .weights.h5 checkpoint to load;
                                 None means random initialisation

    Returns
    -------
    keras.Model
        Compiled-ready model with the selected branch architecture.
        Inputs differ by branch:
          'full': [IQ_2D (N,2,128,1), I_1D (N,128,1), Q_1D (N,128,1)]
          'I'   : I_1D (N,128,1)
          'Q'   : Q_1D (N,128,1)
          'IQ'  : IQ_2D (N,2,128,1)

    Raises
    ------
    ValueError
        If ``branch`` is not one of the four recognised values.
    ValueError
        If ``weights`` is provided but the file does not exist.
    """
    valid_branches = {'full', 'I', 'Q', 'IQ'}
    if branch not in valid_branches:
        raise ValueError(
            f"Unknown branch '{branch}'. Must be one of {valid_branches}."
        )

    if weights is not None and not os.path.exists(weights):
        raise ValueError(
            f"Weights file not found: {weights}\n"
            "Pass weights=None for random initialisation."
        )

    if branch == 'full':
        # Delegate entirely to the original MCLDNN implementation
        from src.models.mcldnn import MCLDNN
        return MCLDNN(weights=weights, classes=classes,
                      dropout_rate=dropout_rate, l2_dense=l2_dense,
                      l2_lstm=l2_lstm)

    dr = dropout_rate

    if branch == 'I':
        model = _build_I_branch(classes=classes, dr=dr, l2_dense=l2_dense,
                                l2_lstm=l2_lstm)
    elif branch == 'Q':
        model = _build_Q_branch(classes=classes, dr=dr, l2_dense=l2_dense,
                                l2_lstm=l2_lstm)
    else:  # 'IQ'
        model = _build_IQ_branch(classes=classes, dr=dr, l2_dense=l2_dense,
                                 l2_lstm=l2_lstm)

    if weights is not None:
        model.load_weights(weights)
        print(f"[MCLDNN_{branch}] Loaded weights from {weights}")

    return model


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np

    for br in ('I', 'Q', 'IQ'):
        print(f"\n=== branch='{br}' (5 classes) ===")
        m = build_mcldnn_branch(classes=5, branch=br)
        m.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
        m.summary()

    print("\n=== branch='full' (5 classes) — delegates to MCLDNN() ===")
    mf = build_mcldnn_branch(classes=5, branch='full')
    mf.summary()
