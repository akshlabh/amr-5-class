"""
mcldnn.py — MCLDNN model (TF2 / Keras 3 compatible)
====================================================
Architecture is identical to the original paper / AMR-Benchmark implementation.
Changes from the original rmlmodels/MCLDNN.py:
  - CuDNNLSTM  → LSTM  (Keras 3 LSTM auto-uses cuDNN when GPU is present)
  - from keras.layers.convolutional import Conv2D  → from keras.layers import Conv2D
  - Reshape([-1, 128, 50])  →  Reshape((1, 128, 50))  (explicit, no -1 ambiguity)
  - classes parameter defaults to 5 (was 11)
  - Weights loading accepts None (random init) or a path string
  - fc1/fc2 Dense: kernel_regularizer=L2(1e-4) added to combat overfitting
    (Without regularization, train_acc reaches 0.72 while val_acc plateaus at 0.57;
     L2 weight decay keeps the model from memorising the training split.)
  NOTE — AlphaDropout + lecun_normal was tried but reverted: SELU self-normalisation
    requires the ENTIRE network to be self-normalizing; the LSTM output feeding fc1
    is not self-normalized (LSTM uses tanh/sigmoid), so AlphaDropout introduces a
    distribution mismatch that slows convergence and lowers the val-accuracy ceiling.
"""

import os
import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    MaxPool1D, Dropout, Flatten, Reshape,
    concatenate, LSTM
)


def MCLDNN(weights=None,
           input_shape1=(2, 128),
           input_shape2=(128, 1),
           classes: int = 5,
           dropout_rate: float = 0.5,
           l2_dense: float = 1e-3,
           l2_lstm: float = 1e-4,
           **kwargs) -> Model:
    """
    Build the MCLDNN model.

    Parameters
    ----------
    weights      : None or str path to .h5 / .weights.h5 file
    input_shape1 : tuple  shape of the 2D IQ input  (2, 128)
    input_shape2 : tuple  shape of each 1D branch   (128, 1)
    classes      : int    number of output classes   (5 for primary, 4 for ablation)
    dropout_rate : float  dropout probability (0.5 in original paper)
    l2_dense     : float  L2 on Dense fc1/fc2 (default 1e-3, original paper).
                          Set to 1e-4 for 4-class: the smaller gradient signal
                          from 4 similar QAM classes is overwhelmed by 1e-3,
                          causing the model to collapse to uniform output.
    l2_lstm      : float  L2 on LSTM kernel + recurrent weights (default 1e-4).
                          Set to 0.0 for 4-class: the recurrent_regularizer
                          shrinks spectral_radius(W_h), worsening BPTT vanishing.
                          Removing it preserves gradient flow through 124 time steps.

    Returns
    -------
    keras.Model with three inputs: [IQ_2D, I_branch, Q_branch]
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError(
            f"Weights file not found: {weights}\n"
            "Pass weights=None for random initialisation."
        )

    dr = dropout_rate

    # ── Inputs ────────────────────────────────────────────────────────────────
    input1 = Input(shape=list(input_shape1) + [1], name='input1')   # (2, 128, 1)
    input2 = Input(shape=list(input_shape2),        name='input2')   # (128, 1)
    input3 = Input(shape=list(input_shape2),        name='input3')   # (128, 1)

    # ── Branch 1: 2D CNN on the combined IQ frame ─────────────────────────────
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input1)
    # x1 shape: (batch, 2, 128, 50)

    # ── Branch 2: 1D CNN on I channel ─────────────────────────────────────────
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input2)
    # x2 shape: (batch, 128, 50)
    x2 = Reshape((1, 128, 50), name='reshape_i')(x2)
    # x2 shape: (batch, 1, 128, 50)

    # ── Branch 3: 1D CNN on Q channel ─────────────────────────────────────────
    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input3)
    # x3 shape: (batch, 128, 50)
    x3 = Reshape((1, 128, 50), name='reshape_q')(x3)
    # x3 shape: (batch, 1, 128, 50)

    # ── Combine I + Q branches, apply shared 2D conv ──────────────────────────
    x = concatenate([x2, x3], axis=1)               # (batch, 2, 128, 50)
    x = Conv2D(50, (1, 8), padding='same', activation='relu',
               name='conv2', kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x shape: (batch, 2, 128, 50)

    # ── Merge with Branch 1 ───────────────────────────────────────────────────
    x = concatenate([x1, x], axis=-1)               # (batch, 2, 128, 100)
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv4', kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x shape: (batch, 1, 124, 100)

    # ── Reshape for LSTM ──────────────────────────────────────────────────────
    x = Reshape((124, 100), name='reshape_lstm')(x)  # (batch, 124, 100)

    # ── LSTM stack ────────────────────────────────────────────────────────────
    # l2_lstm=1e-4 for 5-class (original paper); l2_lstm=0 for 4-class.
    # The recurrent_regularizer shrinks spectral_radius(W_h) toward 0,
    # worsening vanishing gradients through 124 BPTT steps on the harder task.
    _l2_lstm_reg = keras.regularizers.L2(l2_lstm) if l2_lstm > 0 else None
    x = LSTM(units=128, return_sequences=True, name='lstm_1',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)
    x = LSTM(units=128, return_sequences=False, name='lstm_2',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)

    # ── Classifier head ───────────────────────────────────────────────────────
    # L2 on all layers (Conv L2=1e-4, LSTM L2=1e-4, Dense L2=1e-3) to prevent the
    # catastrophic overfitting seen in previous runs (train_acc→99%, val_acc→60%).
    x = Dense(128, activation='selu', name='fc1',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop1')(x)
    x = Dense(128, activation='selu', name='fc2',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop2')(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=x, name='MCLDNN')

    if weights is not None:
        model.load_weights(weights)
        print(f"[MCLDNN] Loaded weights from {weights}")

    return model


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== 5-class MCLDNN ===")
    m5 = MCLDNN(classes=5)
    m5.compile(loss='categorical_crossentropy',
               optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               metrics=['accuracy'])
    m5.summary()

    print("\n=== 4-class MCLDNN (ablation) ===")
    m4 = MCLDNN(classes=4)
    m4.summary()
