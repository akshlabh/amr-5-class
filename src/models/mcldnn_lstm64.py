"""
mcldnn_lstm64.py — MCLDNN variant with LSTM hidden size 128 → 64
================================================================
Identical architecture to mcldnn.py EXCEPT both LSTM layers use
hidden_size=64 instead of 128.

Parameter reduction vs baseline:
  LSTM-1 : 4×(100×128 + 128² + 128) = 117,248  →  4×(100×64 + 64² + 64) = 42,240  (−75K)
  LSTM-2 : 4×(128×128 + 128² + 128) = 131,584  →  4×(64×64  + 64² + 64) = 33,024  (−99K)
  fc1    : 128×128+128 = 16,512                 →  64×128+128              = 8,320   (−8K)
  Total savings: ≈ −182K params

Expected total: ~222K  (baseline ~404K, attention ~205K)
"""

import os
import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate, LSTM
)


def MCLDNN_LSTM64(weights=None,
                  input_shape1=(2, 128),
                  input_shape2=(128, 1),
                  classes: int = 5,
                  dropout_rate: float = 0.5,
                  l2_dense: float = 1e-3,
                  l2_lstm: float = 1e-4,
                  **kwargs) -> Model:
    """
    MCLDNN with LSTM hidden size 64 (down from 128).

    Conv block is IDENTICAL to baseline mcldnn.py.
    Only the LSTM units and the first fc layer (which receives
    the 64-dim LSTM output) differ.

    Parameters
    ----------
    weights      : None or str path to .weights.h5 checkpoint
    input_shape1 : (2, 128)  IQ combined input
    input_shape2 : (128, 1)  per-channel input
    classes      : int       number of output classes
    dropout_rate : float     Dropout rate on fc1 and fc2
    l2_dense     : float     L2 regularisation on Dense layers
    l2_lstm      : float     L2 on LSTM kernel + recurrent weights

    Returns
    -------
    keras.Model  three inputs: [IQ_2D (2,128,1), I (128,1), Q (128,1)]
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError(f"Weights file not found: {weights}")

    dr = dropout_rate
    _l2_lstm_reg = keras.regularizers.L2(l2_lstm) if l2_lstm > 0 else None

    # ── Inputs ────────────────────────────────────────────────────────────────
    input1 = Input(shape=list(input_shape1) + [1], name='input1')  # (2, 128, 1)
    input2 = Input(shape=list(input_shape2),        name='input2')  # (128, 1)
    input3 = Input(shape=list(input_shape2),        name='input3')  # (128, 1)

    # ── Conv block (IDENTICAL to baseline) ───────────────────────────────────
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input1)
    # x1: (batch, 2, 128, 50)

    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input2)
    x2 = Reshape((1, 128, 50), name='reshape_i')(x2)
    # x2: (batch, 1, 128, 50)

    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3', kernel_initializer='glorot_uniform',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input3)
    x3 = Reshape((1, 128, 50), name='reshape_q')(x3)
    # x3: (batch, 1, 128, 50)

    x = concatenate([x2, x3], axis=1)               # (batch, 2, 128, 50)
    x = Conv2D(50, (1, 8), padding='same', activation='relu',
               name='conv2', kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x: (batch, 2, 128, 50)

    x = concatenate([x1, x], axis=-1)               # (batch, 2, 128, 100)
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv4', kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x: (batch, 1, 124, 100)

    x = Reshape((124, 100), name='reshape_lstm')(x)  # (batch, 124, 100)

    # ── LSTM stack — hidden size 64 (changed from 128) ────────────────────────
    x = LSTM(units=64, return_sequences=True, name='lstm_1',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)
    # x: (batch, 124, 64)

    x = LSTM(units=64, return_sequences=False, name='lstm_2',
             kernel_regularizer=_l2_lstm_reg,
             recurrent_regularizer=_l2_lstm_reg)(x)
    # x: (batch, 64)

    # ── Classifier head ───────────────────────────────────────────────────────
    # fc1 input is 64 (not 128) because LSTM output dim changed
    x = Dense(128, activation='selu', name='fc1',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop1')(x)
    x = Dense(128, activation='selu', name='fc2',
              kernel_regularizer=keras.regularizers.L2(l2_dense))(x)
    x = Dropout(dr, name='drop2')(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[input1, input2, input3],
                  outputs=x, name='MCLDNN_LSTM64')

    if weights is not None:
        model.load_weights(weights)
        print(f"[MCLDNN_LSTM64] Loaded weights from {weights}")

    return model


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    m = MCLDNN_LSTM64(classes=5)
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])
    m.summary()
    total = sum(p.numpy().size for p in m.trainable_weights)
    print(f"\nTotal trainable params: {total:,}")
