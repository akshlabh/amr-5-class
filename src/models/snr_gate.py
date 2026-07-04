"""
snr_gate.py — Lightweight binary SNR-region classifier
======================================================

This model is not a modulation classifier.  It is a small routing model for
the hybrid attention pipeline:

    class 0: low SNR  (SNR <= 0 dB)  -> route to differential attention
    class 1: high SNR (SNR >  0 dB)  -> route to normal attention

Input is a compact analytic feature vector extracted from each normalized IQ
window by ``extract_snr_gate_features``.
"""

from __future__ import annotations

import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model


def build_snr_gate(input_dim: int = 29,
                   classes: int = 2,
                   dropout_rate: float = 0.15,
                   learning_rate: float = 1e-3) -> Model:
    """
    Build and compile a small MLP for SNR-region classification.

    Parameters
    ----------
    input_dim : int
        Number of analytic input features.
    classes : int
        Output classes. Keep at 2 for low/high SNR routing.
    dropout_rate : float
        Dropout probability in the hidden layers.
    learning_rate : float
        Adam learning rate.
    """
    inp = Input(shape=(input_dim,), name="snr_features")

    x = BatchNormalization(name="bn_input")(inp)
    x = Dense(64, activation="relu",
              kernel_regularizer=keras.regularizers.L2(1e-4),
              name="fc1")(x)
    x = Dropout(dropout_rate, name="drop1")(x)
    x = Dense(32, activation="relu",
              kernel_regularizer=keras.regularizers.L2(1e-4),
              name="fc2")(x)
    x = Dropout(dropout_rate, name="drop2")(x)
    x = Dense(16, activation="relu",
              kernel_regularizer=keras.regularizers.L2(1e-4),
              name="fc3")(x)
    out = Dense(classes, activation="softmax", name="softmax")(x)

    model = Model(inp, out, name="Lightweight_SNR_Gate")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_snr_gate()
    m.summary()
