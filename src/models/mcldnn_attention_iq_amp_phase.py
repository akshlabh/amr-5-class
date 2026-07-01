"""
mcldnn_attention_iq_amp_phase.py — normal attention on [I, Q, A(t), phase(t)]
================================================================================

This is a separate normal-attention model, not a differential-attention model.

Input
-----
One time-sequence tensor of shape (batch, 128, 4):

    channel 0: I(t)
    channel 1: Q(t)
    channel 2: A(t)       = mean-normalized amplitude
    channel 3: phase(t)   = atan2(Q, I) / pi, bounded to [-1, 1]

Design
------
The model uses a multi-scale Conv1D front-end before normal self-attention.
This lets it see short and medium local structures from the four physical
channels before attention compares all time positions globally.

No class weighting is used by this model/config.  The loss is standard
categorical cross-entropy.
"""

from __future__ import annotations

import keras
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    concatenate,
)
from keras.models import Model
import keras.ops as ops
import numpy as np


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding of shape (1, seq_len, d_model)."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _build_graph(classes: int, dropout_rate: float, attn_dropout: float = 0.1):
    """Build graph shared by training model and attention extractor."""
    dr = dropout_rate
    l2_cnn = keras.regularizers.L2(1e-4)
    l2_dense = keras.regularizers.L2(3e-3)

    inp = Input(shape=(128, 4), name="input_iq_amp_phase")

    # Multi-scale local encoder. Different kernel widths see different radio
    # structures: very local sample geometry, medium symbol transitions, and
    # broader amplitude/phase patterns.
    b3 = Conv1D(48, 3, padding="same", activation="relu",
                name="ms_conv_k3", kernel_regularizer=l2_cnn)(inp)
    b7 = Conv1D(48, 7, padding="same", activation="relu",
                name="ms_conv_k7", kernel_regularizer=l2_cnn)(inp)
    b11 = Conv1D(48, 11, padding="same", activation="relu",
                 name="ms_conv_k11", kernel_regularizer=l2_cnn)(inp)
    x = concatenate([b3, b7, b11], axis=-1, name="ms_concat")
    x = Dropout(attn_dropout, name="ms_drop")(x)

    # Match the 124-step, 100-dim sequence used by the existing attention
    # models, making attention maps directly comparable.
    x = Conv1D(100, 5, padding="valid", activation="relu",
               name="conv_to_attention_tokens",
               kernel_regularizer=l2_cnn)(x)          # (batch, 124, 100)

    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100),
        dtype="float32",
    )
    x = x + pe

    mha = MultiHeadAttention(num_heads=4, key_dim=25, dropout=attn_dropout,
                             name="mha")
    attn_out, attn_weights = mha(
        query=x,
        value=x,
        key=x,
        return_attention_scores=True,
    )
    x = LayerNormalization(name="attn_norm")(attn_out + x)
    x = Dropout(attn_dropout, name="attn_drop")(x)

    ffn = Dense(256, activation="relu", name="ffn1",
                kernel_regularizer=l2_cnn)(x)
    ffn = Dropout(attn_dropout, name="ffn_drop")(ffn)
    ffn = Dense(100, name="ffn2", kernel_regularizer=l2_cnn)(ffn)
    x = LayerNormalization(name="ffn_norm")(ffn + x)

    # Three complementary summaries:
    #   mean: overall statistics,
    #   max: peak/local evidence,
    #   learned temporal pooling: lets the model choose important timesteps.
    x_mean = ops.mean(x, axis=1)
    x_max = ops.max(x, axis=1)
    pool_logits = Dense(1, name="temporal_pool_logits")(x)
    pool_weights = ops.softmax(pool_logits, axis=1)
    x_weighted = ops.sum(x * pool_weights, axis=1)
    context = concatenate([x_mean, x_max, x_weighted],
                          axis=-1,
                          name="context_mean_max_attnpool")

    out = Dense(128, activation="selu", name="fc1",
                kernel_regularizer=l2_dense)(context)
    out = Dropout(dr, name="drop1")(out)
    out = Dense(128, activation="selu", name="fc2",
                kernel_regularizer=l2_dense)(out)
    out = Dropout(dr, name="drop2")(out)
    softmax_out = Dense(classes, activation="softmax", name="softmax")(out)

    return [inp], softmax_out, attn_weights


def build_mcldnn_attention_iq_amp_phase(
    classes: int = 5,
    dropout_rate: float = 0.55,
    learning_rate: float = 5e-4,
    label_smoothing: float = 0.0,
) -> Model:
    """Build and compile the single-output training model."""
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out,
                  name="MCLDNN_Attention_IQ_Amp_Phase")
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    )
    return model


def build_mcldnn_attention_iq_amp_phase_extractor(
    classes: int = 5,
    dropout_rate: float = 0.55,
    weights_path: str | None = None,
) -> Model:
    """Build dual-output extractor: [softmax, attention weights]."""
    import os

    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(
        inputs=inputs,
        outputs=[softmax_out, attn_weights],
        name="MCLDNN_Attention_IQ_Amp_Phase_Extractor",
    )
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        extractor.load_weights(weights_path)
        print(f"[iq-amp-phase extractor] Loaded weights from {weights_path}")
    return extractor


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")

    model = build_mcldnn_attention_iq_amp_phase(classes=5)
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params; exceeds 300,000."
    )

    dummy = np.zeros((4, 128, 4), dtype="float32")
    pred = model.predict([dummy], verbose=0)
    print(f"Training output shape: {pred.shape}")
    assert pred.shape == (4, 5)

    extractor = build_mcldnn_attention_iq_amp_phase_extractor(classes=5)
    pred2, attn = extractor.predict([dummy], verbose=0)
    print(f"Extractor softmax shape: {pred2.shape}")
    print(f"Extractor attention shape: {attn.shape}")
    assert pred2.shape == (4, 5)
    assert attn.shape == (4, 4, 124, 124)
    print("All shape checks PASSED")
