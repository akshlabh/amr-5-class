"""
mcldnn_attention_strong.py
==========================

Higher-capacity MCLDNN attention model for the 11-class RadioML experiment.

Why this model exists
---------------------
The normal attention model is intentionally small, while the original MCLDNN
baseline has a larger 2-layer LSTM sequence model.  At high SNR, the LSTM often
wins because it has both more capacity and a natural notion of temporal order.

This model keeps the same 3-branch MCLDNN CNN front-end, then gives attention
the missing pieces:

1. projection from 100 -> d_model
2. fixed sinusoidal positional encoding
3. two residual Transformer-style self-attention blocks
4. gated attention pooling plus mean/max statistics
5. larger but regularized classifier head

It is deliberately not a patch-work physical-feature model.  It is a stronger,
fairer attention baseline.
"""

from __future__ import annotations

import os

import keras
import keras.ops as ops
import numpy as np
from keras.layers import (
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
    Softmax,
    concatenate,
)
from keras.models import Model


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Return fixed sinusoidal positional encoding of shape (1, seq_len, d_model)."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _transformer_block(
    x,
    *,
    block_id: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    ffn_dim: int,
    attn_dropout: float,
    l2_weight: float,
    return_scores: bool = False,
):
    """Pre-norm residual self-attention block."""
    reg = keras.regularizers.L2(l2_weight) if l2_weight > 0 else None

    x_norm = LayerNormalization(name=f"block{block_id}_attn_prenorm")(x)
    mha = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attn_dropout,
        kernel_regularizer=reg,
        name=f"block{block_id}_mha",
    )
    attn_out, attn_scores = mha(
        query=x_norm,
        value=x_norm,
        key=x_norm,
        return_attention_scores=True,
    )
    attn_out = Dropout(attn_dropout, name=f"block{block_id}_attn_dropout")(attn_out)
    x = x + attn_out

    ffn_norm = LayerNormalization(name=f"block{block_id}_ffn_prenorm")(x)
    ffn = Dense(
        ffn_dim,
        activation="gelu",
        kernel_regularizer=reg,
        name=f"block{block_id}_ffn1",
    )(ffn_norm)
    ffn = Dropout(attn_dropout, name=f"block{block_id}_ffn_dropout")(ffn)
    ffn = Dense(
        d_model,
        kernel_regularizer=reg,
        name=f"block{block_id}_ffn2",
    )(ffn)
    ffn = Dropout(attn_dropout, name=f"block{block_id}_ffn_out_dropout")(ffn)
    x = x + ffn

    if return_scores:
        return x, attn_scores
    return x, None


def _build_graph(
    classes: int,
    dropout_rate: float,
    *,
    d_model: int = 128,
    num_heads: int = 4,
    key_dim: int = 32,
    ffn_dim: int = 256,
    n_blocks: int = 2,
    attn_dropout: float = 0.15,
    l2_conv: float = 1e-4,
    l2_transformer: float = 1e-4,
    l2_dense: float = 1e-3,
):
    reg_conv = keras.regularizers.L2(l2_conv) if l2_conv > 0 else None
    reg_dense = keras.regularizers.L2(l2_dense) if l2_dense > 0 else None

    input1 = Input(shape=(2, 128, 1), name="input1")
    input2 = Input(shape=(128, 1), name="input2")
    input3 = Input(shape=(128, 1), name="input3")

    # Same 3-branch MCLDNN CNN front-end.
    x1 = Conv2D(
        50,
        (2, 8),
        padding="same",
        activation="relu",
        kernel_regularizer=reg_conv,
        name="conv1_1",
    )(input1)

    x2 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        kernel_regularizer=reg_conv,
        name="conv1_2",
    )(input2)
    x2 = Reshape((1, 128, 50), name="reshape_x2")(x2)

    x3 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        kernel_regularizer=reg_conv,
        name="conv1_3",
    )(input3)
    x3 = Reshape((1, 128, 50), name="reshape_x3")(x3)

    x = concatenate([x2, x3], axis=1, name="concat_iq_branches")
    x = Conv2D(
        50,
        (1, 8),
        padding="same",
        activation="relu",
        kernel_regularizer=reg_conv,
        name="conv2",
    )(x)

    x = concatenate([x1, x], axis=-1, name="concat_joint_branch")
    x = Conv2D(
        100,
        (2, 5),
        padding="valid",
        activation="relu",
        kernel_regularizer=reg_conv,
        name="conv4",
    )(x)
    x = Reshape((124, 100), name="reshape_final")(x)

    # Stronger attention sequence path.
    x = Dense(
        d_model,
        activation=None,
        kernel_regularizer=keras.regularizers.L2(l2_transformer),
        name="sequence_projection",
    )(x)

    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=d_model),
        dtype="float32",
    )
    x = x + pe
    x = Dropout(attn_dropout, name="positional_dropout")(x)

    last_scores = None
    for block_id in range(1, n_blocks + 1):
        x, scores = _transformer_block(
            x,
            block_id=block_id,
            d_model=d_model,
            num_heads=num_heads,
            key_dim=key_dim,
            ffn_dim=ffn_dim,
            attn_dropout=attn_dropout,
            l2_weight=l2_transformer,
            return_scores=(block_id == n_blocks),
        )
        if scores is not None:
            last_scores = scores

    x = LayerNormalization(name="final_sequence_norm")(x)

    # Gated temporal pooling: learn which timesteps matter, then preserve global
    # statistics too.  This gives attention a better sequence summary than a
    # single last hidden state replacement.
    gate_logits = Dense(1, name="temporal_gate_logits")(x)
    gate = Softmax(axis=1, name="temporal_gate")(gate_logits)
    pooled_gate = ops.sum(x * gate, axis=1)
    pooled_mean = ops.mean(x, axis=1)
    pooled_max = ops.max(x, axis=1)
    context = concatenate(
        [pooled_gate, pooled_mean, pooled_max],
        axis=-1,
        name="pooled_gate_mean_max",
    )

    out = Dense(
        160,
        activation="selu",
        kernel_regularizer=reg_dense,
        name="fc1",
    )(context)
    out = Dropout(dropout_rate, name="drop1")(out)
    out = Dense(
        128,
        activation="selu",
        kernel_regularizer=reg_dense,
        name="fc2",
    )(out)
    out = Dropout(dropout_rate, name="drop2")(out)
    softmax_out = Dense(classes, activation="softmax", name="softmax")(out)

    return [input1, input2, input3], softmax_out, last_scores


def build_mcldnn_attention_strong(
    classes: int = 11,
    dropout_rate: float = 0.45,
    learning_rate: float = 5e-4,
) -> Model:
    """Build and compile the stronger attention training model."""
    inputs, softmax_out, _ = _build_graph(classes=classes, dropout_rate=dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out, name="MCLDNN_Attention_Strong")
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    )
    return model


def build_mcldnn_attention_strong_extractor(
    classes: int = 11,
    dropout_rate: float = 0.45,
    weights_path: str | None = None,
) -> Model:
    """Build extractor returning [softmax, final-block attention scores]."""
    inputs, softmax_out, attn_scores = _build_graph(
        classes=classes,
        dropout_rate=dropout_rate,
    )
    model = Model(
        inputs=inputs,
        outputs=[softmax_out, attn_scores],
        name="MCLDNN_Attention_Strong_Extractor",
    )
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    m = build_mcldnn_attention_strong(classes=11)
    m.summary()
    print(f"Params: {m.count_params():,}")
