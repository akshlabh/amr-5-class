"""
mcldnn_diffattention_strong.py
==============================

Higher-capacity MCLDNN with Differential Self-Attention for the 11-class
RadioML experiment.

This is the differential-attention counterpart of ``mcldnn_attention_strong``.
It keeps the same 3-branch MCLDNN CNN front-end and the same strong sequence
scaffolding, but replaces normal MHA blocks with differential-attention blocks.

Main additions over the old lightweight diff-attention model:

1. 100 -> 128 sequence projection
2. sinusoidal positional encoding in 128-D space
3. two residual differential-attention blocks
4. transformer-style FFN inside each block
5. gated temporal pooling + mean/max pooling
6. larger regularized classifier head
"""

from __future__ import annotations

import math
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
    Layer,
    LayerNormalization,
    Reshape,
    Softmax,
    concatenate,
)
from keras.models import Model


def _lambda_init_fn(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(Layer):
    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.scale = self.add_weight(
            name="scale",
            shape=(last_dim,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        rms = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon})
        return cfg


class DifferentialAttention(Layer):
    """Gated multi-head differential self-attention."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
        cancel_gate_init: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.depth = int(depth)
        self.dropout_rate = float(dropout)
        self.cancel_gate_init = float(cancel_gate_init)
        self.lambda_init = float(_lambda_init_fn(self.depth))

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        h = self.num_heads
        d = self.head_dim

        self.q_proj = Dense(h * 2 * d, use_bias=False, name="q_proj")
        self.k_proj = Dense(h * 2 * d, use_bias=False, name="k_proj")
        self.v_proj = Dense(h * 2 * d, use_bias=False, name="v_proj")
        self.out_proj = Dense(d_model, use_bias=False, name="out_proj")

        lam_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
        self.lambda_q1 = self.add_weight(
            name="lambda_q1", shape=(d,), initializer=lam_init, trainable=True
        )
        self.lambda_k1 = self.add_weight(
            name="lambda_k1", shape=(d,), initializer=lam_init, trainable=True
        )
        self.lambda_q2 = self.add_weight(
            name="lambda_q2", shape=(d,), initializer=lam_init, trainable=True
        )
        self.lambda_k2 = self.add_weight(
            name="lambda_k2", shape=(d,), initializer=lam_init, trainable=True
        )

        gate_init = np.log(self.cancel_gate_init / (1.0 - self.cancel_gate_init))
        self.cancel_gate_logit = self.add_weight(
            name="cancel_gate_logit",
            shape=(),
            initializer=keras.initializers.Constant(gate_init),
            trainable=True,
        )

        self.sub_norm = RMSNorm(epsilon=1e-5, name="sub_rmsnorm")
        self.attn_dropout = Dropout(self.dropout_rate, name="diff_attn_dropout")
        super().build(input_shape)

    def call(self, x, training=None, return_attention_scores: bool = False):
        shape = ops.shape(x)
        batch = shape[0]
        seq_len = shape[1]
        h = self.num_heads
        d = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.reshape(q, (batch, seq_len, h, 2, d))
        k = ops.reshape(k, (batch, seq_len, h, 2, d))
        v = ops.reshape(v, (batch, seq_len, h, 2 * d))

        q = ops.transpose(q, (0, 2, 3, 1, 4))
        k = ops.transpose(k, (0, 2, 3, 1, 4))
        v = ops.transpose(v, (0, 2, 1, 3))

        q1 = q[:, :, 0, :, :]
        q2 = q[:, :, 1, :, :]
        k1 = k[:, :, 0, :, :]
        k2 = k[:, :, 1, :, :]

        scale = 1.0 / math.sqrt(float(d))
        scores1 = ops.matmul(q1, ops.transpose(k1, (0, 1, 3, 2))) * scale
        scores2 = ops.matmul(q2, ops.transpose(k2, (0, 1, 3, 2))) * scale

        attn1 = ops.softmax(scores1, axis=-1)
        attn2 = ops.softmax(scores2, axis=-1)

        lambda_val = (
            ops.exp(ops.sum(self.lambda_q1 * self.lambda_k1))
            - ops.exp(ops.sum(self.lambda_q2 * self.lambda_k2))
            + self.lambda_init
        )
        cancel_gate = ops.sigmoid(self.cancel_gate_logit)
        lambda_eff = cancel_gate * lambda_val

        diff_attn = attn1 - lambda_eff * attn2
        diff_attn = self.attn_dropout(diff_attn, training=training)

        out = ops.matmul(diff_attn, v)
        out = self.sub_norm(out)
        out = out * (1.0 - self.cancel_gate_init * self.lambda_init)

        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch, seq_len, h * 2 * d))
        out = self.out_proj(out)

        if return_attention_scores:
            return out, diff_attn
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "depth": self.depth,
                "dropout": self.dropout_rate,
                "cancel_gate_init": self.cancel_gate_init,
            }
        )
        return cfg


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _diff_transformer_block(
    x,
    *,
    block_id: int,
    d_model: int,
    num_heads: int,
    head_dim: int,
    ffn_dim: int,
    attn_dropout: float,
    l2_weight: float,
    cancel_gate_init: float,
    return_scores: bool = False,
):
    reg = keras.regularizers.L2(l2_weight) if l2_weight > 0 else None

    x_norm = LayerNormalization(name=f"block{block_id}_diff_attn_prenorm")(x)
    diff_attn = DifferentialAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        depth=block_id,
        dropout=attn_dropout,
        cancel_gate_init=cancel_gate_init,
        name=f"block{block_id}_diff_attention",
    )
    attn_out, attn_scores = diff_attn(
        x_norm,
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
    head_dim: int = 16,
    ffn_dim: int = 256,
    n_blocks: int = 2,
    attn_dropout: float = 0.15,
    cancel_gate_init: float = 0.05,
    l2_conv: float = 1e-4,
    l2_transformer: float = 1e-4,
    l2_dense: float = 1e-3,
):
    reg_conv = keras.regularizers.L2(l2_conv) if l2_conv > 0 else None
    reg_dense = keras.regularizers.L2(l2_dense) if l2_dense > 0 else None

    input1 = Input(shape=(2, 128, 1), name="input1")
    input2 = Input(shape=(128, 1), name="input2")
    input3 = Input(shape=(128, 1), name="input3")

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
        x, scores = _diff_transformer_block(
            x,
            block_id=block_id,
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_dim=ffn_dim,
            attn_dropout=attn_dropout,
            l2_weight=l2_transformer,
            cancel_gate_init=cancel_gate_init,
            return_scores=(block_id == n_blocks),
        )
        if scores is not None:
            last_scores = scores

    x = LayerNormalization(name="final_sequence_norm")(x)

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


def build_mcldnn_diffattention_strong(
    classes: int = 11,
    dropout_rate: float = 0.45,
    learning_rate: float = 5e-4,
) -> Model:
    """Build and compile the stronger differential-attention training model."""
    inputs, softmax_out, _ = _build_graph(classes=classes, dropout_rate=dropout_rate)
    model = Model(
        inputs=inputs,
        outputs=softmax_out,
        name="MCLDNN_DiffAttention_Strong",
    )
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    )
    return model


def build_mcldnn_diffattention_strong_extractor(
    classes: int = 11,
    dropout_rate: float = 0.45,
    weights_path: str | None = None,
) -> Model:
    """Build extractor returning [softmax, final differential-attention scores]."""
    inputs, softmax_out, attn_scores = _build_graph(
        classes=classes,
        dropout_rate=dropout_rate,
    )
    model = Model(
        inputs=inputs,
        outputs=[softmax_out, attn_scores],
        name="MCLDNN_DiffAttention_Strong_Extractor",
    )
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    m = build_mcldnn_diffattention_strong(classes=11)
    m.summary()
    print(f"Params: {m.count_params():,}")
