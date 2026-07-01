"""
mcldnn_diffattention.py - MCLDNN with Differential Self-Attention
=================================================================

This is a clean sibling of ``mcldnn_attention.py``.

What stays the same:
  - same three inputs as the normal MCLDNN/attention model
  - same convolutional front-end layer names and shapes
  - same sinusoidal positional encoding
  - same residual/norm/FFN/pooling/classifier head
  - same single-output training model plus dual-output extractor pattern

What changes:
  - the standard Keras MultiHeadAttention layer is replaced by a Keras 3
    implementation of Differential Attention.

Differential attention idea
---------------------------
Normal self-attention computes:

    softmax(Q K^T / sqrt(d)) V

Differential attention computes two independent attention maps and subtracts
one from the other:

    (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V

The subtraction is intended to cancel common/generic attention noise and leave
a sharper signed attention map.  The returned extractor attention tensor is
therefore signed; it is not a probability distribution like ordinary softmax
attention.
"""

from __future__ import annotations

import math

import keras
from keras.layers import (
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Input,
    Layer,
    LayerNormalization,
    Reshape,
    concatenate,
)
from keras.models import Model
import keras.ops as ops
import numpy as np


def _lambda_init_fn(depth: int) -> float:
    """Depth-dependent lambda initialization from Differential Transformer."""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(Layer):
    """Small RMSNorm layer using Keras ops.

    Unlike LayerNormalization, this does not subtract the mean.  It normalizes
    by root-mean-square magnitude along the last dimension only.
    """

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
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class DifferentialAttention(Layer):
    """Multi-head differential self-attention.

    Parameters
    ----------
    num_heads:
        Number of differential heads.  Each differential head internally has
        two Q/K branches.
    head_dim:
        Dimension of each Q1/Q2/K1/K2 branch.
    depth:
        1-indexed layer depth used for lambda initialization.
    dropout:
        Dropout applied to the signed differential attention map.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.depth = int(depth)
        self.dropout_rate = float(dropout)
        self.lambda_init = float(_lambda_init_fn(self.depth))

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        h = self.num_heads
        d = self.head_dim

        # Q and K contain two independent projections per differential head.
        self.q_proj = Dense(h * 2 * d, use_bias=False, name="q_proj")
        self.k_proj = Dense(h * 2 * d, use_bias=False, name="k_proj")

        # V is 2*d per differential head, matching the reference structure.
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

        q = ops.transpose(q, (0, 2, 3, 1, 4))  # (batch, heads, 2, seq, dim)
        k = ops.transpose(k, (0, 2, 3, 1, 4))
        v = ops.transpose(v, (0, 2, 1, 3))     # (batch, heads, seq, 2*dim)

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

        diff_attn = attn1 - lambda_val * attn2
        diff_attn = self.attn_dropout(diff_attn, training=training)

        out = ops.matmul(diff_attn, v)          # (batch, heads, seq, 2*dim)
        out = self.sub_norm(out)
        out = out * (1.0 - self.lambda_init)

        out = ops.transpose(out, (0, 2, 1, 3))  # (batch, seq, heads, 2*dim)
        out = ops.reshape(out, (batch, seq_len, h * 2 * d))
        out = self.out_proj(out)

        if return_attention_scores:
            return out, diff_attn
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "depth": self.depth,
                "dropout": self.dropout_rate,
            }
        )
        return config


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding, matching mcldnn_attention.py."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _build_graph(classes: int, dropout_rate: float, attn_dropout: float = 0.1):
    """Build graph shared by training model and extractor model."""
    dr = dropout_rate

    input1 = Input(shape=(2, 128, 1), name="input1")
    input2 = Input(shape=(128, 1), name="input2")
    input3 = Input(shape=(128, 1), name="input3")

    # Convolutional front-end: intentionally identical to mcldnn_attention.py.
    x1 = Conv2D(
        50,
        (2, 8),
        padding="same",
        activation="relu",
        name="conv1_1",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(input1)

    x2 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        name="conv1_2",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(input2)
    x2 = Reshape((1, 128, 50), name="reshape_x2")(x2)

    x3 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        name="conv1_3",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(input3)
    x3 = Reshape((1, 128, 50), name="reshape_x3")(x3)

    x = concatenate([x2, x3], axis=1)
    x = Conv2D(
        50,
        (1, 8),
        padding="same",
        activation="relu",
        name="conv2",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(x)

    x = concatenate([x1, x], axis=-1)
    x = Conv2D(
        100,
        (2, 5),
        padding="valid",
        activation="relu",
        name="conv4",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(x)

    x = Reshape((124, 100), name="reshape_final")(x)

    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100), dtype="float32"
    )
    x = x + pe

    # 2 differential heads * 2 branches * 25 dims = 100 Q/K/V width.
    # This keeps the attention projection size close to the normal 4-head
    # attention model while using the differential-attention structure.
    diff_mha = DifferentialAttention(
        num_heads=2,
        head_dim=25,
        depth=1,
        dropout=attn_dropout,
        name="diff_mha",
    )
    attn_out, attn_weights = diff_mha(x, return_attention_scores=True)

    x = LayerNormalization(name="attn_norm")(attn_out + x)
    x = Dropout(attn_dropout, name="attn_drop")(x)

    ffn = Dense(
        256,
        activation="relu",
        name="ffn1",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(x)
    ffn = Dropout(attn_dropout, name="ffn_drop")(ffn)
    ffn = Dense(
        100,
        name="ffn2",
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(ffn)
    x = LayerNormalization(name="ffn_norm")(ffn + x)

    x_mean = ops.mean(x, axis=1)
    x_max = ops.max(x, axis=1)
    context = x_mean + x_max

    out = Dense(
        128,
        activation="selu",
        name="fc1",
        kernel_regularizer=keras.regularizers.L2(3e-3),
    )(context)
    out = Dropout(dr, name="drop1")(out)
    out = Dense(
        128,
        activation="selu",
        name="fc2",
        kernel_regularizer=keras.regularizers.L2(3e-3),
    )(out)
    out = Dropout(dr, name="drop2")(out)
    softmax_out = Dense(classes, activation="softmax", name="softmax")(out)

    return [input1, input2, input3], softmax_out, attn_weights


def build_mcldnn_diffattention(
    classes: int = 5,
    dropout_rate: float = 0.6,
    learning_rate: float = 1e-3,
) -> Model:
    """Build and compile the single-output training model."""
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out, name="MCLDNN_DiffAttention")
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    )
    return model


def build_mcldnn_diffattention_extractor(
    classes: int = 5,
    dropout_rate: float = 0.6,
    weights_path: str | None = None,
) -> Model:
    """Build dual-output extractor: [softmax, signed differential attention]."""
    import os

    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(
        inputs=inputs,
        outputs=[softmax_out, attn_weights],
        name="MCLDNN_DiffAttention_Extractor",
    )
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                "Train the model first with build_mcldnn_diffattention()."
            )
        extractor.load_weights(weights_path)
        print(f"[extractor] Loaded weights from {weights_path}")
    return extractor


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")

    model = build_mcldnn_diffattention(classes=5)
    model.summary()

    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000

    dummy1 = np.zeros((4, 2, 128, 1), dtype="float32")
    dummy2 = np.zeros((4, 128, 1), dtype="float32")
    dummy3 = np.zeros((4, 128, 1), dtype="float32")

    pred = model.predict([dummy1, dummy2, dummy3], verbose=0)
    print(f"Training output shape: {pred.shape}")
    assert pred.shape == (4, 5)

    extractor = build_mcldnn_diffattention_extractor(classes=5)
    pred2, attn = extractor.predict([dummy1, dummy2, dummy3], verbose=0)
    print(f"Extractor softmax shape: {pred2.shape}")
    print(f"Extractor attention shape: {attn.shape}")
    assert pred2.shape == (4, 5)
    assert attn.shape == (4, 2, 124, 124)
