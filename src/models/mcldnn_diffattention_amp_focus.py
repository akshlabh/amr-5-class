"""
mcldnn_diffattention_amp_focus.py — Differential attention + focused QAM amplitude cues
=====================================================================================

Research target
---------------
Reduce QAM16 <-> QAM64 confusion while keeping the differential-attention
experiment physically grounded.

Main path:
    IQ / I / Q -> MCLDNN CNN fusion -> gated differential attention -> pooling

Focused physical branch:
    sequence input : [A(t), A(t)^2, boundary_crossing(t), excess_amplitude(t)]
    global input   : [log1p(PAR), amplitude_p95]

Fusion:
    final_context = diff_attention_context + gate * amplitude_context

No phase features are used.  No delta/time-transition amplitude features are
used, because low-SNR sample-to-sample transitions can be dominated by noise.
"""

from __future__ import annotations

import keras
from keras.layers import (
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    Multiply,
    Add,
    Reshape,
    concatenate,
)
from keras.models import Model
import keras.ops as ops
import numpy as np

from src.models.mcldnn_diffattention import DifferentialAttention


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding, matching the attention models."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _build_graph(classes: int, dropout_rate: float, attn_dropout: float = 0.1):
    """Build graph shared by training model and extractor model."""
    dr = dropout_rate
    l2_cnn = keras.regularizers.L2(1e-4)
    l2_dense = keras.regularizers.L2(3e-3)

    input1 = Input(shape=(2, 128, 1), name="input1")
    input2 = Input(shape=(128, 1), name="input2")
    input3 = Input(shape=(128, 1), name="input3")
    input4 = Input(shape=(128, 4), name="input4_amplitude_focus_sequence")
    input5 = Input(shape=(2,), name="input5_par_p95_global")

    # ------------------------------------------------------------------
    # MCLDNN CNN front-end, same topology as the existing attention models.
    # ------------------------------------------------------------------
    x1 = Conv2D(
        50,
        (2, 8),
        padding="same",
        activation="relu",
        name="conv1_1",
        kernel_regularizer=l2_cnn,
    )(input1)

    x2 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        name="conv1_2",
        kernel_regularizer=l2_cnn,
    )(input2)
    x2 = Reshape((1, 128, 50), name="reshape_x2")(x2)

    x3 = Conv1D(
        50,
        8,
        padding="causal",
        activation="relu",
        name="conv1_3",
        kernel_regularizer=l2_cnn,
    )(input3)
    x3 = Reshape((1, 128, 50), name="reshape_x3")(x3)

    x = concatenate([x2, x3], axis=1)
    x = Conv2D(
        50,
        (1, 8),
        padding="same",
        activation="relu",
        name="conv2",
        kernel_regularizer=l2_cnn,
    )(x)

    x = concatenate([x1, x], axis=-1)
    x = Conv2D(
        100,
        (2, 5),
        padding="valid",
        activation="relu",
        name="conv4",
        kernel_regularizer=l2_cnn,
    )(x)

    x = Reshape((124, 100), name="reshape_final")(x)
    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100),
        dtype="float32",
    )
    x = x + pe

    # ------------------------------------------------------------------
    # Gated differential self-attention.
    # ------------------------------------------------------------------
    diff_mha = DifferentialAttention(
        num_heads=2,
        head_dim=25,
        depth=1,
        dropout=attn_dropout,
        cancel_gate_init=0.05,
        name="diff_mha",
    )
    attn_out, attn_weights = diff_mha(x, return_attention_scores=True)

    x = LayerNormalization(name="attn_norm")(attn_out + x)
    x = Dropout(attn_dropout, name="attn_drop")(x)

    ffn = Dense(256, activation="relu", name="ffn1",
                kernel_regularizer=l2_cnn)(x)
    ffn = Dropout(attn_dropout, name="ffn_drop")(ffn)
    ffn = Dense(100, name="ffn2", kernel_regularizer=l2_cnn)(ffn)
    x = LayerNormalization(name="ffn_norm")(ffn + x)

    x_mean = ops.mean(x, axis=1)
    x_max = ops.max(x, axis=1)
    main_context = x_mean + x_max

    # ------------------------------------------------------------------
    # Focused QAM amplitude/PAR branch.
    # ------------------------------------------------------------------
    amp_seq = Conv1D(24, 5, padding="valid", activation="relu",
                     name="amp_focus_conv1",
                     kernel_regularizer=l2_cnn)(input4)   # (batch, 124, 24)
    amp_seq = Conv1D(24, 3, padding="same", activation="relu",
                     name="amp_focus_conv2",
                     kernel_regularizer=l2_cnn)(amp_seq)
    amp_seq = Dropout(attn_dropout, name="amp_focus_drop")(amp_seq)

    amp_mean = ops.mean(amp_seq, axis=1)
    amp_max = ops.max(amp_seq, axis=1)
    amp_seq_context = amp_mean + amp_max

    amp_global = Dense(16, activation="relu",
                       name="amp_global_dense1",
                       kernel_regularizer=l2_cnn)(input5)
    amp_global = Dropout(attn_dropout, name="amp_global_drop")(amp_global)
    amp_global = Dense(16, activation="relu",
                       name="amp_global_dense2",
                       kernel_regularizer=l2_cnn)(amp_global)

    amp_context = concatenate([amp_seq_context, amp_global],
                              axis=-1,
                              name="amp_focus_context_concat")
    amp_context = Dense(100, activation="relu",
                        name="amp_focus_context_proj",
                        kernel_regularizer=l2_cnn)(amp_context)

    # Conservative gate: starts mostly closed so the diff-attention backbone is
    # not overpowered early, then learns when amplitude/PAR evidence helps QAM.
    gate_in = concatenate([main_context, amp_context], axis=-1,
                          name="amp_gate_input")
    gate = Dense(100, activation="sigmoid",
                 bias_initializer=keras.initializers.Constant(-1.0),
                 name="amp_gate",
                 kernel_regularizer=l2_cnn)(gate_in)
    gated_amp = Multiply(name="amp_gate_apply")([gate, amp_context])
    context = Add(name="amp_gated_fusion")([main_context, gated_amp])

    out = Dense(128, activation="selu", name="fc1",
                kernel_regularizer=l2_dense)(context)
    out = Dropout(dr, name="drop1")(out)
    out = Dense(128, activation="selu", name="fc2",
                kernel_regularizer=l2_dense)(out)
    out = Dropout(dr, name="drop2")(out)
    softmax_out = Dense(classes, activation="softmax", name="softmax")(out)

    return [input1, input2, input3, input4, input5], softmax_out, attn_weights


def build_mcldnn_diffattention_amp_focus(
    classes: int = 5,
    dropout_rate: float = 0.55,
    learning_rate: float = 5e-4,
    label_smoothing: float = 0.0,
) -> Model:
    """Build and compile the single-output training model."""
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(
        inputs=inputs,
        outputs=softmax_out,
        name="MCLDNN_DiffAttention_Amp_Focus",
    )
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    )
    return model


def build_mcldnn_diffattention_amp_focus_extractor(
    classes: int = 5,
    dropout_rate: float = 0.55,
    weights_path: str | None = None,
) -> Model:
    """Build dual-output extractor: [softmax, signed differential attention]."""
    import os

    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(
        inputs=inputs,
        outputs=[softmax_out, attn_weights],
        name="MCLDNN_DiffAttention_Amp_Focus_Extractor",
    )
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        extractor.load_weights(weights_path)
        print(f"[diff amp focus extractor] Loaded weights from {weights_path}")
    return extractor


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")

    model = build_mcldnn_diffattention_amp_focus(classes=5)
    model.summary()

    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params; exceeds 300,000."
    )

    dummy1 = np.zeros((4, 2, 128, 1), dtype="float32")
    dummy2 = np.zeros((4, 128, 1), dtype="float32")
    dummy3 = np.zeros((4, 128, 1), dtype="float32")
    dummy4 = np.zeros((4, 128, 4), dtype="float32")
    dummy5 = np.zeros((4, 2), dtype="float32")

    pred = model.predict([dummy1, dummy2, dummy3, dummy4, dummy5], verbose=0)
    print(f"Training output shape: {pred.shape}")
    assert pred.shape == (4, 5)

    extractor = build_mcldnn_diffattention_amp_focus_extractor(classes=5)
    pred2, attn = extractor.predict(
        [dummy1, dummy2, dummy3, dummy4, dummy5],
        verbose=0,
    )
    print(f"Extractor softmax shape: {pred2.shape}")
    print(f"Extractor attention shape: {attn.shape}")
    assert pred2.shape == (4, 5)
    assert attn.shape == (4, 2, 124, 124)
    print("All shape checks PASSED")
