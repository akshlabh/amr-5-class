"""
mcldnn_attention_amp_lite.py — MCLDNN-Attention with lite amplitude/PAR features
============================================================================

This experiment removes phase features completely and focuses on the current
research goal: reducing QAM16 ↔ QAM64 confusion.

Main path:
    IQ / I / Q -> CNN fusion -> self-attention -> stats pooling

Amplitude branch:
    [A(t), A(t)^2, delta_A(t), abs(delta_A(t))] -> small Conv1D branch

Fusion:
    final_context = main_context + gate * amplitude_context

The late gate lets the model use amplitude cues where they help QAM separation
without forcing them into the attention tokens.
"""

import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate,
    MultiHeadAttention, LayerNormalization,
    Multiply, Add,
)
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
    """
    Build the amplitude-only attention graph.

    Inputs
    ------
    input1 : (batch, 2, 128, 1)  raw IQ frame
    input2 : (batch, 128, 1)     I channel
    input3 : (batch, 128, 1)     Q channel
    input4 : (batch, 128, 7)     lite amplitude + QAM16-boundary peak channels
    input5 : (batch, 7)          PAR / peak-count global amplitude statistics
    """
    dr = dropout_rate
    l2_cnn = keras.regularizers.L2(1e-4)
    l2_dense = keras.regularizers.L2(3e-3)

    input1 = Input(shape=(2, 128, 1), name='input1')
    input2 = Input(shape=(128, 1), name='input2')
    input3 = Input(shape=(128, 1), name='input3')
    input4 = Input(shape=(128, 7), name='input4_amplitude_peak_lite_sequence')
    input5 = Input(shape=(7,), name='input5_amplitude_peak_global')

    # ------------------------------------------------------------------
    # Original MCLDNN CNN block, same topology and layer names as attention.
    # ------------------------------------------------------------------
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1', kernel_regularizer=l2_cnn)(input1)

    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2', kernel_regularizer=l2_cnn)(input2)
    x2 = Reshape((1, 128, 50), name='reshape_x2')(x2)

    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3', kernel_regularizer=l2_cnn)(input3)
    x3 = Reshape((1, 128, 50), name='reshape_x3')(x3)

    x = concatenate([x2, x3], axis=1)
    x = Conv2D(50, (1, 8), padding='same', activation='relu',
               name='conv2', kernel_regularizer=l2_cnn)(x)

    x = concatenate([x1, x], axis=-1)
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv4', kernel_regularizer=l2_cnn)(x)
    x = Reshape((124, 100), name='reshape_final')(x)

    # ------------------------------------------------------------------
    # Attention block on learned IQ representation.
    # ------------------------------------------------------------------
    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100),
        dtype='float32',
    )
    x = x + pe

    mha = MultiHeadAttention(num_heads=4, key_dim=25, dropout=attn_dropout,
                             name='mha')
    attn_out, attn_weights = mha(
        query=x, value=x, key=x,
        return_attention_scores=True,
    )

    x = LayerNormalization(name='attn_norm')(attn_out + x)
    x = Dropout(attn_dropout, name='attn_drop')(x)

    ffn = Dense(256, activation='relu', name='ffn1',
                kernel_regularizer=l2_cnn)(x)
    ffn = Dropout(attn_dropout, name='ffn_drop')(ffn)
    ffn = Dense(100, name='ffn2',
                kernel_regularizer=l2_cnn)(ffn)
    x = LayerNormalization(name='ffn_norm')(ffn + x)

    x_mean = ops.mean(x, axis=1)
    x_max = ops.max(x, axis=1)
    main_context = x_mean + x_max                         # (batch, 100)

    # ------------------------------------------------------------------
    # Amplitude-only physical branch.
    # ------------------------------------------------------------------
    amp_seq = Conv1D(32, 5, padding='valid', activation='relu',
                     name='amp_peak_conv1',
                     kernel_regularizer=l2_cnn)(input4)   # (batch, 124, 32)
    amp_seq = Conv1D(32, 3, padding='same', activation='relu',
                     name='amp_peak_conv2',
                     kernel_regularizer=l2_cnn)(amp_seq)  # (batch, 124, 32)
    amp_seq = Dropout(attn_dropout, name='amp_peak_drop')(amp_seq)

    amp_mean = ops.mean(amp_seq, axis=1)
    amp_max = ops.max(amp_seq, axis=1)
    amp_seq_context = amp_mean + amp_max                   # (batch, 32)

    amp_global = Dense(32, activation='relu',
                       name='amp_global_dense1',
                       kernel_regularizer=l2_cnn)(input5)
    amp_global = Dropout(attn_dropout, name='amp_global_drop')(amp_global)
    amp_global = Dense(32, activation='relu',
                       name='amp_global_dense2',
                       kernel_regularizer=l2_cnn)(amp_global)

    amp_context = concatenate([amp_seq_context, amp_global],
                              axis=-1,
                              name='amp_peak_context_concat')
    amp_context = Dense(100, activation='relu',
                        name='amp_peak_context_proj',
                        kernel_regularizer=l2_cnn)(amp_context)

    # Gated late fusion. Starts conservative and learns to open the amplitude
    # branch where it reduces QAM16/QAM64 confusion.
    gate_in = concatenate([main_context, amp_context], axis=-1,
                          name='amp_gate_input')
    gate = Dense(100, activation='sigmoid',
                 bias_initializer=keras.initializers.Constant(-1.0),
                 name='amp_gate',
                 kernel_regularizer=l2_cnn)(gate_in)
    gated_amp = Multiply(name='amp_gate_apply')([gate, amp_context])
    context = Add(name='amp_gated_fusion')([main_context, gated_amp])

    out = Dense(128, activation='selu', name='fc1',
                kernel_regularizer=l2_dense)(context)
    out = Dropout(dr, name='drop1')(out)
    out = Dense(128, activation='selu', name='fc2',
                kernel_regularizer=l2_dense)(out)
    out = Dropout(dr, name='drop2')(out)
    softmax_out = Dense(classes, activation='softmax', name='softmax')(out)

    return [input1, input2, input3, input4, input5], softmax_out, attn_weights


def build_mcldnn_attention_amp_lite(classes: int = 5,
                                    dropout_rate: float = 0.6,
                                    learning_rate: float = 1e-3) -> Model:
    """Build and compile the standard-loss lite amplitude/PAR attention model."""
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out,
                  name='MCLDNN_Attention_Amp_Lite')

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                        clipnorm=1.0),
    )
    return model


def build_mcldnn_attention_amp_lite_extractor(classes: int = 5,
                                              dropout_rate: float = 0.6,
                                              weights_path: str = None) -> Model:
    """Build a dual-output extractor [softmax, attn_weights] for analysis."""
    import os

    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(inputs=inputs,
                      outputs=[softmax_out, attn_weights],
                      name='MCLDNN_Attention_Amp_Lite_Extractor')

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        extractor.load_weights(weights_path)
        print(f"[amp lite extractor] Loaded weights from {weights_path}")

    return extractor


if __name__ == '__main__':
    keras.mixed_precision.set_global_policy('float32')

    model = build_mcldnn_attention_amp_lite(classes=5)
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params; exceeds 300,000."
    )

    dummy1 = np.zeros((4, 2, 128, 1), dtype='float32')
    dummy2 = np.zeros((4, 128, 1), dtype='float32')
    dummy3 = np.zeros((4, 128, 1), dtype='float32')
    dummy4 = np.zeros((4, 128, 7), dtype='float32')
    dummy5 = np.zeros((4, 7), dtype='float32')

    y = model.predict([dummy1, dummy2, dummy3, dummy4, dummy5], verbose=0)
    print(f"Training model output shape: {y.shape}")
    assert y.shape == (4, 5)

    extractor = build_mcldnn_attention_amp_lite_extractor(classes=5)
    y2, attn = extractor.predict([dummy1, dummy2, dummy3, dummy4, dummy5],
                                 verbose=0)
    print(f"Extractor softmax shape: {y2.shape}")
    print(f"Extractor attention shape: {attn.shape}")
    assert y2.shape == (4, 5)
    assert attn.shape == (4, 4, 124, 124)
    print("All shape checks PASSED")
