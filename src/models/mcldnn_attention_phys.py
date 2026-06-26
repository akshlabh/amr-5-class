"""
mcldnn_attention_phys.py — MCLDNN-Attention with gated physical features
========================================================================

Version 2 of the physical-feature attention model.

The original MCLDNN-Attention path is kept intact:

    IQ / I / Q -> CNN fusion -> self-attention -> stats pooling

The physical features are processed in a separate small branch:

    amplitude features      : [A(t), A(t)^2, ΔA(t), |ΔA(t)|]
    differential phase      : [cos(dphi), sin(dphi)]

Then the physical branch is fused *after attention pooling* using a learned
gate:

    final_context = main_context + gate * physics_context

This gives the model an escape hatch: when physical features are noisy or not
helpful, the gate can suppress them instead of forcing them into the attention
tokens.
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
    Build the gated physical-feature attention graph.

    Inputs
    ------
    input1 : (batch, 2, 128, 1)  raw IQ frame
    input2 : (batch, 128, 1)     I channel
    input3 : (batch, 128, 1)     Q channel
    input4 : (batch, 128, 4)     [A, A², ΔA, |ΔA|]
    input5 : (batch, 128, 2)     [cos(dphi), sin(dphi)]

    Returns
    -------
    inputs, softmax_out, attn_weights
    """
    dr = dropout_rate
    l2_cnn = keras.regularizers.L2(1e-4)
    l2_dense = keras.regularizers.L2(3e-3)

    # Original three MCLDNN inputs.
    input1 = Input(shape=(2, 128, 1), name='input1')
    input2 = Input(shape=(128, 1), name='input2')
    input3 = Input(shape=(128, 1), name='input3')

    # Physical feature inputs.
    input4 = Input(shape=(128, 4), name='input4_amplitude_features')
    input5 = Input(shape=(128, 2), name='input5_dphase_sincos')

    # ------------------------------------------------------------------
    # Original MCLDNN CNN block, same topology and names as attention model.
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
    # Attention block on the learned IQ representation only.
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
    # Separate physical branch.
    # Conv1D(valid, kernel=5) aligns 128 raw feature steps to the same 124
    # sequence length produced by conv4 in the main CNN path.
    # ------------------------------------------------------------------
    amp_seq = Conv1D(24, 5, padding='valid', activation='relu',
                     name='amp_feat_conv',
                     kernel_regularizer=l2_cnn)(input4)   # (batch, 124, 24)

    phase_seq = Conv1D(24, 5, padding='valid', activation='relu',
                       name='dphase_feat_conv',
                       kernel_regularizer=l2_cnn)(input5) # (batch, 124, 24)

    phys_seq = concatenate([amp_seq, phase_seq], axis=-1,
                           name='phys_seq_concat')        # (batch, 124, 48)
    phys_seq = Dense(64, activation='relu', name='phys_seq_proj',
                     kernel_regularizer=l2_cnn)(phys_seq) # (batch, 124, 64)
    phys_seq = Dropout(attn_dropout, name='phys_seq_drop')(phys_seq)

    phys_mean = ops.mean(phys_seq, axis=1)
    phys_max = ops.max(phys_seq, axis=1)
    phys_context = phys_mean + phys_max                    # (batch, 64)
    phys_context = Dense(100, activation='relu',
                         name='phys_context_proj',
                         kernel_regularizer=l2_cnn)(phys_context)

    # ------------------------------------------------------------------
    # Gated late fusion.
    # The negative bias initializes the sigmoid gate below 0.5, so early
    # training starts closer to the proven IQ-attention path and learns to open
    # the physics branch only where it helps.
    # ------------------------------------------------------------------
    gate_in = concatenate([main_context, phys_context], axis=-1,
                          name='phys_gate_input')          # (batch, 200)
    gate = Dense(100, activation='sigmoid',
                 bias_initializer=keras.initializers.Constant(-1.0),
                 name='phys_gate',
                 kernel_regularizer=l2_cnn)(gate_in)
    gated_phys = Multiply(name='phys_gate_apply')([gate, phys_context])
    context = Add(name='phys_gated_fusion')([main_context, gated_phys])

    # ------------------------------------------------------------------
    # Dense classifier head.
    # ------------------------------------------------------------------
    out = Dense(128, activation='selu', name='fc1',
                kernel_regularizer=l2_dense)(context)
    out = Dropout(dr, name='drop1')(out)
    out = Dense(128, activation='selu', name='fc2',
                kernel_regularizer=l2_dense)(out)
    out = Dropout(dr, name='drop2')(out)
    softmax_out = Dense(classes, activation='softmax', name='softmax')(out)

    return [input1, input2, input3, input4, input5], softmax_out, attn_weights


def build_mcldnn_attention_phys(classes: int = 5,
                                dropout_rate: float = 0.6,
                                learning_rate: float = 1e-3) -> Model:
    """
    Build and compile the standard-loss gated physical-feature attention model.
    """
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out,
                  name='MCLDNN_Attention_Phys')

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                        clipnorm=1.0),
    )
    return model


def build_mcldnn_attention_phys_extractor(classes: int = 5,
                                          dropout_rate: float = 0.6,
                                          weights_path: str = None) -> Model:
    """
    Build a dual-output extractor [softmax, attn_weights] for analysis.
    """
    import os

    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(inputs=inputs,
                      outputs=[softmax_out, attn_weights],
                      name='MCLDNN_Attention_Phys_Extractor')

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        extractor.load_weights(weights_path)
        print(f"[phys extractor] Loaded weights from {weights_path}")

    return extractor


if __name__ == '__main__':
    keras.mixed_precision.set_global_policy('float32')

    model = build_mcldnn_attention_phys(classes=5)
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params; exceeds 300,000."
    )

    dummy1 = np.zeros((4, 2, 128, 1), dtype='float32')
    dummy2 = np.zeros((4, 128, 1), dtype='float32')
    dummy3 = np.zeros((4, 128, 1), dtype='float32')
    dummy4 = np.zeros((4, 128, 4), dtype='float32')
    dummy5 = np.zeros((4, 128, 2), dtype='float32')

    y = model.predict([dummy1, dummy2, dummy3, dummy4, dummy5], verbose=0)
    print(f"Training model output shape: {y.shape}")
    assert y.shape == (4, 5)

    extractor = build_mcldnn_attention_phys_extractor(classes=5)
    y2, attn = extractor.predict([dummy1, dummy2, dummy3, dummy4, dummy5],
                                 verbose=0)
    print(f"Extractor softmax shape: {y2.shape}")
    print(f"Extractor attention shape: {attn.shape}")
    assert y2.shape == (4, 5)
    assert attn.shape == (4, 4, 124, 124)
    print("All shape checks PASSED")

