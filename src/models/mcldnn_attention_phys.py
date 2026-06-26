"""
mcldnn_attention_phys.py — MCLDNN-Attention with physical IQ features
=====================================================================

This model keeps the existing MCLDNN-Attention CNN backbone intact and adds
two deterministic feature inputs before self-attention:

  • amplitude A(t), useful for QAM16 vs QAM64
  • phase unit-vector [cos(phi), sin(phi)], useful for QPSK vs 8PSK

The extra features are fused immediately before MultiHeadAttention so each
attention token contains both learned CNN features and explicit physical
signal coordinates.
"""

import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate,
    MultiHeadAttention, LayerNormalization,
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
    Build the physical-feature attention graph.

    Inputs
    ------
    input1 : (batch, 2, 128, 1)  raw IQ frame
    input2 : (batch, 128, 1)     I channel
    input3 : (batch, 128, 1)     Q channel
    input4 : (batch, 128, 1)     normalized amplitude
    input5 : (batch, 128, 2)     phase [cos(phi), sin(phi)]

    Returns
    -------
    inputs, softmax_out, attn_weights
    """
    dr = dropout_rate
    l2_cnn = keras.regularizers.L2(1e-4)

    # Original three MCLDNN inputs
    input1 = Input(shape=(2, 128, 1), name='input1')
    input2 = Input(shape=(128, 1), name='input2')
    input3 = Input(shape=(128, 1), name='input3')

    # New physical feature inputs
    input4 = Input(shape=(128, 1), name='input4_amplitude')
    input5 = Input(shape=(128, 2), name='input5_phase_sincos')

    # ------------------------------------------------------------------
    # Original MCLDNN CNN block, same topology and names as attention model
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
    # Physical feature branches, aligned to 124 time steps using valid Conv1D
    # ------------------------------------------------------------------
    amp_feat = Conv1D(16, 5, padding='valid', activation='relu',
                      name='amp_conv',
                      kernel_regularizer=l2_cnn)(input4)

    phase_feat = Conv1D(16, 5, padding='valid', activation='relu',
                        name='phase_conv',
                        kernel_regularizer=l2_cnn)(input5)

    # Fuse learned CNN features with explicit physical features, then project
    # back to d_model=100 so the attention block stays unchanged.
    x = concatenate([x, amp_feat, phase_feat], axis=-1,
                    name='phys_feature_concat')
    x = Dense(100, activation='relu', name='phys_feature_proj',
              kernel_regularizer=l2_cnn)(x)

    # ------------------------------------------------------------------
    # Attention block
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

    # Same stats pooling and classifier head as current attention model.
    x_mean = ops.mean(x, axis=1)
    x_max = ops.max(x, axis=1)
    context = x_mean + x_max

    out = Dense(128, activation='selu', name='fc1',
                kernel_regularizer=keras.regularizers.L2(3e-3))(context)
    out = Dropout(dr, name='drop1')(out)
    out = Dense(128, activation='selu', name='fc2',
                kernel_regularizer=keras.regularizers.L2(3e-3))(out)
    out = Dropout(dr, name='drop2')(out)
    softmax_out = Dense(classes, activation='softmax', name='softmax')(out)

    return [input1, input2, input3, input4, input5], softmax_out, attn_weights


def build_mcldnn_attention_phys(classes: int = 5,
                                dropout_rate: float = 0.6,
                                learning_rate: float = 1e-3) -> Model:
    """
    Build and compile the standard-loss physical-feature attention model.
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
    dummy4 = np.zeros((4, 128, 1), dtype='float32')
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

