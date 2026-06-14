"""
mcldnn_attention.py — MCLDNN with 2-layer Transformer Encoder replacing LSTM
=============================================================================
Architecture overview
----------------------
Conv block: IDENTICAL to mcldnn.py (same layer names, filter counts, kernel
             sizes, padding modes, activations).  Output shape: (batch, 124, 100).

Transformer Encoder (2 blocks, replaces both LSTM layers):
  A) Fixed sinusoidal positional encoding (Vaswani et al. 2017)
     — no trainable parameters, position-aware from epoch 0.

  B) Encoder Block 1 (pre-LayerNorm style):
       LN -> MHA(4 heads, key_dim=25) -> Add & Dropout
       LN -> FFN(Dense(128,GELU) -> Dense(100)) -> Add & Dropout

  C) Encoder Block 2 (same structure, returns attn_weights for interpretability):
       LN -> MHA(4 heads, key_dim=25) -> Add & Dropout
       LN -> FFN(Dense(128,GELU) -> Dense(100)) -> Add & Dropout

  D) Final LayerNorm (stabilises pooling input)

  E) Learned temperature temporal pooling:
       Dense(1) -> score / t -> Softmax(time axis) -> weighted sum
       t = exp(log_t), trained end-to-end.
       At low SNR:  t->large -> uniform averaging -> noise cancellation
       At high SNR: t->small -> sharp focus -> symbol boundary detection

Why two Transformer encoder layers?
  A single MHA pass computes "who attends to who" once and immediately pools.
  The second layer refines: given the attended representation from layer 1,
  it can form higher-order associations. All AMR papers achieving >70% at 0 dB
  use >=2 Transformer layers (RadioTransformer, AMCNet, CNN-Transformer).

Why pre-norm (LN before MHA) vs post-norm?
  Pre-norm keeps gradient magnitudes stable throughout training, allowing
  the model to train longer without divergence on small datasets.

Why FFN after MHA?
  MHA produces a linearly-mixed attended output. The FFN applies a learned
  non-linear transformation AFTER global context is incorporated.

Parameter budget (~285k, under 300k limit):
  Conv block:    ~122k
  Sinusoidal PE:     0
  Block 1:        ~66k  (MHA 40k + FFN 26k)
  Block 2:        ~66k
  LayerNorms:      ~1k
  Pooling+head:   ~30k
  Total:         ~285k

Two public functions
---------------------
build_mcldnn_attention(classes, dropout_rate)
    Single-output training model (softmax only). Compiled.

build_mcldnn_attention_extractor(classes, dropout_rate, weights_path)
    Dual-output [softmax, attn_weights_block2]. NOT compiled.
"""

import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate,
    MultiHeadAttention, LayerNormalization,
    Softmax,
)
import keras.ops as ops
import numpy as np


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Returns array of shape (1, seq_len, d_model).  No trainable parameters.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pos   = np.arange(seq_len)[:, None]
    i     = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)


def _transformer_encoder_block(x, num_heads, key_dim, ffn_dim,
                                attn_dropout, drop_rate, block_name):
    """
    One pre-norm Transformer Encoder block.

    Structure:
        LayerNorm -> MHA -> Add & Dropout
        LayerNorm -> FFN(GELU) -> Add & Dropout

    Parameters
    ----------
    x            : tensor (batch, seq_len, d_model)
    num_heads    : int   MHA heads
    key_dim      : int   key/query dimension per head
    ffn_dim      : int   inner dimension of the position-wise FFN
    attn_dropout : float MHA attention weight dropout
    drop_rate    : float residual dropout after MHA and FFN
    block_name   : str   prefix for all layer names

    Returns
    -------
    x            : tensor (batch, seq_len, d_model)
    attn_weights : tensor (batch, num_heads, seq_len, seq_len)
    """
    # Sub-layer 1: Multi-head self-attention (pre-norm)
    x_norm = LayerNormalization(name=block_name + '_ln1')(x)
    mha    = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim,
        dropout=attn_dropout, name=block_name + '_mha'
    )
    attn_out, attn_weights = mha(
        query=x_norm, value=x_norm, key=x_norm,
        return_attention_scores=True
    )
    x = x + Dropout(drop_rate, name=block_name + '_drop1')(attn_out)

    # Sub-layer 2: Position-wise FFN (pre-norm)
    # GELU activation: smoother gradient flow than ReLU on small datasets
    x_norm = LayerNormalization(name=block_name + '_ln2')(x)
    ffn    = Dense(ffn_dim, activation='gelu',
                   name=block_name + '_ffn1')(x_norm)
    ffn    = Dense(x.shape[-1], name=block_name + '_ffn2')(ffn)
    x = x + Dropout(drop_rate, name=block_name + '_drop2')(ffn)

    return x, attn_weights


# ── Shared graph builder ───────────────────────────────────────────────────────
def _build_graph(classes: int, dropout_rate: float, attn_dropout: float = 0.1):
    """
    Build the 2-layer Transformer Encoder graph.

    Returns
    -------
    inputs       : list [input1, input2, input3]
    softmax_out  : tensor (batch, classes)
    attn_weights : tensor (batch, 4, 124, 124)  — Block 2 weights
    """
    dr = dropout_rate

    # ── Inputs (identical to mcldnn.py) ───────────────────────────────────────
    input1 = Input(shape=(2, 128, 1), name='input1')
    input2 = Input(shape=(128, 1),    name='input2')
    input3 = Input(shape=(128, 1),    name='input3')

    # ── CONV BLOCK (identical to mcldnn.py) ───────────────────────────────────

    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input1)

    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input2)
    x2 = Reshape((1, 128, 50), name='reshape_x2')(x2)

    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input3)
    x3 = Reshape((1, 128, 50), name='reshape_x3')(x3)

    x  = concatenate([x2, x3], axis=1)
    x  = Conv2D(50, (1, 8), padding='same', activation='relu',
                name='conv2',
                kernel_regularizer=keras.regularizers.L2(1e-4))(x)

    x  = concatenate([x1, x], axis=-1)
    x  = Conv2D(100, (2, 5), padding='valid', activation='relu',
                name='conv4',
                kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x shape: (batch, 1, 124, 100)

    x = Reshape((124, 100), name='reshape_final')(x)

    # ── TRANSFORMER ENCODER ───────────────────────────────────────────────────

    # Step A: Fixed sinusoidal positional encoding
    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100), dtype='float32'
    )
    x = x + pe

    # Step B: Encoder Block 1
    # Learns first-order temporal dependencies:
    # "Which time steps share similar IQ feature patterns?"
    x, _ = _transformer_encoder_block(
        x, num_heads=4, key_dim=25, ffn_dim=128,
        attn_dropout=attn_dropout, drop_rate=attn_dropout,
        block_name='enc1'
    )

    # Step C: Encoder Block 2
    # Refines using second-order dependencies:
    # "Given Block 1 attended features, which modulation class is consistent?"
    # attn_weights from Block 2 are more semantically meaningful for visualisation.
    x, attn_weights = _transformer_encoder_block(
        x, num_heads=4, key_dim=25, ffn_dim=128,
        attn_dropout=attn_dropout, drop_rate=attn_dropout,
        block_name='enc2'
    )

    # Step D: Final LayerNorm before pooling
    x = LayerNormalization(name='final_ln')(x)

    # Step E: Learned temperature temporal pooling
    # At low SNR:  tau->large -> softmax spreads  -> noise cancellation
    # At high SNR: tau->small -> softmax sharpens -> selective focus
    score   = Dense(1, name='temporal_score')(x)
    log_tau = keras.Variable(
        initializer=keras.initializers.Zeros(), shape=(),
        dtype='float32', trainable=True, name='log_tau'
    )
    tau     = ops.exp(log_tau)
    score_t = score / (tau + 1e-6)
    alpha   = Softmax(axis=1, name='temporal_alpha')(score_t)
    alpha_T = ops.transpose(alpha, axes=(0, 2, 1))
    context = ops.matmul(alpha_T, x)
    context = Reshape((100,), name='context')(context)

    # ── Dense classifier head ─────────────────────────────────────────────────
    out = Dense(128, activation='selu', name='fc1',
                kernel_regularizer=keras.regularizers.L2(3e-3))(context)
    out = Dropout(dr, name='drop1')(out)
    out = Dense(128, activation='selu', name='fc2',
                kernel_regularizer=keras.regularizers.L2(3e-3))(out)
    out = Dropout(dr, name='drop2')(out)
    softmax_out = Dense(classes, activation='softmax', name='softmax')(out)

    return [input1, input2, input3], softmax_out, attn_weights


# ── Public API ─────────────────────────────────────────────────────────────────

def build_mcldnn_attention(classes: int = 5,
                           dropout_rate: float = 0.5) -> Model:
    """
    Build and compile the MCLDNN-Attention TRAINING model.

    Single output: softmax (batch, classes).
    Compiled: categorical_crossentropy + Adam(clipnorm=1.0).
    """
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)
    model = Model(inputs=inputs, outputs=softmax_out, name='MCLDNN_Attention')
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    )
    return model


def build_mcldnn_attention_extractor(classes: int = 5,
                                     dropout_rate: float = 0.5,
                                     weights_path: str = None) -> Model:
    """
    Build the MCLDNN-Attention EXTRACTOR for interpretability.

    Dual output: [softmax (batch, classes), attn_weights_block2 (batch, 4, 124, 124)]
    NOT compiled. Use only for model.predict().
    """
    import os
    inputs, softmax_out, attn_weights = _build_graph(classes, dropout_rate)
    extractor = Model(inputs=inputs,
                      outputs=[softmax_out, attn_weights],
                      name='MCLDNN_Attention_Extractor')

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                "Train the model first with build_mcldnn_attention()."
            )
        extractor.load_weights(weights_path)
        print(f"[extractor] Loaded weights from {weights_path}")

    return extractor


# ── Standalone verification ───────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    keras.mixed_precision.set_global_policy('float32')

    print("=== MCLDNN-Attention v3 (2-layer Transformer Encoder) ===")
    model = build_mcldnn_attention(classes=5, dropout_rate=0.6)
    model.summary()

    total = model.count_params()
    print(f"\nTotal trainable parameters: {total:,}")
    assert total < 300_000, f"FAIL: {total:,} params exceeds 300,000 limit"
    print("Parameter constraint (<300,000): PASS")

    dummy1 = np.zeros((4, 2, 128, 1),  dtype='float32')
    dummy2 = np.zeros((4, 128, 1),      dtype='float32')
    dummy3 = np.zeros((4, 128, 1),      dtype='float32')

    out = model.predict([dummy1, dummy2, dummy3], verbose=0)
    assert out.shape == (4, 5)
    print(f"Training model output shape: {out.shape}  PASS")

    extractor = build_mcldnn_attention_extractor(classes=5)
    soft, attn = extractor.predict([dummy1, dummy2, dummy3], verbose=0)
    assert soft.shape == (4, 5)
    assert attn.shape == (4, 4, 124, 124)
    print(f"Extractor softmax: {soft.shape}  PASS")
    print(f"Extractor attn:    {attn.shape}  PASS")
    print("\nAll checks PASSED")
