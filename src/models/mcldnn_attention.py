"""
mcldnn_attention.py — MCLDNN with self-attention replacing LSTM layers
=======================================================================
Architecture overview
----------------------
Conv block: IDENTICAL to mcldnn.py (same layer names, filter counts, kernel
             sizes, padding modes, activations).  Output shape: (batch, 124, 100).

Attention block (replaces both LSTM layers):
  A) Learnable positional encoding via Embedding(124, 100)
  B) Multi-head self-attention: 4 heads × 25 key_dim = 100 d_model
     Returns both output and attention weights (for interpretability).
  C) Residual + LayerNorm  (Transformer-style)
  D) Learned temporal pooling: Dense(1) → Softmax over time → weighted sum
     Equivalent to the LSTM's final hidden-state summary, but differentiable
     and visualisable.

Two public functions
---------------------
build_mcldnn_attention(classes, dropout_rate)
    Returns a SINGLE-output training model (softmax only).
    Compiled with standard categorical_crossentropy + Adam.
    Use this for model.fit() — no Y_dict needed, val_accuracy works out of
    the box with Keras callbacks.

build_mcldnn_attention_extractor(classes, dropout_rate, weights_path)
    Builds the SAME computational graph and loads the trained weights, then
    returns a DUAL-output model [softmax, attn_weights] for use in
    evaluate_attention.py / predict() calls only.
    NOT compiled — never call .fit() on this model.

Why two models instead of one?
    Keras 3 requires a loss entry for EVERY named model output.  If attn_weights
    is an output during training, Keras demands loss={'softmax': ..., 'mha': ...}.
    There is no clean way to specify "no loss" for a named output in dict-keyed
    compile.  Building a single-output training model avoids this entirely while
    keeping the extractor available for interpretability.

Why self-attention here?
  The LSTM processes the 124-step sequence left-to-right with a fixed hidden
  state.  Self-attention can directly compare any two time steps regardless
  of distance — important for QAM signals whose symbol statistics repeat
  cyclically.  The attention weights also serve as a direct interpretability
  tool comparable to the SHAP temporal profiles from Task 2/4.

Constraints satisfied
  • Same three inputs as MCLDNN: [input1(2,128,1), input2(128,1), input3(128,1)]
  • No new pip dependencies (all layers are in keras.layers)
  • Total params < 300,000 (verified in __main__ block)
  • keras.mixed_precision.set_global_policy('float32') called first in __main__
"""

import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate,
    MultiHeadAttention, LayerNormalization,
    Softmax, Embedding,
)
import keras.ops as ops
import numpy as np


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Returns array of shape (1, seq_len, d_model).  Does NOT require
    any gradient updates — position information is mathematically
    correct from epoch 0, unlike a learned Embedding which needs
    thousands of samples to converge to meaningful position vectors.
    """
    pos = np.arange(seq_len)[:, None]               # (seq_len, 1)
    i   = np.arange(d_model)[None, :]               # (1, d_model)
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    # Even indices: sin, odd indices: cos
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle[None, :, :].astype(np.float32)     # (1, seq_len, d_model)


# ── Shared graph builder ───────────────────────────────────────────────────────
def _build_graph(classes: int, dropout_rate: float, attn_dropout: float = 0.1):
    """
    Build the full MCLDNN-Attention computational graph and return all
    tensors needed to assemble either the training model or the extractor.

    Returns
    -------
    inputs       : list [input1, input2, input3]
    softmax_out  : tensor  (batch, classes) — used as training model output
    attn_weights : tensor  (batch, 4, 124, 124) — used by extractor only
    """
    dr = dropout_rate

    # ── Inputs (identical to mcldnn.py) ───────────────────────────────────────
    input1 = Input(shape=(2, 128, 1), name='input1')   # (batch, 2, 128, 1)
    input2 = Input(shape=(128, 1),    name='input2')   # (batch, 128, 1)
    input3 = Input(shape=(128, 1),    name='input3')   # (batch, 128, 1)

    # ── CONV BLOCK — identical to mcldnn.py (same names, sizes, padding) ──────

    # Branch 1: 2D CNN on the combined IQ frame
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input1)
    # x1 shape: (batch, 2, 128, 50)

    # Branch 2: 1D CNN on I channel
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input2)
    # x2 shape: (batch, 128, 50)
    x2 = Reshape((1, 128, 50), name='reshape_x2')(x2)
    # x2 shape: (batch, 1, 128, 50)

    # Branch 3: 1D CNN on Q channel
    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3',
                kernel_regularizer=keras.regularizers.L2(1e-4))(input3)
    # x3 shape: (batch, 128, 50)
    x3 = Reshape((1, 128, 50), name='reshape_x3')(x3)
    # x3 shape: (batch, 1, 128, 50)

    # Combine I + Q branches, apply shared 2D conv
    x = concatenate([x2, x3], axis=1)               # (batch, 2, 128, 50)
    x = Conv2D(50, (1, 8), padding='same', activation='relu',
               name='conv2',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x shape: (batch, 2, 128, 50)

    # Merge with Branch 1
    x = concatenate([x1, x], axis=-1)               # (batch, 2, 128, 100)
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv4',
               kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    # x shape: (batch, 1, 124, 100)

    # Reshape for sequence model
    x = Reshape((124, 100), name='reshape_final')(x) # (batch, 124, 100)

    # ── ATTENTION BLOCK — replaces both LSTM layers ────────────────────────────

    # Step A: Fixed sinusoidal positional encoding (Vaswani et al. 2017)
    pe = ops.convert_to_tensor(
        _sinusoidal_encoding(seq_len=124, d_model=100),
        dtype='float32'
    )                                               # (1, 124, 100)
    x = x + pe                                      # (batch, 124, 100)

    # Step B: Multi-head self-attention (4 heads)
    mha = MultiHeadAttention(num_heads=4, key_dim=25, dropout=attn_dropout,
                             name='mha')
    attn_out, attn_weights = mha(
        query=x, value=x, key=x,
        return_attention_scores=True
    )
    # attn_out shape    : (batch, 124, 100)
    # attn_weights shape: (batch, 4, 124, 124)

    # Step C: Residual connection + Layer Normalisation + post-attention dropout
    x = LayerNormalization(name='attn_norm')(attn_out + x)  # (batch, 124, 100)
    x = Dropout(attn_dropout, name='attn_drop')(x)           # (batch, 124, 100)

    # Step D: FFN (Transformer-style point-wise) projects to 100
    ffn = Dense(256, activation='relu', name='ffn1',
                kernel_regularizer=keras.regularizers.L2(1e-4))(x)
    ffn = Dropout(attn_dropout, name='ffn_drop')(ffn)
    ffn = Dense(100, name='ffn2',
                kernel_regularizer=keras.regularizers.L2(1e-4))(ffn)
    x = LayerNormalization(name='ffn_norm')(ffn + x)   # (batch, 124, 100)

    # Stats pooling
    x_mean = ops.mean(x, axis=1)                        # (batch, 100)
    x_max  = ops.max(x, axis=1)                         # (batch, 100)
    context = concatenate([x_mean, x_max], axis=-1)     # (batch, 200)

    # ── Dense classifier head ────────────────────────────────────────────────────
    # L2 increased from 1e-3 → 3e-3 to add stronger weight-decay pressure on
    # the fully-connected layers.  With only 21k training samples the model
    # can easily memorise IQ sequences via large fc weights; stronger L2
    # keeps them small and forces generalisation.
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
                           dropout_rate: float = 0.6,
                           learning_rate: float = 1e-3) -> Model:
    """
    Build and compile the MCLDNN-Attention TRAINING model.

    Single output: softmax classification (batch, classes).
    Compiled with standard categorical_crossentropy + Adam, so Keras
    callbacks (val_accuracy, ModelCheckpoint, ReduceLROnPlateau) all work
    without any special Y_dict — just pass Y_train directly to model.fit().

    Parameters
    ----------
    classes      : int   Number of output classes (default 5)
    dropout_rate : float Dropout probability after each Dense fc layer

    Returns
    -------
    keras.Model  single-output training model
    """
    inputs, softmax_out, _ = _build_graph(classes, dropout_rate)

    model = Model(inputs=inputs, outputs=softmax_out, name='MCLDNN_Attention')

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    )
    return model


def build_mcldnn_attention_extractor(classes: int = 5,
                                     dropout_rate: float = 0.6,
                                     weights_path: str = None) -> Model:
    """
    Build the MCLDNN-Attention EXTRACTOR model for interpretability.

    Dual output: [softmax (batch, classes), attn_weights (batch, 4, 124, 124)]
    Loads weights from a checkpoint trained by build_mcldnn_attention() —
    the graph is identical, so weight loading works layer-by-layer by name.

    NOT compiled.  Use only for model.predict() calls in evaluate_attention.py.

    Parameters
    ----------
    classes      : int   Number of output classes (default 5)
    dropout_rate : float Must match the value used during training
    weights_path : str or None  Path to .weights.h5 checkpoint; None = random

    Returns
    -------
    keras.Model  dual-output extractor model
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


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    keras.mixed_precision.set_global_policy('float32')

    print("=== MCLDNN-Attention training model (5-class) ===")
    model = build_mcldnn_attention(classes=5)
    model.summary()

    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params — exceeds 300,000 limit!"
    )
    print("Param count check PASSED (<300,000)")

    # Verify output shapes
    import numpy as np
    dummy1 = np.zeros((4, 2, 128, 1), dtype='float32')
    dummy2 = np.zeros((4, 128, 1),    dtype='float32')
    dummy3 = np.zeros((4, 128, 1),    dtype='float32')

    # Training model: single output
    out_softmax = model.predict([dummy1, dummy2, dummy3], verbose=0)
    print(f"Training model output shape: {out_softmax.shape}  (expect (4, 5))")
    assert out_softmax.shape == (4, 5), f"Wrong shape: {out_softmax.shape}"

    # Extractor model: dual output
    print("\n=== MCLDNN-Attention extractor (dual output) ===")
    extractor = build_mcldnn_attention_extractor(classes=5)
    out_soft2, out_attn = extractor.predict([dummy1, dummy2, dummy3], verbose=0)
    print(f"Extractor softmax shape    : {out_soft2.shape}  (expect (4, 5))")
    print(f"Extractor attn_weights shape: {out_attn.shape}  (expect (4, 4, 124, 124))")
    assert out_soft2.shape == (4, 5)
    assert out_attn.shape  == (4, 4, 124, 124), f"Wrong attn shape: {out_attn.shape}"
    print("All shape checks PASSED")
