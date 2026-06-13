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

Dual output: [softmax_logits, attn_weights]
  attn_weights shape: (batch, num_heads, seq_len, seq_len) = (batch, 4, 124, 124)
  This allows evaluate_attention.py to extract attention maps without
  rebuilding the model.

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

import os
import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, Conv2D,
    Dropout, Reshape, concatenate,
    MultiHeadAttention, LayerNormalization,
    Softmax, Embedding, Lambda
)
import keras.ops as ops


def build_mcldnn_attention(classes: int = 5,
                           dropout_rate: float = 0.5) -> Model:
    """
    Build MCLDNN-Attention model.

    Parameters
    ----------
    classes      : int   Number of output classes (default 5)
    dropout_rate : float Dropout probability after each Dense fc layer

    Returns
    -------
    keras.Model with:
      inputs  : [input1 (2,128,1), input2 (128,1), input3 (128,1)]
      outputs : [softmax_out (batch, classes), attn_weights (batch, 4, 124, 124)]
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

    # Step A: Learnable positional encoding
    # Create integer positions [0..123] and embed them into 100-dim space.
    # Using a Lambda layer to generate the position indices on-the-fly so the
    # graph is fully symbolic (no Python-side tensor dependency).
    pos_embedding = Embedding(input_dim=124, output_dim=100,
                              name='pos_embedding')
    # Generate position indices: shape (1, 124), will broadcast over batch
    positions = ops.arange(0, 124, dtype='int32')        # (124,)
    positions = ops.reshape(positions, (1, 124))          # (1, 124)
    pos_enc   = pos_embedding(positions)                  # (1, 124, 100)
    x = x + pos_enc                                       # (batch, 124, 100)

    # Step B: Multi-head self-attention (4 heads, key_dim=25 → d_model=100)
    mha = MultiHeadAttention(num_heads=4, key_dim=25, name='mha')
    attn_out, attn_weights = mha(
        query=x, value=x, key=x,
        return_attention_scores=True
    )
    # attn_out shape    : (batch, 124, 100)
    # attn_weights shape: (batch, 4, 124, 124)

    # Step C: Residual connection + Layer Normalisation
    x = LayerNormalization(name='attn_norm')(attn_out + x)  # (batch, 124, 100)

    # Step D: Learned temporal pooling (replaces LSTM's final hidden state)
    # Dense(1) over each time step → scalar score → Softmax across time
    # → weighted sum → context vector (batch, 100)
    score  = Dense(1, name='temporal_score')(x)             # (batch, 124, 1)
    alpha  = Softmax(axis=1, name='temporal_alpha')(score)  # (batch, 124, 1)
    # Weighted sum over time dimension: alpha * x summed over axis 1
    # Use Dot on axes [1,1] to compute sum_t alpha_t * x_t
    # alpha: (batch, 124, 1), x: (batch, 124, 100)
    # Transpose alpha to (batch, 1, 124) then matmul with x (batch, 124, 100)
    alpha_T = keras.ops.transpose(alpha, axes=(0, 2, 1))    # (batch, 1, 124)
    context = keras.ops.matmul(alpha_T, x)                  # (batch, 1, 100)
    context = Reshape((100,), name='context')(context)      # (batch, 100)

    # ── Dense classifier head — identical to mcldnn.py ─────────────────────────
    out = Dense(128, activation='selu', name='fc1',
                kernel_regularizer=keras.regularizers.L2(1e-3))(context)
    out = Dropout(dr, name='drop1')(out)
    out = Dense(128, activation='selu', name='fc2',
                kernel_regularizer=keras.regularizers.L2(1e-3))(out)
    out = Dropout(dr, name='drop2')(out)
    softmax_out = Dense(classes, activation='softmax', name='softmax')(out)

    # ── Build dual-output model ────────────────────────────────────────────────
    model = Model(
        inputs=[input1, input2, input3],
        outputs=[softmax_out, attn_weights],
        name='MCLDNN_Attention'
    )

    # ── Compile ────────────────────────────────────────────────────────────────
    model.compile(
        loss={'softmax': 'categorical_crossentropy'},
        metrics={'softmax': ['accuracy']},
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    )

    return model


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import keras
    keras.mixed_precision.set_global_policy('float32')

    print("=== MCLDNN-Attention (5-class) ===")
    model = build_mcldnn_attention(classes=5)
    model.summary()

    total_params = sum(
        w.numpy().size for w in model.trainable_weights
    ) if hasattr(model.trainable_weights[0], 'numpy') else model.count_params()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    assert model.count_params() < 300_000, (
        f"Model has {model.count_params():,} params — exceeds 300,000 limit!"
    )
    print("Param count check PASSED (<300,000)")

    # Verify output shapes with dummy batch
    import numpy as np
    dummy1 = np.zeros((4, 2, 128, 1), dtype='float32')
    dummy2 = np.zeros((4, 128, 1),    dtype='float32')
    dummy3 = np.zeros((4, 128, 1),    dtype='float32')
    out_softmax, out_attn = model.predict([dummy1, dummy2, dummy3], verbose=0)
    print(f"softmax output shape : {out_softmax.shape}   (expect (4, 5))")
    print(f"attn_weights shape   : {out_attn.shape}      (expect (4, 4, 124, 124))")
    assert out_softmax.shape == (4, 5),       f"Wrong softmax shape: {out_softmax.shape}"
    assert out_attn.shape    == (4, 4, 124, 124), f"Wrong attn shape: {out_attn.shape}"
    print("Shape checks PASSED")
