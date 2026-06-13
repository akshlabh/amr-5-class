"""
attention_analysis.py — Reusable utilities for attention weight analysis
========================================================================
Used by evaluate_attention.py and notebook cells to extract, aggregate,
and compare attention weights from the MCLDNN-Attention model.

All functions work on numpy arrays and have no Keras/TF imports at
module level — they can be imported safely in any context.
"""

import os
import pickle
import warnings
import numpy as np


def extract_attention_weights(model, inputs, n_samples: int = 50):
    """
    Run model.predict() and return the attention weights output.

    The MCLDNN-Attention model has two outputs:
      [softmax_out, attn_weights]
    This helper extracts only the second output.

    Parameters
    ----------
    model     : keras.Model   Dual-output MCLDNN-Attention model
    inputs    : list of np.ndarray  Model inputs [X_IQ, X_I, X_Q]
                each sliced to n_samples along axis 0.
    n_samples : int  Maximum number of samples to use.  If the input
                has fewer samples, all are used.

    Returns
    -------
    np.ndarray  shape (N, num_heads, seq_len, seq_len)
                           = (N, 4, 124, 124)
    """
    n = min(n_samples, inputs[0].shape[0])
    batch = [inp[:n] for inp in inputs]
    _, attn_w = model.predict(batch, verbose=0)
    return attn_w   # (N, 4, 124, 124)


def mean_attention_profile(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute the mean attention profile across heads and query positions.

    Reduces (N, 4, 124, 124) → (N, 124) by:
      1. Mean over heads     (axis=1): (N, 124, 124)
      2. Mean over query pos (axis=1 after step 1, i.e. axis=-2 of original):
         (N, 124)

    The result is the average attention weight each key time-step receives,
    averaged over all query positions and all heads.  This is the attention
    analogue of the SHAP temporal profile from Task 2.

    Parameters
    ----------
    attn_weights : np.ndarray  shape (N, 4, 124, 124)
                               [batch, heads, query, key]

    Returns
    -------
    np.ndarray  shape (N, 124)   one profile per sample
    """
    # Mean over heads: (N, 124, 124)
    mean_heads = attn_weights.mean(axis=1)
    # Mean over query positions (axis=1 of mean_heads): (N, 124)
    profile = mean_heads.mean(axis=1)
    return profile


def attention_par(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute the Peak-to-Average Ratio (PAR) of the attention profile.

    For each sample: PAR = max(profile) / mean(profile)

    High PAR means the model concentrates attention on a few key time steps
    (focused, like a spike in the SHAP profile).  Low PAR means uniform
    attention (the model finds no particularly informative time step).

    This metric is directly comparable to the SHAP PAR computed in Task 4.

    Parameters
    ----------
    attn_weights : np.ndarray  shape (N, 4, 124, 124)

    Returns
    -------
    np.ndarray  shape (N,)  PAR value per sample
    """
    profile = mean_attention_profile(attn_weights)   # (N, 124)
    # PAR per sample: max / mean over the 124 time steps
    par = profile.max(axis=1) / (profile.mean(axis=1) + 1e-10)
    return par


def compare_shap_vs_attention(shap_pkl_path: str,
                              attn_weights_dir: str,
                              snrs: list) -> list:
    """
    Compare SHAP temporal profiles with attention profiles at each SNR.

    Loads:
      - task4_data.pkl from shap_pkl_path  (contains mean_abs_I, mean_abs_Q)
      - attn_weights_snr{snr}.npy from attn_weights_dir

    For each SNR, computes Pearson correlation between:
      mean_shap_profile  = (mean_abs_I[snr] + mean_abs_Q[snr]) / 2  (128 steps)
      mean_attn_profile  = mean_attention_profile(attn_weights)[...] (124 steps)

    Note: SHAP profiles have 128 time steps, attention profiles have 124
    (due to Conv2D valid padding removing 4 steps).  We align by truncating
    the SHAP profile to the first 124 steps.

    Parameters
    ----------
    shap_pkl_path    : str  Path to task4_data.pkl
    attn_weights_dir : str  Directory containing attn_weights_snr*.npy files
    snrs             : list  SNR values to compare (e.g. [-6,-4,-2,0,2,4,6])

    Returns
    -------
    list of dict  [{snr, pearson_r, p_value, n_samples}, ...]
    """
    from scipy.stats import pearsonr

    if not os.path.exists(shap_pkl_path):
        warnings.warn(f"SHAP data not found: {shap_pkl_path}")
        return []

    with open(shap_pkl_path, 'rb') as f:
        task4 = pickle.load(f)

    # task4 keys: 'mean_abs_I', 'mean_abs_Q', 'snrs'
    # Each is a dict {snr: np.array(128,)}
    mean_abs_I = task4.get('mean_abs_I', {})
    mean_abs_Q = task4.get('mean_abs_Q', {})

    results = []
    for snr in snrs:
        attn_path = os.path.join(attn_weights_dir,
                                 f'attn_weights_snr{snr:+d}.npy')
        if not os.path.exists(attn_path):
            warnings.warn(f"Attention weights not found for SNR={snr}: {attn_path}")
            continue

        attn_w   = np.load(attn_path)                      # (N, 4, 124, 124)
        attn_prof = mean_attention_profile(attn_w)          # (N, 124)
        mean_attn = attn_prof.mean(axis=0)                  # (124,)

        # Build SHAP profile for this SNR
        shap_I = mean_abs_I.get(snr)
        shap_Q = mean_abs_Q.get(snr)
        if shap_I is None or shap_Q is None:
            warnings.warn(f"SHAP data missing for SNR={snr}")
            continue
        shap_prof = (np.asarray(shap_I) + np.asarray(shap_Q)) / 2.0  # (128,)
        shap_prof = shap_prof[:124]   # align to 124 (conv valid padding)

        r, p = pearsonr(shap_prof, mean_attn)
        results.append({
            'snr':       snr,
            'pearson_r': float(r),
            'p_value':   float(p),
            'n_samples': attn_w.shape[0]
        })
        print(f"  SNR={snr:+3d} dB | r={r:+.3f} | p={p:.4f} | "
              f"n={attn_w.shape[0]}")

    return results
