"""
signal_features.py — deterministic IQ-derived physical features
===============================================================

These helpers compute robust, rotation-aware features from normalized IQ
windows.  They are intentionally deterministic: no fitted scaler, no learned
state, and no change to dataset splits or labels.

Feature design
--------------
QAM16 vs QAM64 is mainly an amplitude-distribution problem, so the amplitude
branch exposes radius and short-term radius variation:

    [A(t), A(t)^2, ΔA(t), |ΔA(t)|]

QPSK vs 8PSK is mainly a phase-transition problem.  Absolute phase can rotate
with the channel, so the phase branch uses differential phase:

    dphi(t) = angle(x[t] * conj(x[t-1]))
    [cos(dphi), sin(dphi)]
"""

from __future__ import annotations

import numpy as np


def _validate_iq_array(X_raw: np.ndarray) -> np.ndarray:
    """Return X_raw as float32 after validating shape and finite values."""
    X = np.asarray(X_raw, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(
            "Expected IQ array with shape (N, 2, T); "
            f"got {X.shape!r}."
        )
    if X.shape[2] < 2:
        raise ValueError(
            "Expected at least two time steps for differential phase; "
            f"got T={X.shape[2]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("IQ array contains NaN or infinite values.")
    return X


def _normalized_amplitude(X_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute per-sample mean-normalized amplitude A(t).

    The dataset loader already RMS-normalizes IQ windows.  This additional
    mean-amplitude normalization makes the amplitude branch focus on relative
    radius structure inside the window, not absolute received power.
    """
    X = _validate_iq_array(X_raw)
    I = X[:, 0, :]
    Q = X[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)
    amp = amp / (np.mean(amp, axis=1, keepdims=True) + eps)
    return amp.astype(np.float32)


def extract_amplitude(X_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Backward-compatible instantaneous amplitude helper.

    Returns
    -------
    np.ndarray
        Shape (N, T, 1), dtype float32.
    """
    amp = _normalized_amplitude(X_raw, eps=eps)
    return amp[..., np.newaxis].astype(np.float32)


def extract_amplitude_features(X_raw: np.ndarray,
                               eps: float = 1e-8) -> np.ndarray:
    """
    Compute the QAM-focused amplitude feature stack.

    Channels
    --------
    0. A(t)          : mean-normalized radius
    1. A(t)^2        : radius energy / stronger outer-ring cue
    2. ΔA(t)         : signed local radius change
    3. |ΔA(t)|       : local amplitude variability

    Returns
    -------
    np.ndarray
        Shape (N, T, 4), dtype float32.
    """
    amp = _normalized_amplitude(X_raw, eps=eps)
    amp_sq = amp ** 2
    delta_amp = np.diff(amp, axis=1, prepend=amp[:, :1])
    abs_delta_amp = np.abs(delta_amp)

    return np.stack(
        [amp, amp_sq, delta_amp, abs_delta_amp],
        axis=-1,
    ).astype(np.float32)


def extract_phase_sincos(X_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Backward-compatible instantaneous phase unit-vector helper.

    For the v2 physical model, prefer extract_differential_phase_sincos().

    Returns
    -------
    np.ndarray
        Shape (N, T, 2), dtype float32, with channels [cos(phi), sin(phi)].
    """
    X = _validate_iq_array(X_raw)
    I = X[:, 0, :]
    Q = X[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)
    phase_cos = I / (amp + eps)
    phase_sin = Q / (amp + eps)
    return np.stack([phase_cos, phase_sin], axis=-1).astype(np.float32)


def extract_differential_phase_sincos(X_raw: np.ndarray,
                                      eps: float = 1e-8) -> np.ndarray:
    """
    Compute rotation-aware differential phase features.

    For complex samples x[t] = I[t] + jQ[t]:

        x[t] * conj(x[t-1])

    has angle dphi(t), the phase change between consecutive samples.  Encoding
    dphi as [cos(dphi), sin(dphi)] avoids atan2 wrap discontinuities and is
    more robust to global constellation rotation than absolute phase.

    The first timestep has no previous sample, so it is encoded as no phase
    change: cos(dphi)=1, sin(dphi)=0.

    Returns
    -------
    np.ndarray
        Shape (N, T, 2), dtype float32, with channels
        [cos(dphi), sin(dphi)].
    """
    X = _validate_iq_array(X_raw)
    I = X[:, 0, :]
    Q = X[:, 1, :]
    N, T = I.shape

    cos_dphi = np.ones((N, T), dtype=np.float32)
    sin_dphi = np.zeros((N, T), dtype=np.float32)

    real = I[:, 1:] * I[:, :-1] + Q[:, 1:] * Q[:, :-1]
    imag = Q[:, 1:] * I[:, :-1] - I[:, 1:] * Q[:, :-1]
    mag = np.sqrt(real ** 2 + imag ** 2)
    valid = mag > eps

    cos_part = np.ones_like(real, dtype=np.float32)
    sin_part = np.zeros_like(imag, dtype=np.float32)
    cos_part[valid] = real[valid] / mag[valid]
    sin_part[valid] = imag[valid] / mag[valid]

    cos_dphi[:, 1:] = cos_part
    sin_dphi[:, 1:] = sin_part

    return np.stack([cos_dphi, sin_dphi], axis=-1).astype(np.float32)


def prepare_physical_features(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the physical feature tensors expected by mcldnn_attention_phys v2.

    Returns
    -------
    (X_amp, X_phase)
        X_amp shape   : (N, T, 4)  = [A, A², ΔA, |ΔA|]
        X_phase shape : (N, T, 2)  = [cos(dphi), sin(dphi)]
    """
    return (
        extract_amplitude_features(X_raw),
        extract_differential_phase_sincos(X_raw),
    )

