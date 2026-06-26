"""
signal_features.py — deterministic IQ-derived features
======================================================

The neural models already receive raw IQ samples, but some modulation
differences are easier to expose in polar coordinates:

  • QAM16 vs QAM64: relative amplitude/radius distribution
  • QPSK vs 8PSK  : phase-state structure

These helpers compute per-timestep features from normalized IQ samples without
changing the underlying dataset split or labels.
"""

import numpy as np


def extract_amplitude(X_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute normalized instantaneous amplitude.

    Parameters
    ----------
    X_raw : np.ndarray
        IQ samples with shape (N, 2, 128), where channel 0 is I and channel 1
        is Q. The repository loader already RMS-normalizes each sample.
    eps : float
        Small numerical guard against division by zero.

    Returns
    -------
    np.ndarray
        Shape (N, 128, 1), dtype float32. Amplitude is divided by each sample's
        mean amplitude so the model sees relative radius variation instead of
        absolute received power.
    """
    I = X_raw[:, 0, :]
    Q = X_raw[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)
    amp = amp / (np.mean(amp, axis=1, keepdims=True) + eps)
    return amp[..., np.newaxis].astype(np.float32)


def extract_phase_sincos(X_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute wrap-safe instantaneous phase features.

    Instead of feeding raw atan2 phase, which has a discontinuity at +π/-π, we
    use the equivalent unit-vector representation:

        cos(phi) = I / sqrt(I² + Q²)
        sin(phi) = Q / sqrt(I² + Q²)

    Parameters
    ----------
    X_raw : np.ndarray
        IQ samples with shape (N, 2, 128).
    eps : float
        Small numerical guard against division by zero.

    Returns
    -------
    np.ndarray
        Shape (N, 128, 2), dtype float32, with channels [cos(phi), sin(phi)].
    """
    I = X_raw[:, 0, :]
    Q = X_raw[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)
    phase_cos = I / (amp + eps)
    phase_sin = Q / (amp + eps)
    return np.stack([phase_cos, phase_sin], axis=-1).astype(np.float32)


def prepare_physical_features(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper returning amplitude and phase feature tensors.

    Returns
    -------
    (X_amp, X_phase)
        X_amp shape   : (N, 128, 1)
        X_phase shape : (N, 128, 2)
    """
    return extract_amplitude(X_raw), extract_phase_sincos(X_raw)

