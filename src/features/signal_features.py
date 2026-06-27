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


def estimate_qam16_amplitude_boundary(X_raw: np.ndarray,
                                      y_onehot: np.ndarray,
                                      classes: list[str],
                                      quantile: float = 95.0,
                                      eps: float = 1e-8) -> float:
    """
    Estimate a robust QAM16 outer-amplitude boundary from training data only.

    The boundary is computed on mean-normalized amplitude A(t).  Using a high
    percentile rather than a hard maximum makes the threshold resistant to
    noisy outliers while still describing the practical QAM16 outer envelope.

    Parameters
    ----------
    X_raw : np.ndarray
        Training IQ windows, shape (N, 2, T).
    y_onehot : np.ndarray
        One-hot labels for the same N windows.
    classes : list[str]
        Class names in one-hot order.
    quantile : float
        Percentile of QAM16 amplitudes used as the boundary.

    Returns
    -------
    float
        Scalar threshold T such that A(t) > T is treated as a QAM16-boundary
        exceedance cue.
    """
    if "QAM16" not in classes:
        raise ValueError("QAM16 must be present to estimate its amplitude boundary.")
    Y = np.asarray(y_onehot)
    if Y.ndim != 2 or Y.shape[0] != np.asarray(X_raw).shape[0]:
        raise ValueError(
            "Expected y_onehot with shape (N, C) matching X_raw; "
            f"got y={Y.shape!r}, X={np.asarray(X_raw).shape!r}."
        )

    qam16_idx = classes.index("QAM16")
    mask = np.argmax(Y, axis=1) == qam16_idx
    if not np.any(mask):
        raise ValueError("No QAM16 samples found in the provided training labels.")

    amp_qam16 = _normalized_amplitude(np.asarray(X_raw)[mask], eps=eps)
    boundary = float(np.percentile(amp_qam16.reshape(-1), quantile))
    if not np.isfinite(boundary) or boundary <= 0:
        raise ValueError(f"Invalid QAM16 amplitude boundary: {boundary!r}")
    return boundary


def estimate_qam16_amplitude_boundaries(X_raw: np.ndarray,
                                        y_onehot: np.ndarray,
                                        classes: list[str],
                                        quantiles: tuple[float, ...] | list[float] = (95.0, 98.0, 99.0),
                                        eps: float = 1e-8) -> np.ndarray:
    """
    Estimate multiple QAM16 amplitude boundaries from training data only.

    A single 95th-percentile boundary is useful for detecting outer-radius
    events, but it can over-fire on QAM16 noise peaks.  Multiple boundaries
    let the model distinguish:

      * ordinary upper-tail QAM16 samples     (above q95),
      * stronger but still plausible peaks    (above q98),
      * extreme QAM16-tail crossings          (above q99).

    QAM64 should show more consistent evidence across the stronger boundaries,
    not merely one isolated q95 crossing.
    """
    qs = np.asarray(quantiles, dtype=np.float32).reshape(-1)
    if qs.size < 1:
        raise ValueError("quantiles must contain at least one percentile.")
    if not np.all(np.isfinite(qs)) or np.any(qs <= 0) or np.any(qs >= 100):
        raise ValueError(f"Invalid QAM16 amplitude boundary quantiles: {quantiles!r}")
    qs = np.sort(qs)

    if "QAM16" not in classes:
        raise ValueError("QAM16 must be present to estimate its amplitude boundaries.")
    Y = np.asarray(y_onehot)
    if Y.ndim != 2 or Y.shape[0] != np.asarray(X_raw).shape[0]:
        raise ValueError(
            "Expected y_onehot with shape (N, C) matching X_raw; "
            f"got y={Y.shape!r}, X={np.asarray(X_raw).shape!r}."
        )

    qam16_idx = classes.index("QAM16")
    mask = np.argmax(Y, axis=1) == qam16_idx
    if not np.any(mask):
        raise ValueError("No QAM16 samples found in the provided training labels.")

    amp_qam16 = _normalized_amplitude(np.asarray(X_raw)[mask], eps=eps)
    boundaries = np.percentile(amp_qam16.reshape(-1), qs).astype(np.float32)
    if not np.isfinite(boundaries).all() or np.any(boundaries <= 0):
        raise ValueError(f"Invalid QAM16 amplitude boundaries: {boundaries!r}")
    return boundaries


def extract_amplitude_peak_features(X_raw: np.ndarray,
                                    qam16_boundary: float | np.ndarray,
                                    eps: float = 1e-8
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic QAM16/QAM64 peak and PAR features.

    This is a more targeted amplitude representation than
    extract_amplitude_features().  It makes the professor's physical cue
    explicit: QAM64 should produce more samples that cross the learned QAM16
    amplitude boundary.

    Per-time-step channels, shape (N, T, 12)
    ---------------------------------------
    0. A(t)                 : mean-normalized amplitude
    1. A(t)^2               : normalized instantaneous power
    2. delta_A(t)           : signed local amplitude change
    3. |delta_A(t)|         : local amplitude variation
    4. exceed_q95(t)        : weak outer-tail crossing
    5. exceed_q98(t)        : stronger QAM64 evidence
    6. exceed_q99(t)        : extreme QAM16-tail crossing
    7. rel_excess_q98(t)    : max(A(t)-q98,0) / q98
    8. rel_excess_q99(t)    : max(A(t)-q99,0) / q99
    9. local_density_q98(t) : 3-sample local mean of exceed_q98
    10. power_ratio(t)      : power(t) / mean_power
    11. peak_score(t)       : rel_excess_q98 * local_density_q98

    Global channels, shape (N, 13)
    -----------------------------
    0. log1p(PAR)           : log(1 + max(power) / mean(power))
    1. log1p(robust_PAR)    : log(1 + p99(power) / mean(power))
    2. peak_ratio_q95       : fraction above q95
    3. peak_ratio_q98       : fraction above q98
    4. peak_ratio_q99       : fraction above q99
    5. sustained_ratio_q98  : adjacent q98 crossing fraction
    6. mean_rel_excess_q98  : average relative excess above q98
    7. mean_rel_excess_q99  : average relative excess above q99
    8. max_rel_excess_q98   : largest relative excess above q98
    9. amplitude_std        : spread of A(t)
    10. amplitude_p95       : 95th percentile of A(t)
    11. amplitude_p99       : 99th percentile of A(t)
    12. tail_separation     : (p99 - q98) / q98

    Returns
    -------
    (seq_features, global_features)
        seq_features shape    : (N, T, 12), dtype float32
        global_features shape : (N, 13), dtype float32
    """
    boundaries = np.asarray(qam16_boundary, dtype=np.float32).reshape(-1)
    if boundaries.size == 1:
        # Backward-compatible fallback for callers that still pass one scalar.
        # Prefer train.py's [95, 98, 99] learned boundaries for real experiments.
        boundaries = np.asarray([boundaries[0], boundaries[0] * 1.06, boundaries[0] * 1.10],
                                dtype=np.float32)
    if boundaries.size < 3:
        raise ValueError(
            "qam16_boundary must contain at least three boundaries "
            "[q95, q98, q99] for robust peak/PAR features."
        )
    boundaries = np.sort(boundaries[:3]).astype(np.float32)
    if not np.isfinite(boundaries).all() or np.any(boundaries <= 0):
        raise ValueError(
            "qam16_boundary must contain positive finite values; "
            f"got {boundaries!r}."
        )
    T95, T98, T99 = [float(v) for v in boundaries]

    amp = _normalized_amplitude(X_raw, eps=eps)
    power = amp ** 2
    mean_power = np.mean(power, axis=1, keepdims=True) + eps
    power_ratio = power / mean_power
    delta_amp = np.diff(amp, axis=1, prepend=amp[:, :1])
    abs_delta_amp = np.abs(delta_amp)

    exceed95 = (amp > T95).astype(np.float32)
    exceed98 = (amp > T98).astype(np.float32)
    exceed99 = (amp > T99).astype(np.float32)

    rel_excess98 = np.maximum(amp - T98, 0.0).astype(np.float32) / (T98 + eps)
    rel_excess99 = np.maximum(amp - T99, 0.0).astype(np.float32) / (T99 + eps)

    # Local density separates isolated noisy crossings from more persistent
    # outer-level evidence.  This is intentionally simple and deterministic.
    left98 = np.concatenate([exceed98[:, :1], exceed98[:, :-1]], axis=1)
    right98 = np.concatenate([exceed98[:, 1:], exceed98[:, -1:]], axis=1)
    local_density98 = (left98 + exceed98 + right98) / 3.0
    peak_score = rel_excess98 * local_density98

    seq_features = np.stack(
        [
            amp,
            power,
            delta_amp,
            abs_delta_amp,
            exceed95,
            exceed98,
            exceed99,
            rel_excess98,
            rel_excess99,
            local_density98,
            power_ratio,
            peak_score,
        ],
        axis=-1,
    ).astype(np.float32)

    par = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)
    robust_par = np.percentile(power, 99, axis=1) / (np.mean(power, axis=1) + eps)
    peak_ratio95 = np.mean(exceed95, axis=1)
    peak_ratio98 = np.mean(exceed98, axis=1)
    peak_ratio99 = np.mean(exceed99, axis=1)
    sustained_ratio98 = np.mean(exceed98[:, 1:] * exceed98[:, :-1], axis=1)
    mean_rel_excess98 = np.mean(rel_excess98, axis=1)
    mean_rel_excess99 = np.mean(rel_excess99, axis=1)
    max_rel_excess98 = np.max(rel_excess98, axis=1)
    amp_std = np.std(amp, axis=1)
    amp_p95 = np.percentile(amp, 95, axis=1)
    amp_p99 = np.percentile(amp, 99, axis=1)
    tail_separation = (amp_p99 - T98) / (T98 + eps)

    global_features = np.stack(
        [
            np.log1p(par),
            np.log1p(robust_par),
            peak_ratio95,
            peak_ratio98,
            peak_ratio99,
            sustained_ratio98,
            mean_rel_excess98,
            mean_rel_excess99,
            max_rel_excess98,
            amp_std,
            amp_p95,
            amp_p99,
            tail_separation,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(seq_features).all():
        raise ValueError("Amplitude peak sequence features contain NaN or infinite values.")
    if not np.isfinite(global_features).all():
        raise ValueError("Amplitude peak global features contain NaN or infinite values.")

    return seq_features, global_features


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
