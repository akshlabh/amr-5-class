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


def extract_amplitude_peak_features(X_raw: np.ndarray,
                                    qam16_boundary: float,
                                    eps: float = 1e-8
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic QAM16/QAM64 peak and PAR features.

    This is a more targeted amplitude representation than
    extract_amplitude_features().  It makes the professor's physical cue
    explicit: QAM64 should produce more samples that cross the learned QAM16
    amplitude boundary.

    Per-time-step channels, shape (N, T, 8)
    --------------------------------------
    0. A(t)                 : mean-normalized amplitude
    1. A(t)^2               : normalized instantaneous power
    2. delta_A(t)           : signed local amplitude change
    3. |delta_A(t)|         : local amplitude variation
    4. exceed_T(t)          : 1 if A(t) > T_qam16 else 0
    5. excess_T(t)          : max(A(t) - T_qam16, 0)
    6. relative_excess_T(t) : excess_T(t) / T_qam16
    7. excess_power_T(t)    : relative_excess_T(t)^2

    Global channels, shape (N, 7)
    ----------------------------
    0. log1p(PAR)           : log(1 + max(power) / mean(power))
    1. peak_ratio           : fraction of samples crossing T_qam16
    2. mean_excess          : mean amount above T_qam16
    3. max_excess           : largest amount above T_qam16
    4. amplitude_std        : spread of A(t)
    5. amplitude_p95        : 95th percentile of A(t)
    6. amplitude_p99        : 99th percentile of A(t)

    Returns
    -------
    (seq_features, global_features)
        seq_features shape    : (N, T, 8), dtype float32
        global_features shape : (N, 7), dtype float32
    """
    T = float(qam16_boundary)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"qam16_boundary must be a positive finite scalar; got {T!r}.")

    amp = _normalized_amplitude(X_raw, eps=eps)
    power = amp ** 2
    delta_amp = np.diff(amp, axis=1, prepend=amp[:, :1])
    abs_delta_amp = np.abs(delta_amp)

    exceed = (amp > T).astype(np.float32)
    excess = np.maximum(amp - T, 0.0).astype(np.float32)
    rel_excess = excess / (T + eps)
    excess_power = rel_excess ** 2

    seq_features = np.stack(
        [
            amp,
            power,
            delta_amp,
            abs_delta_amp,
            exceed,
            excess,
            rel_excess,
            excess_power,
        ],
        axis=-1,
    ).astype(np.float32)

    par = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)
    peak_ratio = np.mean(exceed, axis=1)
    mean_excess = np.mean(excess, axis=1)
    max_excess = np.max(excess, axis=1)
    amp_std = np.std(amp, axis=1)
    amp_p95 = np.percentile(amp, 95, axis=1)
    amp_p99 = np.percentile(amp, 99, axis=1)

    global_features = np.stack(
        [
            np.log1p(par),
            peak_ratio,
            mean_excess,
            max_excess,
            amp_std,
            amp_p95,
            amp_p99,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(seq_features).all():
        raise ValueError("Amplitude peak sequence features contain NaN or infinite values.")
    if not np.isfinite(global_features).all():
        raise ValueError("Amplitude peak global features contain NaN or infinite values.")

    return seq_features, global_features


def extract_amplitude_peak_lite_features(X_raw: np.ndarray,
                                         qam16_boundary: float,
                                         eps: float = 1e-8
                                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cleaner single-boundary PAR feature set.

    This variant intentionally drops signed delta_A(t).  For QAM16/QAM64, the
    direction of local amplitude change is less physically meaningful than the
    radius level, power, amount of variation, and boundary-crossing evidence.

    Per-time-step channels, shape (N, T, 7)
    --------------------------------------
    0. A(t)                 : mean-normalized amplitude
    1. A(t)^2               : normalized instantaneous power
    2. |delta_A(t)|         : local amplitude variation magnitude
    3. exceed_T(t)          : 1 if A(t) > T_qam16 else 0
    4. excess_T(t)          : max(A(t) - T_qam16, 0)
    5. relative_excess_T(t) : excess_T(t) / T_qam16
    6. excess_power_T(t)    : relative_excess_T(t)^2

    Global channels, shape (N, 7)
    ----------------------------
    0. log1p(PAR)
    1. peak_ratio
    2. mean_excess
    3. max_excess
    4. amplitude_std
    5. amplitude_p95
    6. amplitude_p99
    """
    T = float(qam16_boundary)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"qam16_boundary must be a positive finite scalar; got {T!r}.")

    amp = _normalized_amplitude(X_raw, eps=eps)
    power = amp ** 2
    delta_amp = np.diff(amp, axis=1, prepend=amp[:, :1])
    abs_delta_amp = np.abs(delta_amp)

    exceed = (amp > T).astype(np.float32)
    excess = np.maximum(amp - T, 0.0).astype(np.float32)
    rel_excess = excess / (T + eps)
    excess_power = rel_excess ** 2

    seq_features = np.stack(
        [
            amp,
            power,
            abs_delta_amp,
            exceed,
            excess,
            rel_excess,
            excess_power,
        ],
        axis=-1,
    ).astype(np.float32)

    par = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)
    peak_ratio = np.mean(exceed, axis=1)
    mean_excess = np.mean(excess, axis=1)
    max_excess = np.max(excess, axis=1)
    amp_std = np.std(amp, axis=1)
    amp_p95 = np.percentile(amp, 95, axis=1)
    amp_p99 = np.percentile(amp, 99, axis=1)

    global_features = np.stack(
        [
            np.log1p(par),
            peak_ratio,
            mean_excess,
            max_excess,
            amp_std,
            amp_p95,
            amp_p99,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(seq_features).all():
        raise ValueError("Amplitude peak-lite sequence features contain NaN or infinite values.")
    if not np.isfinite(global_features).all():
        raise ValueError("Amplitude peak-lite global features contain NaN or infinite values.")

    return seq_features, global_features


def extract_amplitude_static_peak_features(X_raw: np.ndarray,
                                           qam16_boundary: float,
                                           eps: float = 1e-8
                                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute static amplitude + single-boundary PAR features.

    This is the cleanest QAM16/QAM64 physical ablation: no signed delta_A(t)
    and no |delta_A(t)|.  It avoids adjacent-sample transition features because
    those can be dominated by noise at low SNR.

    Per-time-step channels, shape (N, T, 6)
    --------------------------------------
    0. A(t)                 : mean-normalized amplitude
    1. A(t)^2               : normalized instantaneous power
    2. exceed_T(t)          : 1 if A(t) > T_qam16 else 0
    3. excess_T(t)          : max(A(t) - T_qam16, 0)
    4. relative_excess_T(t) : excess_T(t) / T_qam16
    5. excess_power_T(t)    : relative_excess_T(t)^2

    Global channels, shape (N, 7)
    ----------------------------
    0. log1p(PAR)
    1. peak_ratio
    2. mean_excess
    3. max_excess
    4. amplitude_std
    5. amplitude_p95
    6. amplitude_p99
    """
    T = float(qam16_boundary)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"qam16_boundary must be a positive finite scalar; got {T!r}.")

    amp = _normalized_amplitude(X_raw, eps=eps)
    power = amp ** 2

    exceed = (amp > T).astype(np.float32)
    excess = np.maximum(amp - T, 0.0).astype(np.float32)
    rel_excess = excess / (T + eps)
    excess_power = rel_excess ** 2

    seq_features = np.stack(
        [
            amp,
            power,
            exceed,
            excess,
            rel_excess,
            excess_power,
        ],
        axis=-1,
    ).astype(np.float32)

    par = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)
    peak_ratio = np.mean(exceed, axis=1)
    mean_excess = np.mean(excess, axis=1)
    max_excess = np.max(excess, axis=1)
    amp_std = np.std(amp, axis=1)
    amp_p95 = np.percentile(amp, 95, axis=1)
    amp_p99 = np.percentile(amp, 99, axis=1)

    global_features = np.stack(
        [
            np.log1p(par),
            peak_ratio,
            mean_excess,
            max_excess,
            amp_std,
            amp_p95,
            amp_p99,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(seq_features).all():
        raise ValueError("Amplitude static-peak sequence features contain NaN or infinite values.")
    if not np.isfinite(global_features).all():
        raise ValueError("Amplitude static-peak global features contain NaN or infinite values.")

    return seq_features, global_features


def extract_amplitude_focus_features(X_raw: np.ndarray,
                                     qam16_boundary: float,
                                     eps: float = 1e-8
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the focused amplitude/PAR feature set for QAM16/QAM64 separation.

    This is the most compact professor-facing QAM physical branch.  It avoids
    adjacent-sample transition features because those can be dominated by noise
    at low SNR, and keeps only direct radius/outer-boundary evidence.

    Per-time-step channels, shape (N, T, 4)
    --------------------------------------
    0. A(t)                 : mean-normalized amplitude / radius
    1. A(t)^2               : normalized instantaneous power
    2. exceed_T(t)          : 1 if A(t) > T_qam16 else 0
    3. excess_T(t)          : max(A(t) - T_qam16, 0)

    Global channels, shape (N, 2)
    ----------------------------
    0. log1p(PAR)           : log(1 + max(power) / mean(power))
    1. amplitude_p95        : 95th percentile of A(t)

    The QAM16 boundary T_qam16 must be estimated from the training split only
    by estimate_qam16_amplitude_boundary() to avoid validation/test leakage.
    """
    T = float(qam16_boundary)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"qam16_boundary must be a positive finite scalar; got {T!r}.")

    amp = _normalized_amplitude(X_raw, eps=eps)
    power = amp ** 2
    exceed = (amp > T).astype(np.float32)
    excess = np.maximum(amp - T, 0.0).astype(np.float32)

    seq_features = np.stack(
        [
            amp,
            power,
            exceed,
            excess,
        ],
        axis=-1,
    ).astype(np.float32)

    par = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)
    amp_p95 = np.percentile(amp, 95, axis=1)

    global_features = np.stack(
        [
            np.log1p(par),
            amp_p95,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(seq_features).all():
        raise ValueError("Focused amplitude sequence features contain NaN or infinite values.")
    if not np.isfinite(global_features).all():
        raise ValueError("Focused amplitude global features contain NaN or infinite values.")

    return seq_features, global_features


def extract_iq_amplitude_phase_channels(X_raw: np.ndarray,
                                        eps: float = 1e-8) -> np.ndarray:
    """
    Build the requested 4-channel sequence input: [I, Q, A(t), phase(t)].

    Channels, shape (N, T, 4)
    ------------------------
    0. I(t)       : in-phase component from the normalized IQ window
    1. Q(t)       : quadrature component from the normalized IQ window
    2. A(t)       : per-window mean-normalized amplitude/radius
    3. phase(t)   : atan2(Q, I) / pi, bounded to [-1, 1]

    Notes
    -----
    The phase scalar is intentionally bounded so the neural network receives
    a numerically stable feature.  I and Q are kept alongside phase, so even
    when phase wraps at -pi/pi, the original circular information is still
    available to the model through the raw IQ channels.
    """
    X = _validate_iq_array(X_raw)
    I = X[:, 0, :]
    Q = X[:, 1, :]
    amp = _normalized_amplitude(X, eps=eps)
    phase = (np.arctan2(Q, I) / np.pi).astype(np.float32)

    features = np.stack(
        [
            I,
            Q,
            amp,
            phase,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("IQ-amplitude-phase features contain NaN or infinite values.")
    return features


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
