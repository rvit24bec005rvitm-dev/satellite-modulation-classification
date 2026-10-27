"""
utils/signal_processing.py
Shared helpers for signal processing: FFT, constellation, channel impairments,
synthetic modulation signal generation.
All pure NumPy / SciPy – no deep-learning dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_signal(iq: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score normalise an I/Q array.  std=0 is protected against."""
    safe_std = std if std != 0.0 else 1.0
    return (iq - mean) / safe_std


# ──────────────────────────────────────────────────────────────────────────────
# Instantaneous signal attributes
# ──────────────────────────────────────────────────────────────────────────────

def amplitude(iq: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude  |A[n]| = sqrt(I² + Q²).  Shape: (N,)"""
    return np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)


def phase_rad(iq: np.ndarray) -> np.ndarray:
    """Instantaneous phase in radians.  Shape: (N,)"""
    return np.arctan2(iq[:, 1], iq[:, 0])


def power_dbw(iq: np.ndarray) -> float:
    """Mean signal power in dBW."""
    pwr = np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2)
    return 10.0 * np.log10(pwr + 1e-30)


# ──────────────────────────────────────────────────────────────────────────────
# Power Spectral Density (FFT-based)
# ──────────────────────────────────────────────────────────────────────────────

def compute_psd(iq: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Centred two-sided PSD of the complex signal s = I + jQ.
    Returns (frequencies, psd_db) both shape (N,).
    """
    s = iq[:, 0] + 1j * iq[:, 1]
    n = len(s)
    S = np.fft.fftshift(np.fft.fft(s))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    psd = (np.abs(S) ** 2) / n
    psd_db = 10.0 * np.log10(psd + 1e-30)
    return freqs, psd_db


# ──────────────────────────────────────────────────────────────────────────────
# Spectrogram
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(
    iq: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT spectrogram of complex I/Q.
    Returns (frequencies, times, Zxx_db).
    """
    s = iq[:, 0] + 1j * iq[:, 1]
    freqs, times, Zxx = sp_signal.stft(
        s, fs=fs, nperseg=nperseg, return_onesided=False
    )
    freqs = np.fft.fftshift(freqs)
    Zxx_db = 10.0 * np.log10(np.abs(np.fft.fftshift(Zxx, axes=0)) ** 2 + 1e-30)
    return freqs, times, Zxx_db


def _apsk_refs(n1: int, n2: int, r1: float, r2: float) -> np.ndarray:
    """Generate 2-ring APSK reference constellation."""
    ring1 = np.array(
        [[r1 * np.cos(2 * np.pi * k / n1), r1 * np.sin(2 * np.pi * k / n1)]
         for k in range(n1)],
        dtype=np.float32,
    )
    ring2 = np.array(
        [[r2 * np.cos(2 * np.pi * k / n2 + np.pi / n2),
          r2 * np.sin(2 * np.pi * k / n2 + np.pi / n2)]
         for k in range(n2)],
        dtype=np.float32,
    )
    combined = np.vstack([ring1, ring2])
    return combined / np.max(np.abs(combined))


# ──────────────────────────────────────────────────────────────────────────────
# Theoretical constellation reference points
# ──────────────────────────────────────────────────────────────────────────────

CONSTELLATION_REFS: dict[str, np.ndarray] = {
    "BPSK": np.array([[-1, 0], [1, 0]], dtype=np.float32),
    "QPSK": np.array(
        [[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32
    ) / np.sqrt(2),
    "8PSK": np.array(
        [[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)],
        dtype=np.float32,
    ),
    "8-PSK": np.array(
        [[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)],
        dtype=np.float32,
    ),
    "QAM16": np.array(
        [
            [i, q]
            for i in [-3, -1, 1, 3]
            for q in [-3, -1, 1, 3]
        ],
        dtype=np.float32,
    ) / np.sqrt(10),
    "16-QAM": np.array(
        [
            [i, q]
            for i in [-3, -1, 1, 3]
            for q in [-3, -1, 1, 3]
        ],
        dtype=np.float32,
    ) / np.sqrt(10),
    "16APSK": _apsk_refs(4, 12, r1=1.0, r2=2.53),
    "16-APSK": _apsk_refs(4, 12, r1=1.0, r2=2.53),
    "32APSK": _apsk_refs(4, 12, r1=1.0, r2=2.53),  # simplified 2-ring
    "32-APSK": _apsk_refs(4, 12, r1=1.0, r2=2.53),
    "GMSK": np.array([[-1, 0], [1, 0]], dtype=np.float32),  # approximate BPSK-like
    "BFSK": np.array([[-1, 0], [1, 0]], dtype=np.float32),
}


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic I/Q signal generation
# ──────────────────────────────────────────────────────────────────────────────

def _rrc_filter(num_taps: int, sps: int, alpha: float) -> np.ndarray:
    """Root-Raised Cosine FIR filter coefficients."""
    t = (np.arange(num_taps) - num_taps // 2) / sps
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1.0 - alpha + 4.0 * alpha / np.pi)
        elif abs(ti) == 1.0 / (4.0 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1.0 - alpha))
                + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            ) / (np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2))
    h /= np.sqrt(np.sum(h ** 2))
    return h


def generate_iq_signal(
    modulation: str,
    num_samples: int = 128,
    sps: int = 8,
    snr_db: float = 10.0,
    rrc_alpha: float = 0.35,
    doppler_hz: float = 0.0,
    cfo_hz: float = 0.0,
    rician_k: float = 0.0,
    twta_alpha_a: float = 0.0,
    phase_noise_std: float = 0.0,
    rng_seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic baseband I/Q signal with optional channel impairments.

    Parameters
    ----------
    modulation : one of BPSK | QPSK | 8PSK | QAM16 | 16APSK | 32APSK | GMSK | BFSK
    num_samples : number of output time samples (default 128 to match CNN input)
    sps         : samples per symbol (used for RRC shaping)
    snr_db      : additive white Gaussian noise level in dB
    rrc_alpha   : roll-off factor for Root-Raised Cosine pulse shaping
    doppler_hz  : normalised Doppler frequency (0–0.5) simulating LEO satellite motion
    cfo_hz      : normalised carrier frequency offset
    rician_k    : Rician K-factor (0 = Rayleigh, >10 ≈ AWGN/LOS channel)
    twta_alpha_a: Saleh AM/AM gain compression (0 = linear, 0.5 typical TWTA)
    phase_noise_std: standard deviation of additive phase noise (radians)
    rng_seed    : optional random seed for reproducibility

    Returns
    -------
    iq : np.ndarray, shape (num_samples, 2)  I=[:,0], Q=[:,1]
    """
    rng = np.random.default_rng(rng_seed)
    num_symbols = (num_samples // sps) + 4  # extra for filter settling

    mod = modulation.upper().replace("-", "").replace("_", "")

    # ── 1. Symbol generation ─────────────────────────────────────────
    if mod in ("BPSK", "GMSK", "BFSK"):
        bits = rng.integers(0, 2, size=num_symbols)
        symbols = (2 * bits - 1).astype(complex)

    elif mod == "QPSK":
        bits = rng.integers(0, 2, size=(num_symbols, 2))
        symbols = ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)

    elif mod == "8PSK":
        idx = rng.integers(0, 8, size=num_symbols)
        angles = idx * np.pi / 4
        symbols = np.exp(1j * angles)

    elif mod == "QAM16":
        levels = np.array([-3, -1, 1, 3])
        i_sym = rng.choice(levels, size=num_symbols)
        q_sym = rng.choice(levels, size=num_symbols)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(10)

    elif mod == "16APSK":
        ring1_n, ring2_n = 4, 12
        ring1_r, ring2_r = 1.0, 2.53
        total = ring1_n + ring2_n
        idx = rng.integers(0, total, size=num_symbols)
        symbols = np.where(
            idx < ring1_n,
            ring1_r * np.exp(1j * 2 * np.pi * idx / ring1_n),
            ring2_r * np.exp(1j * (2 * np.pi * (idx - ring1_n) / ring2_n + np.pi / ring2_n)),
        ).astype(complex) / ring2_r

    elif mod == "32APSK":
        ring1_n, ring2_n, ring3_n = 4, 12, 16
        ring1_r, ring2_r, ring3_r = 1.0, 2.53, 4.30
        total = ring1_n + ring2_n + ring3_n
        idx = rng.integers(0, total, size=num_symbols)
        s = np.zeros(num_symbols, dtype=complex)
        m1 = idx < ring1_n
        m2 = (idx >= ring1_n) & (idx < ring1_n + ring2_n)
        m3 = idx >= ring1_n + ring2_n
        s[m1] = ring1_r * np.exp(1j * 2 * np.pi * idx[m1] / ring1_n)
        s[m2] = ring2_r * np.exp(1j * (2 * np.pi * (idx[m2] - ring1_n) / ring2_n + np.pi / ring2_n))
        s[m3] = ring3_r * np.exp(1j * (2 * np.pi * (idx[m3] - ring1_n - ring2_n) / ring3_n))
        symbols = s / ring3_r

    else:
        # Fallback: BPSK
        bits = rng.integers(0, 2, size=num_symbols)
        symbols = (2 * bits - 1).astype(complex)

    # ── 2. Upsample ──────────────────────────────────────────────────
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    # ── 3. RRC Pulse shaping ─────────────────────────────────────────
    h = _rrc_filter(num_taps=min(64, len(upsampled) - 1), sps=sps, alpha=rrc_alpha)
    shaped = np.convolve(upsampled, h, mode="same")

    # Trim to num_samples
    shaped = shaped[:num_samples]
    if len(shaped) < num_samples:
        shaped = np.pad(shaped, (0, num_samples - len(shaped)))

    # ── 4. TWTA Saleh AM/AM distortion ───────────────────────────────
    if twta_alpha_a > 0.0:
        alpha_a, beta_a = twta_alpha_a, 0.1
        r = np.abs(shaped)
        gain = alpha_a / (1.0 + beta_a * r ** 2)
        shaped = gain * r * np.exp(1j * np.angle(shaped))

    # ── 5. Rician fading ─────────────────────────────────────────────
    if rician_k > 0.0:
        los = np.sqrt(rician_k / (rician_k + 1.0))
        scatter_std = np.sqrt(0.5 / (rician_k + 1.0))
        scatter = scatter_std * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))
        channel = los + scatter
        shaped = shaped * channel

    # ── 6. Phase noise ────────────────────────────────────────────────
    if phase_noise_std > 0.0:
        pn = rng.normal(0, phase_noise_std, num_samples)
        shaped = shaped * np.exp(1j * pn)

    # ── 7. Doppler + CFO ─────────────────────────────────────────────
    total_freq_offset = doppler_hz + cfo_hz
    if total_freq_offset != 0.0:
        n = np.arange(num_samples)
        shaped = shaped * np.exp(1j * 2.0 * np.pi * total_freq_offset * n)

    # ── 8. AWGN ──────────────────────────────────────────────────────
    sig_power = np.mean(np.abs(shaped) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0) + 1e-30)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    )
    noisy = shaped + noise

    # Normalise to unit power
    final_power = np.mean(np.abs(noisy) ** 2)
    if final_power > 0:
        noisy /= np.sqrt(final_power)

    iq = np.stack([noisy.real, noisy.imag], axis=-1).astype(np.float32)
    return iq


# ──────────────────────────────────────────────────────────────────────────────
# Classification features (expert features for ML baseline)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(iq: np.ndarray) -> dict[str, float]:
    """Extract interpretable signal features for display / ML baselines."""
    from scipy.stats import kurtosis, skew

    I, Q = iq[:, 0], iq[:, 1]
    s = I + 1j * Q
    A = np.abs(s)
    phi = np.angle(s)

    feats = {
        "I_Mean": float(np.mean(I)),
        "Q_Mean": float(np.mean(Q)),
        "I_Std": float(np.std(I)),
        "Q_Std": float(np.std(Q)),
        "Amplitude_Mean": float(np.mean(A)),
        "Amplitude_Std": float(np.std(A)),
        "Phase_Std_rad": float(np.std(phi)),
        "Kurtosis_I": float(kurtosis(I)),
        "Kurtosis_Q": float(kurtosis(Q)),
        "Skewness_I": float(skew(I)),
        "Skewness_Q": float(skew(Q)),
        "Signal_Power_dBW": float(power_dbw(iq)),
        "PAPR_dB": float(10 * np.log10(np.max(A ** 2) / (np.mean(A ** 2) + 1e-30))),
    }
    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Shannon Entropy (for uncertainty / XAI)
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray) -> float:
    """H(p) = -sum p_i * log2(p_i).  Input: probability vector."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


# ──────────────────────────────────────────────────────────────────────────────
# Saleh TWTA Amplifier Non-Linearity
# ──────────────────────────────────────────────────────────────────────────────

def saleh_twta(
    iq: np.ndarray,
    alpha_a: float = 2.1587,
    beta_a: float = 1.1517,
    alpha_phi: float = 4.0033,
    beta_phi: float = 9.1040,
) -> np.ndarray:
    """
    Apply Saleh model TWTA non-linear AM/AM and AM/PM distortion to an I/Q signal.
    """
    I, Q = iq[..., 0], iq[..., 1]
    r = np.sqrt(I**2 + Q**2)
    phi = np.arctan2(Q, I)

    # Saleh AM/AM and AM/PM formulas
    A_r = (alpha_a * r) / (1.0 + beta_a * r**2 + 1e-12)
    Phi_r = (alpha_phi * r**2) / (1.0 + beta_phi * r**2 + 1e-12)

    new_phi = phi + Phi_r
    out_I = A_r * np.cos(new_phi)
    out_Q = A_r * np.sin(new_phi)
    return np.stack([out_I, out_Q], axis=-1).astype(np.float32)

