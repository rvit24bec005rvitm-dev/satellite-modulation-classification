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
    psd = (np.abs(S) ** 2) / (n * fs)
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
    return combined / np.sqrt(np.mean(np.sum(combined**2, axis=1)))


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
    "32APSK": np.array([[r * np.cos(2*np.pi*k/n + offset), r * np.sin(2*np.pi*k/n + offset)]
                         for n, r, offset in [(4, 1., 0.), (12, 2.53, np.pi/12), (16, 4.3, 0.)]
                         for k in range(n)], dtype=np.float32),
    # Continuous-phase FSK schemes do not have a discrete symbol constellation.
    "GMSK": np.empty((0, 2), dtype=np.float32),
    "BFSK": np.empty((0, 2), dtype=np.float32),
}
CONSTELLATION_REFS["32APSK"] /= np.sqrt(np.mean(np.sum(CONSTELLATION_REFS["32APSK"]**2, axis=1)))
CONSTELLATION_REFS["32-APSK"] = CONSTELLATION_REFS["32APSK"]



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
    rrc_enabled: bool = True,
    rician_enabled: bool = False,
    twta_drive: float = 1.0,
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
    if num_samples < 16 or num_samples > 65536 or not 1 <= sps <= 32:
        raise ValueError("Invalid signal length or samples per symbol")
    if not 0 < rrc_alpha <= 1 or not np.isfinite(snr_db):
        raise ValueError("Invalid roll-off or SNR")
    rng = np.random.default_rng(rng_seed)
    num_symbols = int(np.ceil(num_samples / sps)) + 16  # extra for filter settling

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

    elif mod in ("QAM16", "16QAM"):
        levels = np.array([-3, -1, 1, 3])
        i_sym = rng.choice(levels, size=num_symbols)
        q_sym = rng.choice(levels, size=num_symbols)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(10)

    elif mod in ("16APSK", "32APSK"):
        refs = CONSTELLATION_REFS[mod]
        chosen = refs[rng.integers(0, len(refs), size=num_symbols)]
        symbols = chosen[:, 0] + 1j * chosen[:, 1]

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    if mod in ("GMSK", "BFSK"):
        # CPFSK: h=0.5, deviation ±Rs/4. GMSK smooths the NRZ
        # frequency pulse with a Gaussian lowpass (BT=0.3).
        nrz = np.repeat(symbols.real, sps)
        if mod == "GMSK":
            t = np.arange(-4*sps, 4*sps+1) / sps
            g = np.exp(-2 * (np.pi * 0.3 * t)**2 / np.log(2))
            nrz = np.convolve(nrz, g/g.sum(), mode="same")
        shaped = np.exp(1j * np.cumsum(np.pi * nrz / (2*sps)))
    elif rrc_enabled:
        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols
        h = _rrc_filter(num_taps=8*sps+1, sps=sps, alpha=rrc_alpha)
        shaped = np.convolve(upsampled, h, mode="same")
    else:
        shaped = np.repeat(symbols, sps)
    # Discard the initial pulse-shaping transient.
    shaped = shaped[8*sps:8*sps + num_samples]

    # ── 4. TWTA Saleh AM/AM distortion ───────────────────────────────
    if twta_alpha_a > 0.0:
        shaped_iq = np.column_stack((shaped.real, shaped.imag))
        distorted = saleh_twta(shaped_iq * twta_drive, alpha_a=twta_alpha_a)
        shaped = distorted[:, 0] + 1j*distorted[:, 1]

    # ── 5. Rician fading ─────────────────────────────────────────────
    if rician_enabled or rician_k > 0.0:
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

def extract_features(iq: np.ndarray) -> dict[str, float | None]:
    """Extract interpretable signal features for display / ML baselines."""
    from scipy.stats import kurtosis, skew

    I, Q = iq[:, 0], iq[:, 1]
    s = I + 1j * Q
    A = np.abs(s)
    phi = np.angle(s)

    def moment(function, values):
        return float(function(values)) if np.std(values) > 1e-15 else None

    feats = {
        "I_Mean": float(np.mean(I)),
        "Q_Mean": float(np.mean(Q)),
        "I_Std": float(np.std(I)),
        "Q_Std": float(np.std(Q)),
        "Amplitude_Mean": float(np.mean(A)),
        "Amplitude_Std": float(np.std(A)),
        "Phase_Std_rad": float(np.std(phi)),
        "Kurtosis_I": moment(kurtosis, I),
        "Kurtosis_Q": moment(kurtosis, Q),
        "Skewness_I": moment(skew, I),
        "Skewness_Q": moment(skew, Q),
        "Signal_Power_dBW": float(power_dbw(iq)),
        "PAPR_dB": float(10 * np.log10(np.max(A ** 2) / np.mean(A ** 2))) if np.mean(A ** 2) > 0 else None,
    }
    return {k: v if v is not None and np.isfinite(v) else None for k, v in feats.items()}


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


# ──────────────────────────────────────────────────────────────────────────────
# Audible RF Baseband Heterodyne Synthesizer
# ──────────────────────────────────────────────────────────────────────────────

def generate_audio_waveform(
    iq: np.ndarray,
    carrier_tone_hz: float = 880.0,
    sample_rate: int = 16000,
    duration_sec: float = 1.5,
) -> bytes:
    """
    Synthesize an audible radio tone from raw baseband I/Q samples via heterodyne demodulation:
      s_audio(t) = I(t)*cos(2*pi*fc*t) - Q(t)*sin(2*pi*fc*t)
    Returns WAV bytes suitable for st.audio().
    """
    import io
    from scipy.io import wavfile

    num_samples = len(iq)
    total_audio_samples = int(sample_rate * duration_sec)

    # Repeat or tile the I/Q samples to span the desired audio duration
    repeats = int(np.ceil(total_audio_samples / num_samples))
    tiled_i = np.tile(iq[:, 0], repeats)[:total_audio_samples]
    tiled_q = np.tile(iq[:, 1], repeats)[:total_audio_samples]

    # Smooth the transitions to avoid abrupt jumps
    t = np.linspace(0, duration_sec, total_audio_samples, endpoint=False)
    
    # Heterodyne mixing with audible carrier tone
    audio_signal = tiled_i * np.cos(2 * np.pi * carrier_tone_hz * t) - tiled_q * np.sin(2 * np.pi * carrier_tone_hz * t)

    # Apply 50ms smooth window attack and decay envelope
    fade_len = int(sample_rate * 0.05)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio_signal[:fade_len] *= fade_in
    audio_signal[-fade_len:] *= fade_out

    # Normalize to 16-bit PCM range
    max_val = np.max(np.abs(audio_signal)) + 1e-12
    pcm_audio = (audio_signal / max_val * 32767 * 0.85).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm_audio)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Eye Diagram Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_eye_diagram(iq: np.ndarray, sps: int = 8, traces_to_keep: int = 40) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Extract overlapping 2-symbol trace segments to construct In-Phase and Quadrature Eye Diagrams.
    Returns (time_axis, i_traces, q_traces).
    """
    span = 2 * sps
    i_channel = iq[:, 0]
    q_channel = iq[:, 1]
    
    # Interpolate for ultra-smooth eye curves
    oversample = 4
    interp_len = len(iq) * oversample
    i_up = np.interp(np.arange(interp_len) / oversample, np.arange(len(iq)), i_channel)
    q_up = np.interp(np.arange(interp_len) / oversample, np.arange(len(iq)), q_channel)

    span_up = span * oversample
    num_traces = min(traces_to_keep, 1 + (interp_len - span_up) // (sps * oversample))
    
    t_axis = np.arange(span_up) / (sps * oversample) - 1.0
    i_traces = []
    q_traces = []

    for idx in range(max(1, num_traces)):
        start = idx * sps * oversample
        end = start + span_up
        if end <= len(i_up):
            i_traces.append(i_up[start:end])
            q_traces.append(q_up[start:end])

    return t_axis, i_traces, q_traces


# ──────────────────────────────────────────────────────────────────────────────
# LEO Satellite Orbital Pass Simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_orbit_pass(
    altitude_km: float = 550.0,
    carrier_freq_ghz: float = 29.5,
    max_elevation_deg: float = 78.0,
) -> dict[str, np.ndarray | float | str]:
    """
    Simulates a 10-minute LEO ground station overflight (e.g. Starlink or Earth Observation satellite).
    Computes elevation, distance, Doppler frequency curve, and orbital parameters.
    """
    # Earth physical constants
    R_E = 6378.137  # Earth radius in km
    GM = 398600.4418  # Earth gravitational constant km^3 / s^2
    c_kms = 299792.458  # Speed of light in km/s

    # Orbital mechanics
    r_orbit = R_E + altitude_km
    v_orbit = np.sqrt(GM / r_orbit)  # ~7.6 km/s for LEO
    period_min = 2 * np.pi * np.sqrt(r_orbit**3 / GM) / 60.0

    # 10 minute pass window (-300s to +300s)
    t = np.linspace(-300, 300, 201)

    # Angular distance traveled along ground track
    omega_orbit = v_orbit / r_orbit
    theta = omega_orbit * t  # radians

    # Distance to ground station (approximation based on max elevation geometry)
    min_dist = altitude_km / np.sin(np.radians(max_elevation_deg))
    dist = np.sqrt(min_dist**2 + (v_orbit * t)**2)

    # Line of sight velocity: v_los = d(dist)/dt
    v_los = (v_orbit**2 * t) / np.maximum(dist, 1e-6)

    # Doppler shift: df = - (v_los / c) * fc
    carrier_hz = carrier_freq_ghz * 1e9
    doppler_khz = -(v_los / c_kms) * (carrier_hz / 1e3)

    # Elevation angle (degrees)
    elev_deg = np.degrees(np.arctan2(altitude_km, np.abs(v_orbit * t) + 1e-6))
    elev_deg = np.clip(elev_deg * (max_elevation_deg / 90.0), 0.0, 90.0)

    # Simulated ground track (Latitude / Longitude trace passing over ground station)
    ground_station = {"name": "Svalbard Satellite Station (SGS)", "lat": 78.229, "lon": 15.407}
    track_lats = ground_station["lat"] + (t / 300.0) * 12.0
    track_lons = ground_station["lon"] - (t / 300.0) * 8.0

    return {
        "time_sec": t,
        "elevation_deg": elev_deg,
        "doppler_khz": doppler_khz,
        "range_km": dist,
        "v_orbit_kms": float(v_orbit),
        "period_min": float(period_min),
        "max_doppler_khz": float(np.max(np.abs(doppler_khz))),
        "station": ground_station,
        "track_lats": track_lats,
        "track_lons": track_lons,
    }

