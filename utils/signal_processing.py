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
    """Mean power in dB relative to squared sample units (legacy function name)."""
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


from utils.modulation import CONSTELLATION_REFS, constellation, canonical_modulation, map_bits, frequency_modulate


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


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(snr_db) or not -100 <= snr_db <= 300:
        raise ValueError('SNR must be finite and between -100 and 300 dB.')
    power = float(np.mean(np.abs(signal)**2))
    noise = np.sqrt(power * 10**(-snr_db/10) / 2) * (rng.standard_normal(len(signal)) + 1j*rng.standard_normal(len(signal)))
    return signal + noise, noise


def generate_iq_signal(
    modulation: str, num_samples: int = 128, sps: int = 8, snr_db: float = 10., rrc_alpha: float = .35,
    doppler_hz: float = 0., cfo_hz: float = 0., rician_k: float = 0., twta_alpha_a: float = 0.,
    phase_noise_std: float = 0., rng_seed: int | None = None, rrc_enabled: bool = True,
    rician_enabled: bool | None = None, twta_drive: float = 1., sample_rate: float = 1.,
    gaussian_bt: float = .3, bfsk_deviation_hz: float | None = None, continuous_phase: bool = True,
    apsk_radii=None, apsk_offsets=None, phase_noise_mode: str = 'iid', fading_block_samples: int = 1,
    normalize_output: bool = True, return_metadata: bool = False,
):
    """Seeded baseband transmitter → TWTA → flat fading → phase error → CFO/Doppler → AWGN.

    Frequencies use Hz (cycles/sample when sample_rate=1). Phase noise is either
    IID phase jitter or a Wiener phase walk with the configured per-sample increment σ.
    Output normalization is explicit; pre-normalization power stages are returned
    on request so RMS calibration cannot be mistaken for conserved amplifier gain.
    """
    if not isinstance(num_samples, (int, np.integer)) or not 16 <= num_samples <= 65536 or not isinstance(sps, (int, np.integer)) or not 1 <= sps <= 32:
        raise ValueError('Invalid signal length or samples per symbol.')
    values = [rrc_alpha, sample_rate, doppler_hz, cfo_hz, rician_k, phase_noise_std, twta_drive, twta_alpha_a]
    if not np.isfinite(values).all() or not 0 < rrc_alpha <= 1 or sample_rate <= 0 or min(rician_k, phase_noise_std, twta_drive, twta_alpha_a) < 0:
        raise ValueError('Invalid channel parameters, roll-off or sample rate.')
    if phase_noise_mode not in ['iid', 'wiener'] or not isinstance(fading_block_samples, (int, np.integer)) or fading_block_samples < 1:
        raise ValueError('Invalid phase-noise mode or fading block length.')
    offset = doppler_hz + cfo_hz
    if abs(offset) > sample_rate/2: raise ValueError('Combined carrier offset exceeds Nyquist.')
    mod = canonical_modulation(modulation)
    refs = constellation(mod, apsk_radii, apsk_offsets)
    if mod in ['GMSK', 'BFSK']:
        deviation = sample_rate/(4*sps) if mod == 'GMSK' or bfsk_deviation_hz is None else bfsk_deviation_hz
        if abs(offset) + deviation >= sample_rate/2: raise ValueError('FSK frequency states plus carrier offset exceed Nyquist.')
    # Independent streams make controlled channel comparisons reproducible.
    symbol_seed, fading_seed, phase_seed, noise_seed = np.random.SeedSequence(rng_seed).spawn(4)
    rng = np.random.default_rng(symbol_seed)
    guard_symbols = max(8, int(np.ceil(.8/gaussian_bt))) if mod == 'GMSK' else 8
    symbols_count = int(np.ceil(num_samples/sps)) + 2*guard_symbols
    if mod in ['GMSK', 'BFSK']:
        bits = rng.integers(0, 2, size=symbols_count)
        shaped = frequency_modulate(bits, sps, sample_rate, deviation, gaussian_bt if mod == 'GMSK' else None, continuous_phase)
    else:
        width = int(np.log2(len(refs)))
        bits = rng.integers(0, 2, size=symbols_count * width)
        symbols = map_bits(bits, mod, apsk_radii, apsk_offsets)
        if rrc_enabled:
            up = np.zeros(len(symbols)*sps, dtype=complex); up[::sps] = symbols
            shaped = np.convolve(up, _rrc_filter(8*sps+1, sps, rrc_alpha), mode='same')
        else: shaped = np.repeat(symbols, sps)
    shaped = shaped[guard_symbols*sps:guard_symbols*sps+num_samples]
    tx_gain = 1/np.sqrt(np.mean(np.abs(shaped)**2))
    shaped *= tx_gain  # transmitter RMS calibration, before the channel
    power = lambda z: float(np.mean(np.abs(z)**2))
    stages = {'transmitter': power(shaped)}
    if twta_alpha_a > 0 and twta_drive > 0:
        distorted = saleh_twta(np.column_stack((shaped.real, shaped.imag))*twta_drive, alpha_a=twta_alpha_a)
        shaped = distorted[:, 0].astype(float) + 1j*distorted[:, 1]
    stages['afterTwta'] = power(shaped)
    fading = rician_k > 0 if rician_enabled is None else rician_enabled
    if fading:
        rng_fade = np.random.default_rng(fading_seed)
        blocks = int(np.ceil(num_samples/fading_block_samples))
        scatter = (rng_fade.standard_normal(blocks) + 1j*rng_fade.standard_normal(blocks))/np.sqrt(2*(rician_k+1))
        channel = np.repeat(np.sqrt(rician_k/(rician_k+1)) + scatter, fading_block_samples)[:num_samples]
        shaped *= channel
    stages['afterFading'] = power(shaped)
    if phase_noise_std > 0:
        angles = np.random.default_rng(phase_seed).normal(0, phase_noise_std, num_samples)
        if phase_noise_mode == 'wiener': angles = np.cumsum(angles)
        shaped *= np.exp(1j*angles)
    shaped *= np.exp(2j*np.pi*offset*np.arange(num_samples)/sample_rate)
    noisy, noise = add_awgn(shaped, snr_db, np.random.default_rng(noise_seed))
    stages['beforeOutputScale'] = power(noisy)
    gain = 1/np.sqrt(stages['beforeOutputScale']) if normalize_output and stages['beforeOutputScale'] > 0 else 1.
    noisy *= gain
    iq = np.column_stack((noisy.real, noisy.imag)).astype(np.float32)
    metadata = {'channelOrder': ['transmitter', 'TWTA', 'flat Rician/Rayleigh fading', 'phase noise', 'CFO + Doppler', 'AWGN', 'output amplitude policy'],
                'pulseShaping':'Gaussian continuous-phase' if mod=='GMSK' else 'BFSK frequency states' if mod=='BFSK' else 'RRC' if rrc_enabled else 'rectangular symbol hold',
                'rrcApplied':bool(rrc_enabled and len(refs)), 'fadingApplied':bool(fading), 'twtaApplied':bool(twta_alpha_a>0 and twta_drive>0),
                'powerStages': stages, 'outputGain': float(gain), 'transmitterGain': float(tx_gain),
                'outputPowerPolicy': 'unit-rms' if normalize_output else 'preserve-channel-gain',
                'requestedSnrDb': snr_db, 'realizedSnrDb': float(10*np.log10(max(power(shaped), 1e-300)/max(power(noise), 1e-300))),
                'symbolRate': sample_rate/sps, 'filterGroupDelaySamples': (max(4,int(np.ceil(.8/gaussian_bt)))*sps if mod == 'GMSK' else 4*sps if rrc_enabled and len(refs) else 0),
                'filterDelayPolicy':'Centered FIR convolution with guarded source symbols; nominal causal delay is compensated.',
                'phaseNoiseUnits':'radians per sample increment' if phase_noise_mode == 'wiener' else 'radians (independent jitter)',
                'discardedGuardSamples': guard_symbols*sps, 'mapping': 'Natural-binary index, MSB first; educational APSK, rectangular 8×4 32-QAM; no DVB-S2 claim',
                'fadingModel': f'Independent flat fading per {fading_block_samples} samples' if fading else 'bypassed',
                'referenceNote': 'Transmitter symbol table. Pulse shaping and channel effects move received samples away from ideal symbols.',
                'reference': (refs * (tx_gain * gain if not rrc_enabled else gain)).tolist()}
    if mod in ['GMSK', 'BFSK']:
        metadata.update(frequencyDeviationHz=deviation, modulationIndex=2*deviation*sps/sample_rate,
                        phaseContinuous=mod == 'GMSK' or continuous_phase, gaussianBt=gaussian_bt if mod == 'GMSK' else None)
    return (iq, metadata) if return_metadata else iq


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
        scale = np.max(np.abs(values))
        if scale == 0: return None
        normalized = values / scale
        return float(function(normalized)) if np.std(normalized) > 8*np.finfo(float).eps else None

    feats = {
        "I_Mean": float(np.mean(I)),
        "Q_Mean": float(np.mean(Q)),
        "I_Std": float(np.std(I)),
        "Q_Std": float(np.std(Q)),
        "Amplitude_Mean": float(np.mean(A)),
        "Amplitude_Std": float(np.std(A)),
        "Phase_Std_rad": float(np.std(phi[A > 0])) if np.any(A > 0) else None,
        "Kurtosis_I": moment(kurtosis, I),
        "Kurtosis_Q": moment(kurtosis, Q),
        "Skewness_I": moment(skew, I),
        "Skewness_Q": moment(skew, Q),
        "Signal_Power_dB_relative": float(power_dbw(iq)) if np.mean(A**2) > 0 else None,
        "I_Variance": float(np.var(I)),
        "Q_Variance": float(np.var(Q)),
        "Crest_Factor": float(np.max(A) / np.sqrt(np.mean(A**2))) if np.mean(A**2) > 0 else None,
        "PAPR_dB": float(10 * np.log10(np.max(A ** 2) / np.mean(A ** 2))) if np.mean(A ** 2) > 0 else None,
    }
    return {k: v if v is not None and np.isfinite(v) else None for k, v in feats.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Shannon Entropy (for uncertainty / XAI)
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray) -> float:
    """H(p) = -sum p_i * log2(p_i).  Input: probability vector."""
    p = np.asarray(probs,dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(),1,atol=1e-5):
        raise ValueError('Entropy requires a finite normalized probability vector.')
    positive = p[p > 0]
    return float(-np.sum(positive*np.log2(positive)))


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
) -> dict:
    """Spherical circular-orbit demonstration with a rotating Earth and one station.

    The closest-approach geometry fixes the requested peak elevation. Range rate,
    Doppler, elevation and geographic track all use the SAME propagated vectors.
    This is an educational two-body scenario, not a real spacecraft/TLE feed.
    """
    if not np.isfinite([altitude_km,carrier_freq_ghz,max_elevation_deg]).all() or altitude_km <= 0 or carrier_freq_ghz <= 0 or not 0 < max_elevation_deg <= 90:
        raise ValueError('Altitude/carrier must be positive and peak elevation must be in (0,90].')
    radius_earth,mu,c_kms=6378.137,398600.4418,299792.458
    orbital_radius=radius_earth+altitude_km
    omega=np.sqrt(mu/orbital_radius**3)
    earth_omega=2*np.pi/86164.0905
    station={'name':'Svalbard reference station (simulated pass)','lat':78.229,'lon':15.407}
    latitude,longitude=np.radians([station['lat'],station['lon']])
    up=np.array([np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)])
    north=np.array([-np.sin(latitude)*np.cos(longitude),-np.sin(latitude)*np.sin(longitude),np.cos(latitude)])
    east=np.array([-np.sin(longitude),np.cos(longitude),0.])
    elevation=np.radians(max_elevation_deg)
    closest=-radius_earth*np.sin(elevation)+np.sqrt(orbital_radius**2-radius_earth**2*np.cos(elevation)**2)
    initial=radius_earth*up+closest*(np.sin(elevation)*up+np.cos(elevation)*north)
    time=np.linspace(-300,300,201)
    position=np.cos(omega*time)[:,None]*initial+np.sin(omega*time)[:,None]*orbital_radius*east
    velocity=omega*(-np.sin(omega*time)[:,None]*initial+np.cos(omega*time)[:,None]*orbital_radius*east)
    def spin(v,angle):
        angle=np.asarray(angle)
        return np.column_stack((v[:,0]*np.cos(angle)-v[:,1]*np.sin(angle),v[:,0]*np.sin(angle)+v[:,1]*np.cos(angle),v[:,2]))
    ground=spin(np.tile(radius_earth*up,(len(time),1)),earth_omega*time)
    ground_velocity=earth_omega*np.column_stack((-ground[:,1],ground[:,0],np.zeros(len(time))))
    line=position-ground
    distance=np.linalg.norm(line,axis=1)
    range_rate=np.sum(line*(velocity-ground_velocity),axis=1)/distance
    angles=np.degrees(np.arcsin(np.clip(np.sum(line*ground,axis=1)/(distance*radius_earth),-1,1)))
    ecef=spin(position,-earth_omega*time)
    track_lat=np.degrees(np.arcsin(np.clip(ecef[:,2]/orbital_radius,-1,1)))
    track_lon=np.degrees(np.arctan2(ecef[:,1],ecef[:,0]))
    doppler=-range_rate/c_kms*carrier_freq_ghz*1e6
    return {'time_sec':time,'elevation_deg':angles,'doppler_khz':doppler,'range_km':distance,
            'range_rate_kms':range_rate,'visible':angles>=0,'v_orbit_kms':float(np.sqrt(mu/orbital_radius)),
            'period_min':float(2*np.pi/omega/60),'max_doppler_khz':float(np.max(abs(doppler))),
            'station':station,'track_lats':track_lat,'track_lons':track_lon,
            'model':'Simulated spherical two-body circular orbit; sidereal Earth rotation; no oblateness, perturbations or live ephemeris.'}
