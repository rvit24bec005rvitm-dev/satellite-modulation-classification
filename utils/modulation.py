"""Explicit educational modulation mappings; no DVB-S2/Gray-coding claim."""
from __future__ import annotations
import numpy as np


def canonical_modulation(name: str) -> str:
    value = name.upper().replace('-', '').replace('_', '')
    return {'16QAM': 'QAM16', '32QAM': 'QAM32'}.get(value, value)


def constellation(name: str, radii=None, offsets=None) -> np.ndarray:
    mod = canonical_modulation(name)
    if mod == 'BPSK': points = np.array([-1, 1], dtype=complex)
    elif mod == 'QPSK': points = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    elif mod == '8PSK': points = np.exp(2j * np.pi * np.arange(8) / 8)
    elif mod in ['QAM16', 'QAM32']:
        levels_i = np.arange(-3, 4, 2) if mod == 'QAM16' else np.arange(-7, 8, 2)
        points = np.array([i + 1j*q for i in levels_i for q in np.arange(-3, 4, 2)])
    elif mod in ['16APSK', '32APSK']:
        counts = [4, 12] if mod == '16APSK' else [4, 12, 16]
        radius = np.asarray(radii if radii is not None else ([1, 2.53] if mod == '16APSK' else [1, 2.53, 4.3]), dtype=float)
        angles = np.asarray(offsets if offsets is not None else ([0, np.pi/12] if mod == '16APSK' else [0, np.pi/12, 0]), dtype=float)
        if (radius.shape != (len(counts),) or angles.shape != (len(counts),) or
                not np.isfinite(radius).all() or not np.isfinite(angles).all() or radius[0] <= 0 or np.any(radius > 1000) or np.any(np.diff(radius) <= 0)):
            raise ValueError('APSK requires increasing positive ring radii and one finite angular offset (radians) per ring.')
        points = np.concatenate([r * np.exp(1j * (2*np.pi*np.arange(n)/n + a)) for n, r, a in zip(counts, radius, angles)])
    elif mod in ['GMSK', 'BFSK']: return np.empty((0, 2), dtype=np.float32)
    else: raise ValueError(f'Unsupported modulation: {name}')
    points /= np.sqrt(np.mean(np.abs(points)**2))
    result = np.column_stack((points.real, points.imag)).astype(np.float32)
    if len(np.unique(result,axis=0)) != len(result): raise ValueError('Constellation points collapse at float32 precision; choose better-separated ring radii.')
    return result


def map_bits(bits: np.ndarray, name: str, radii=None, offsets=None) -> np.ndarray:
    """Natural-binary index into the documented reference table, MSB first."""
    refs = constellation(name, radii, offsets)
    if not len(refs): raise ValueError('CPM bits must be frequency modulated, not mapped to a static constellation.')
    width = int(np.log2(len(refs)))
    a = np.asarray(bits)
    if a.ndim != 1 or not len(a) or len(a) % width or not np.isin(a, [0, 1]).all():
        raise ValueError(f'Expected a nonempty binary vector divisible by {width} bits per symbol.')
    indices = a.reshape(-1, width) @ (2**np.arange(width-1, -1, -1))
    chosen = refs[indices]
    return chosen[:, 0].astype(float) + 1j*chosen[:, 1]


def frequency_modulate(bits, sps: int, sample_rate=1., deviation_hz=None, bt: float | None = None,
                       continuous_phase=True, initial_phase=0.) -> np.ndarray:
    """BFSK or Gaussian-filtered CPFSK. GMSK always uses h=0.5, Δf=Rs/4.

    Gaussian impulse response has its power -3 dB bandwidth B=BT/T.
    Phase increments are 2πf/Fs; a sustained GMSK bit contributes ±π/2/symbol.
    """
    a = np.asarray(bits)
    if a.ndim != 1 or not len(a) or not np.isin(a, [0, 1]).all(): raise ValueError('Expected binary bits.')
    if not isinstance(sps, (int, np.integer)) or not 1 <= sps <= 32: raise ValueError('Invalid samples per symbol.')
    if not np.isfinite(sample_rate) or sample_rate <= 0: raise ValueError('Sample rate must be positive and finite.')
    frequency = np.repeat(2*a.astype(float)-1, sps)
    deviation = sample_rate/(4*sps) if deviation_hz is None else deviation_hz
    if not np.isfinite(deviation) or not 0 < deviation < sample_rate/2:
        raise ValueError('Frequency deviation must lie strictly between zero and Nyquist.')
    if bt is not None:
        if not .05 <= bt <= 1: raise ValueError('GMSK BT must be between 0.05 and 1.')
        if not np.isclose(deviation, sample_rate/(4*sps)): raise ValueError('GMSK requires modulation index h=0.5.')
        half_span = max(4, int(np.ceil(.8/bt)))
        t = np.arange(-half_span*sps, half_span*sps+1)/sps
        gaussian = np.exp(-2*(np.pi*bt*t)**2/np.log(2))
        # Pad edge bits to preserve a sustained symbol's exact steady frequency.
        frequency = np.convolve(np.pad(frequency, (half_span*sps, half_span*sps), mode='edge'), gaussian/gaussian.sum(), mode='valid')
    increments = 2*np.pi*deviation/sample_rate * frequency
    if bt is not None or continuous_phase:
        phase = initial_phase + np.cumsum(increments)
    else:
        phase = initial_phase + np.cumsum(increments.reshape(-1, sps), axis=1).reshape(-1)
    return np.exp(1j*phase)


CONSTELLATION_REFS = {key: constellation(key) for key in ['BPSK', 'QPSK', '8PSK', '8-PSK', 'QAM16', '16-QAM',
                                                        'QAM32', '32-QAM', '16APSK', '16-APSK', '32APSK', '32-APSK', 'GMSK', 'BFSK']}
