"""
tests/test_signal_processing.py
Unit tests for signal processing, synthetic generation, and feature extraction.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import (
    normalize_signal,
    generate_iq_signal,
    compute_psd,
    compute_spectrogram,
    shannon_entropy,
    extract_features,
    CONSTELLATION_REFS,
)


def test_normalization():
    sig = np.random.randn(128, 2).astype(np.float32) * 5.0 + 2.0
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    normed = normalize_signal(sig, mean, std)
    assert normed.shape == (128, 2)
    assert np.isclose(float(np.mean(normed)), 0.0, atol=1e-5)
    assert np.isclose(float(np.std(normed)), 1.0, atol=1e-5)


def test_synthetic_signal_generator_shapes():
    for mod in ["BPSK", "QPSK", "8PSK", "QAM16", "16APSK", "32APSK", "GMSK", "BFSK"]:
        sig = generate_iq_signal(mod, num_samples=128, sps=8, rng_seed=42)
        assert sig.shape == (128, 2), f"Failed shape check for {mod}: {sig.shape}"
        assert not np.isnan(sig).any(), f"NaN detected in {mod}"
        assert not np.isinf(sig).any(), f"Inf detected in {mod}"


def test_channel_impairments():
    sig = generate_iq_signal(
        "QPSK",
        num_samples=128,
        sps=8,
        snr_db=15.0,
        doppler_hz=0.05,
        cfo_hz=0.02,
        phase_noise_std=0.05,
        rician_k=10.0,
        twta_alpha_a=0.5,
        rng_seed=123,
    )
    assert sig.shape == (128, 2)
    assert not np.isnan(sig).any()


def test_psd_and_spectrogram():
    sig = np.random.randn(128, 2).astype(np.float32)
    freqs, psd = compute_psd(sig, fs=1e6)
    assert len(freqs) == len(psd)
    assert len(freqs) > 0

    freqs, times, Zxx_db = compute_spectrogram(sig, fs=1e6)
    assert Zxx_db.ndim == 2
    assert len(times) == Zxx_db.shape[1]
    assert len(freqs) == Zxx_db.shape[0]


def test_shannon_entropy():
    # Certain distribution: entropy ~ 0
    p_certain = np.array([1.0, 0.0, 0.0])
    h_certain = shannon_entropy(p_certain)
    assert np.isclose(h_certain, 0.0, atol=1e-5)

    # Uniform distribution: entropy = log2(3) ~ 1.585
    p_uniform = np.array([1/3, 1/3, 1/3])
    h_uniform = shannon_entropy(p_uniform)
    assert np.isclose(h_uniform, np.log2(3), atol=1e-3)


def test_extract_features():
    sig = np.random.randn(128, 2).astype(np.float32)
    feats = extract_features(sig)
    assert isinstance(feats, dict)
    assert "Signal_Power_dBW" in feats
    assert "Kurtosis_I" in feats
    assert "PAPR_dB" in feats
