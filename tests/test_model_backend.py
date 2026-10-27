"""
tests/test_model_backend.py
Unit tests for ModelBackend loading and inference.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.model_backend import ModelBackend


def test_model_backend_loading():
    model_path = ROOT / "satellite_modulation_cnn_all_snr.keras"
    backend = ModelBackend.load(model_path)
    assert backend is not None
    assert backend.backend_name != ""


def test_model_inference_output():
    model_path = ROOT / "satellite_modulation_cnn_all_snr.keras"
    backend = ModelBackend.load(model_path)

    # Test random input
    sample = np.random.randn(128, 2).astype(np.float32)
    probs = backend.predict(sample)

    assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"
    assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-4), f"Probabilities do not sum to 1: {np.sum(probs)}"
    assert (probs >= 0.0).all(), "Negative probability detected"
    assert (probs <= 1.0).all(), "Probability exceeds 1.0"


def test_dataset_samples_compatibility():
    data_path = ROOT / "classifier_samples.npz"
    if not data_path.exists():
        return

    stats_path = ROOT / "normalization_stats_all_snr.npz"
    stats = np.load(stats_path)
    mean = float(stats["mean"])
    std = float(stats["std"])

    backend = ModelBackend.load(ROOT / "satellite_modulation_cnn_all_snr.keras")
    data = np.load(data_path)

    # Test first 5 samples
    for i in range(5):
        sig = (data["signals"][i] - mean) / std
        probs = backend.predict(sig)
        assert probs.shape == (3,)
        assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-4)
