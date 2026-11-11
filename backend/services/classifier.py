"""Cached exact model and one canonical input transform for every inference source."""
from functools import lru_cache
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
from collections import deque
import threading
import numpy as np
from utils.model_backend import ModelBackend
from utils.signal_processing import shannon_entropy
from utils.preprocessing import prepare_model_input, CLASS_LABELS
from backend.core.config import ROOT, MODEL_PATH
from backend.services.signals import analyze, rows
from backend.services.uncertainty import trust_status

HISTORY = deque(maxlen=100)
LOCK = threading.RLock()


@lru_cache(maxsize=1)
def stats():
    with np.load(ROOT/'normalization_stats_all_snr.npz', allow_pickle=False) as data:
        mean, std = float(data['mean']), float(data['std'])
        classes = [str(c).replace('QAM16', '16-QAM') for c in data['modulations']]
    if not np.isfinite([mean, std]).all() or std <= 0 or tuple(classes) != CLASS_LABELS:
        raise RuntimeError('Invalid training normalization statistics or class ordering.')
    return mean, std, classes


@lru_cache(maxsize=1)
def model():
    import hashlib
    backend=ModelBackend.load(MODEL_PATH)
    backend.artifact_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return backend


@lru_cache(maxsize=1)
def dataset():
    with np.load(ROOT/'classifier_samples.npz', allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def preprocess(iq: np.ndarray, start: int = 0) -> np.ndarray:
    mean, std, _ = stats()
    return prepare_model_input(iq, mean, std, start)


def predict_window(iq: np.ndarray, start: int = 0) -> dict:
    t0 = perf_counter()
    x = preprocess(iq, start)
    prepare_ms = (perf_counter()-t0)*1000
    backend = model()
    with LOCK:
        t0 = perf_counter()
        probabilities = backend.predict(x)
        elapsed = (perf_counter()-t0)*1000
    if (probabilities.shape != (3,) or not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
        raise RuntimeError('Model returned invalid probabilities.')
    mean, std, classes = stats()
    rms = float(np.sqrt(np.mean(iq[start:start+128].astype(float)**2)))
    warnings = []
    if rms/std > 5 or rms/std < .05:
        warnings.append('Capture amplitude is far outside the training scale. Values are preserved; confidence may be unreliable.')
    return {'predictedModulation': classes[int(np.argmax(probabilities))], 'confidence': float(probabilities.max()),
            'probabilities': dict(zip(classes, map(float, probabilities))), 'entropyBits': shannon_entropy(probabilities),
            'processingMs': elapsed, 'preprocessingMs': prepare_ms, 'modelId': '1d-cnn', 'trust': trust_status(),
            'preprocessing': {'mean': mean, 'std': std, 'windowStart': start, 'windowLength': 128,
                              'amplitudePolicy': 'Preserve samples; training z-score only'}, 'warnings': warnings}


def classify(iq: np.ndarray, source: str, fs: float = 1, start: int = 0, *, capture_metadata: dict | None = None, capture_position: dict | None = None) -> dict:
    result = {**predict_window(iq, start), 'id': str(uuid4()), 'createdAt': datetime.now(timezone.utc).isoformat(),
              'source': source, 'sampleRate': fs, 'analysis': analyze(iq, fs), 'samples': rows(iq)}
    if capture_metadata is not None:
        result.update(captureMetadata=capture_metadata,capturePosition=capture_position)
    from backend.services.workspace_store import save_analysis
    save_analysis(result,capture_metadata['id'] if capture_metadata else None)
    with LOCK:
        HISTORY.appendleft({k: result[k] for k in ['id', 'createdAt', 'source', 'predictedModulation', 'confidence', 'entropyBits', 'processingMs']})
    return result
