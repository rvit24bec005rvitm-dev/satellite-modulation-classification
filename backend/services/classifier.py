"""One preprocessing path for all input sources, with cached real model inference."""
from functools import lru_cache
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
from collections import deque
import threading
import numpy as np
from utils.model_backend import ModelBackend
from utils.signal_processing import normalize_signal, shannon_entropy
from backend.core.config import ROOT, MODEL_PATH
from backend.services.signals import analyze, rows

HISTORY = deque(maxlen=100)
LOCK = threading.RLock()

@lru_cache(maxsize=1)
def stats():
    with np.load(ROOT/'normalization_stats_all_snr.npz', allow_pickle=False) as data:
        mean,std = float(data['mean']),float(data['std'])
        classes = [str(c).replace('QAM16','16-QAM') for c in data['modulations']]
    if not np.isfinite([mean,std]).all() or std <= 0:
        raise RuntimeError('Invalid training normalization statistics.')
    return mean,std,classes

@lru_cache(maxsize=1)
def model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f'Model artifact is missing: {MODEL_PATH.name}')
    return ModelBackend.load(MODEL_PATH)

@lru_cache(maxsize=1)
def dataset():
    with np.load(ROOT/'classifier_samples.npz', allow_pickle=False) as d:
        return {k:d[k].copy() for k in d.files}


def preprocess(iq: np.ndarray, start: int = 0) -> np.ndarray:
    if start + 128 > len(iq):
        raise ValueError('Classification requires a complete 128-sample window at the selected offset. No padding is applied.')
    mean,std,_ = stats()
    return normalize_signal(iq[start:start+128].astype(np.float32),mean,std).astype(np.float32)


def classify(iq: np.ndarray, source: str, fs: float = 1, start: int = 0) -> dict:
    x = preprocess(iq,start)
    backend = model()
    with LOCK:
        t0 = perf_counter()
        probabilities = backend.predict(x)
        elapsed = (perf_counter()-t0)*1000
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise RuntimeError('Model returned invalid probabilities.')
    mean,std,classes = stats()
    rms = float(np.sqrt(np.mean(iq[start:start+128]**2)))
    warnings = []
    if rms/std > 5 or rms/std < .05:
        warnings.append('Capture amplitude is far outside the training scale. Values are preserved; confidence may be unreliable.')
    result = dict(id=str(uuid4()),createdAt=datetime.now(timezone.utc).isoformat(), source=source,
                  predictedModulation=classes[int(np.argmax(probabilities))], confidence=float(np.max(probabilities)),
                  probabilities=dict(zip(classes,map(float,probabilities))), entropyBits=shannon_entropy(probabilities),
                  processingMs=elapsed, sampleRate=fs, modelId='1d-cnn', preprocessing={'mean':mean,'std':std,'windowStart':start,'windowLength':128,'amplitudePolicy':'Preserve samples; training z-score only'},
                  warnings=warnings, analysis=analyze(iq,fs), samples=rows(iq))
    with LOCK:
        HISTORY.appendleft({k:result[k] for k in ['id','createdAt','source','predictedModulation','confidence','entropyBits','processingMs']})
    return result
