"""Canonical production-model contract. No inferred amplitude calibration or padding."""
from __future__ import annotations
import numpy as np

INPUT_LENGTH = 128
CLASS_LABELS = ('BPSK', 'QPSK', '16-QAM')


def prepare_model_input(iq: np.ndarray, mean: float, std: float, start: int = 0) -> np.ndarray:
    a = np.asarray(iq)
    if a.dtype.kind not in 'fiu' or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('Expected real (N,2) I/Q samples in I-first, Q-second order.')
    if not np.isfinite(a).all() or np.max(np.abs(a), initial=0) > 1e6:
        raise ValueError('I/Q samples must be finite and within the supported amplitude range.')
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0:
        raise ValueError('Window start must be a nonnegative integer.')
    if start + INPUT_LENGTH > len(a):
        raise ValueError('Classification requires a complete 128-sample window at the selected offset. No padding is applied.')
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError('Training normalization requires a finite mean and positive standard deviation.')
    # Preserve the supplied raw amplitude and the original float32 arithmetic.
    return ((a[start:start + INPUT_LENGTH].astype(np.float32) - np.float32(mean)) / np.float32(std)).astype(np.float32)
