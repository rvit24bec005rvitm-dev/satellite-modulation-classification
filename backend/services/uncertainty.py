"""Uncalibrated production status and validation-only calibration/detection mathematics."""
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize_scalar


def trust_status() -> dict:
    return {
        'inputValidity': {'status': 'valid', 'checks': ['numeric', 'finite', 'bounded amplitude', 'complete 128×2 window'],
                          'channelOrder': 'I first, Q second; semantic channel swapping cannot be inferred from arbitrary samples'},
        'calibration': {'status': 'not-calibrated', 'temperature': 1., 'reason': 'Independent validation split provenance is unavailable; no calibration was fitted to demo/test frames.'},
        'ood': {'status': 'not-validated', 'detected': None, 'reason': 'No independent labeled ID/OOD validation and test sets are supplied.'},
        'interpretation': 'Softmax confidence and predictive entropy describe this closed-set model; neither establishes correctness or identifies unknown signals.'}


def calibration_metrics(probabilities, labels, bins=10) -> dict:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if (p.ndim != 2 or len(p) != len(y) or not len(p) or y.dtype.kind not in 'iu' or
            not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5) or
            np.any((y < 0) | (y >= p.shape[1])) or not 1 <= bins <= 100):
        raise ValueError('Expected normalized probabilities and integer in-range class labels.')
    confidence, correct = p.max(1), p.argmax(1) == y
    index = np.minimum((confidence*bins).astype(int), bins-1)
    diagram, ece = [], 0.
    for i in range(bins):
        selected = index == i
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        average = float(confidence[selected].mean()) if count else None
        if count: ece += count/len(p)*abs(accuracy-average)
        diagram.append({'lower': i/bins, 'upper': (i+1)/bins, 'count': count, 'accuracy': accuracy, 'confidence': average})
    return {'ece': float(ece), 'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            'brier': float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2, axis=1))),
            'reliability': diagram, 'samples': len(y), 'bins': bins,
            'brierConvention': 'Mean sum of squared class-probability errors, range 0–2'}


def fit_temperature(validation_logits, validation_labels) -> float:
    """Caller must supply validation data only; never use the final test split."""
    logits, y = np.asarray(validation_logits, dtype=float), np.asarray(validation_labels)
    calibration_metrics(softmax(logits, axis=1), y)
    def objective(log_t):
        z = logits/np.exp(log_t)
        return float(np.mean(logsumexp(z, axis=1)-z[np.arange(len(y)), y]))
    result = minimize_scalar(objective, bounds=(-4, 4), method='bounded')
    if not result.success: raise RuntimeError('Temperature optimization failed.')
    return float(np.exp(result.x))


def detection_metrics(id_scores, ood_scores, tpr=.95, threshold=None) -> dict:
    """Larger score means OOD. Select thresholds on validation; freeze for test."""
    from sklearn.metrics import roc_auc_score, roc_curve
    if not np.isfinite(tpr) or not 0 < tpr <= 1 or (threshold is not None and not np.isfinite(threshold)):
        raise ValueError('Target TPR must lie in (0,1], and a supplied threshold must be finite.')
    a, b = np.asarray(id_scores, dtype=float), np.asarray(ood_scores, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b) or not np.isfinite(np.r_[a, b]).all():
        raise ValueError('Finite nonempty ID and OOD score vectors are required.')
    y, scores = np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]
    fpr, recall, thresholds = roc_curve(y, scores, drop_intermediate=False)
    index = np.flatnonzero(recall >= tpr)[0]
    cutoff = float(thresholds[index]) if threshold is None else float(threshold)
    return {'auroc': float(roc_auc_score(y, scores)), 'threshold': cutoff, 'targetTpr': tpr,
            'fpr': float(np.mean(a >= cutoff)), 'tpr': float(np.mean(b >= cutoff)),
            'idSamples': len(a), 'oodSamples': len(b), 'thresholdSource': 'validation ROC' if threshold is None else 'frozen validation threshold'}
