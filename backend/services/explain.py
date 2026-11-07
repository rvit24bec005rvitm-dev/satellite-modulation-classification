"""Analytic class-logit saliency and Integrated Gradients for the supplied CNN."""
from time import perf_counter
import numpy as np
from backend.services import classifier
from utils.model_backend import NumPyCNNEngine


def explain(iq: np.ndarray, start=0, target=None, method='integrated-gradients', steps=64) -> dict:
    begin = perf_counter()
    x = classifier.preprocess(iq, start)
    engine = classifier.model()._session
    if not isinstance(engine, NumPyCNNEngine):
        raise RuntimeError('Attributions require the supplied NumPy CNN with validated analytic gradients.')
    if method not in ['integrated-gradients', 'saliency'] or not 16 <= steps <= 256:
        raise ValueError('Choose saliency or integrated-gradients with 16–256 integration steps.')
    mean, std, classes = classifier.stats()
    logits = engine.logits(x)
    target_id = int(logits.argmax()) if target is None else classes.index(target)
    baseline = classifier.preprocess(np.zeros((128, 2), dtype=np.float32))
    base_logit = float(engine.logits(baseline)[target_id])
    input_logit = float(logits[target_id])
    gradient = engine.logit_gradient(x, target_id) / std
    attribution, error, used_steps = None, None, 0
    if method == 'integrated-gradients':
        used_steps = steps
        while True:
            total = np.zeros_like(x, dtype=float)
            for index, alpha in enumerate(np.linspace(0, 1, used_steps+1)):
                total += engine.logit_gradient(baseline + alpha*(x-baseline), target_id) * (.5 if index in [0, used_steps] else 1)
            attribution = (x-baseline)*total/used_steps
            error = float(attribution.sum()-(input_logit-base_logit))
            if abs(error) <= max(.001, .02*abs(input_logit-base_logit)) or used_steps >= 256: break
            used_steps = min(used_steps*2, 256)
    else:
        attribution = gradient
    return {'method': method, 'targetClass': classes[target_id], 'targetQuantity': 'pre-softmax class logit',
            'modelId': '1d-cnn', 'windowStart': start, 'windowLength': 128, 'integrationSteps': used_steps,
            'baseline': {'description': 'Zero raw I/Q capture transformed by the same training z-score', 'rawValue': 0., 'normalizedValue': float(-mean/std)},
            'inputLogit': input_logit, 'baselineLogit': base_logit, 'logitDifference': input_logit-base_logit,
            'attributionSum': float(attribution.sum()) if method == 'integrated-gradients' else None,
            'completenessError': error, 'processingMs': (perf_counter()-begin)*1000,
            'samples': [{'sample': start+i, 'i': float(attribution[i, 0]), 'q': float(attribution[i, 1]),
                         'importance': float(np.abs(attribution[i]).sum()), 'rawI': float(iq[start+i, 0]), 'rawQ': float(iq[start+i, 1])} for i in range(128)],
            'limitations': ['Attributions describe this model relative to the chosen baseline; they do not establish physical causality or correctness.',
                            'Integrated Gradients uses finite trapezoidal integration; inspect the reported completeness residual.',
                            'Saliency is a local derivative in raw sample units; it is not an additive contribution map.']}
