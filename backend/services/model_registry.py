"""Trusted-code extension point for independently trained benchmark models.

Register a loader, that model's preprocessing, and (optionally) its own measured
evaluation. Never use another model's probabilities as a stand-in prediction.
Uploaded files cannot register or execute model loaders.
"""
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
from backend.services import classifier
from backend.services.evaluation import evaluation

class Predictor(Protocol):
    backend_name: str
    def predict(self, value: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    loader: Callable[[], Predictor] | None = None
    prepare: Callable[[np.ndarray], np.ndarray] | None = None
    evaluation_provider: Callable[[], dict] | None = None
    parameter_count: int | None = None
    input_shape: tuple[int, int] = (128, 2)

REGISTRY: dict[str, ModelSpec] = {}

def register_model(spec: ModelSpec) -> None:
    if spec.id in REGISTRY:
        raise ValueError(f'Model is already registered: {spec.id}')
    if spec.loader is not None and spec.prepare is None:
        raise ValueError('A trained model requires an explicit preprocessing function.')
    REGISTRY[spec.id] = spec


def describe_models() -> list[dict]:
    result = []
    for spec in REGISTRY.values():
        item = {'id': spec.id, 'name': spec.name}
        if spec.loader is None:
            item.update(status='not-trained', reason='No independently trained artifact or evaluation is supplied.')
        else:
            try:
                backend = spec.loader()
                item.update(status='available', backend=backend.backend_name, inputShape=list(spec.input_shape))
                count = spec.parameter_count
                engine = getattr(backend, '_session', None)
                if count is None and hasattr(engine, 'w'):
                    count = sum(int(v.size) for v in engine.w.values())
                elif count is None and hasattr(engine, 'count_params'):
                    count = int(engine.count_params())
                if count is not None:
                    item['parameters'] = count
                if spec.evaluation_provider is not None:
                    item['accuracy'] = spec.evaluation_provider()['metrics']['accuracy']
            except Exception as exc:
                item.update(status='unavailable', error=str(exc))
        result.append(item)
    return result

register_model(ModelSpec('1d-cnn', '1D-CNN', loader=lambda: classifier.model(),
                         prepare=classifier.preprocess, evaluation_provider=evaluation))
for model_id, name in [('resnet', 'ResNet'), ('cldnn', 'CLDNN'), ('random-forest', 'Random Forest')]:
    register_model(ModelSpec(model_id, name))
