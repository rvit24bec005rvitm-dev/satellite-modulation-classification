"""Trusted local model registry. No uploaded file can register or execute a loader."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol
from time import perf_counter
import hashlib
import json
import platform
import numpy as np
from backend.services import classifier
from backend.services.evaluation import evaluation
from backend.core.config import MODEL_PATH


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
    artifact_path: Path | None = None
    version: str | None = None
    class_labels: tuple[str, ...] = ('BPSK', 'QPSK', '16-QAM')
    preprocessing_metadata: dict | None = None
    evaluation_kind: str = 'historical-stored-report'
    training_metadata: dict = field(default_factory=lambda: {'status':'unavailable', 'reason':'Original training data, source-capture groups, split IDs and seeds were not supplied.'})


REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    if spec.id in REGISTRY: raise ValueError(f'Model is already registered: {spec.id}')
    if spec.loader is not None and spec.prepare is None: raise ValueError('A trained model requires an explicit preprocessing function.')
    REGISTRY[spec.id] = spec


@lru_cache(maxsize=16)
def artifact_metadata(path: Path, modified_ns: int) -> dict:
    return {'artifactName':path.name, 'artifactBytes':path.stat().st_size,
            'artifactSha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def hardware_info() -> dict:
    try:
        from threadpoolctl import threadpool_info
        pools = [{k:p[k] for k in ['internal_api','num_threads','version','architecture'] if k in p} for p in threadpool_info()]
    except ImportError: pools = []
    return {'os':platform.system(), 'release':platform.release(), 'machine':platform.machine(),
            'processor':platform.processor() or 'not reported', 'python':platform.python_version(),
            'numpy':np.__version__, 'threadpools':pools, 'device':'CPU', 'batchSize':1}


def describe_models() -> list[dict]:
    result = []
    for spec in REGISTRY.values():
        item = {'id':spec.id, 'name':spec.name, 'version':spec.version, 'classLabels':list(spec.class_labels),
                'inputShape':list(spec.input_shape), 'trainingMetadata':spec.training_metadata,
                'preprocessing':'Saved scalar training z-score; I/Q amplitude preserved' if spec.id=='1d-cnn' else 'The candidate’s own training transform and documented feature schema',
                'preprocessingConfiguration':spec.preprocessing_metadata}
        if spec.loader is None:
            item.update(status='not-trained', reason='No independently trained original-domain artifact or evaluation is supplied.',trainingEntryPoint='python scripts/train.py --help')
        else:
            try:
                if spec.artifact_path:
                    item.update(artifact_metadata(spec.artifact_path,spec.artifact_path.stat().st_mtime_ns))
                if spec.parameter_count is not None:item['parameters']=spec.parameter_count
                # Stored experiment evidence remains available even when optional inference dependencies are absent.
                if spec.evaluation_provider:
                    stored=spec.evaluation_provider()
                    historical=spec.evaluation_kind=='historical-stored-report'
                    item.update(accuracy=stored['metrics']['accuracy'],evaluationMetadata={
                        'status':spec.evaluation_kind,'datasetId':stored.get('datasetId'),'source':stored['source'],'samples':stored['metrics']['samples'],
                        'independentlyReproduced':not historical,'splitProvenance':'unavailable' if historical else 'group-disjoint recorded train/validation/test',
                        'note':'Supplied historical report; not independently reproduced.' if historical else 'Controlled synthetic test; not comparable to the historical 900-observation report. Use the matched-test comparison.'})
                backend=spec.loader()
                item.update(status='available',backend=backend.backend_name)
                engine=getattr(backend,'_session',None)
                if spec.id=='1d-cnn':
                    mean,std,_=classifier.stats()
                    item['preprocessingConfiguration']={'mean':mean,'std':std,'dtype':'float32','channelOrder':['I','Q'],'unitPowerApplied':False}
                if hasattr(engine,'w'):
                    total=sum(int(v.size) for v in engine.w.values())
                    moving=sum(int(v.size) for k,v in engine.w.items() if k.endswith(('_m','_v')))
                    item.update(parameters=total,trainableParameters=total-moving,nonTrainableParameters=moving)
                elif hasattr(engine,'count_params'):item['parameters']=int(engine.count_params())
            except Exception as exc:item.update(status='unavailable',error=str(exc))
        result.append(item)
    return result


def benchmark_model(model_id: str, sample_id=42, iterations=30) -> dict:
    spec = REGISTRY[model_id]
    if spec.loader is None or spec.prepare is None: raise ValueError('No independently trained artifact.')
    if not 5 <= iterations <= 100 or not 0 <= sample_id < len(classifier.dataset()['signals']): raise ValueError('Invalid benchmark bounds.')
    backend = spec.loader()
    raw = classifier.dataset()['signals'][sample_id]
    with classifier.LOCK:
        x = spec.prepare(raw)
        for _ in range(3): backend.predict(x)
        inference, preparation = [], []
        for _ in range(iterations):
            begin=perf_counter(); x=spec.prepare(raw); preparation.append((perf_counter()-begin)*1000)
            begin=perf_counter(); backend.predict(x); inference.append((perf_counter()-begin)*1000)
    return {'modelId':model_id,'iterations':iterations,'sampleId':sample_id,'warmupIterations':3,
            'meanMs':float(np.mean(inference)),'medianMs':float(np.median(inference)),'p95Ms':float(np.percentile(inference,95)),
            'minMs':min(inference),'maxMs':max(inference),'throughputFps':float(1000/np.mean(inference)), 'latenciesMs':inference,
            'preprocessing':{'latenciesMs':preparation,'medianMs':float(np.median(preparation)), 'p95Ms':float(np.percentile(preparation,95))},
            'hardware':hardware_info(), 'modelVersion':spec.version,
            'scope':'Single-frame CPU inference only; excludes preprocessing, upload, and analysis. Compare timings only within this environment.'}


register_model(ModelSpec('1d-cnn','1D-CNN',loader=classifier.model,prepare=classifier.preprocess,
                         evaluation_provider=evaluation,artifact_path=MODEL_PATH,version='supplied-cnn-v1'))
for model_id,name in [('resnet','ResNet · original domain'),('cldnn','CLDNN · original domain'),('random-forest','Random Forest · original domain')]:
    register_model(ModelSpec(model_id,name))

# A separately identified educational baseline; production IDs are not substituted.
# Absence or an incompatible sklearn runtime produces an honest unavailable state.
from backend.services.experiments import experimental_rf, prepare_experimental_rf, experimental_rf_evaluation
from backend.core.config import ROOT
_experimental_artifact = ROOT/'experiments/runs/rf-control-v1/random-forest.joblib'
if _experimental_artifact.is_file():
    _rf_metadata=json.loads((_experimental_artifact.parent/'report.json').read_text())
    register_model(ModelSpec('rf-experimental','Random Forest · synthetic experiment',loader=experimental_rf,
                             prepare=prepare_experimental_rf,evaluation_provider=experimental_rf_evaluation,
                             artifact_path=_experimental_artifact,version='controlled-synthetic-rf-v1',
                             evaluation_kind='held-out-synthetic-experiment',
                             preprocessing_metadata={**_rf_metadata['normalization'],'featureSchema':_rf_metadata.get('featureSchema')},
                             training_metadata={'status':'trained-experimental','reason':'Seeded 1,800-frame generated educational dataset; source groups and validation/test splits recorded. Not validated on independent satellite captures.'}))

from backend.services.experiments import experimental_network,prepare_network,network_evaluation
for _run,_name in [('resnet-synthetic-v1','ResNet'),('cldnn-synthetic-v1','CLDNN')]:
    _directory=ROOT/'experiments/runs'/_run
    if (_directory/'report.json').is_file() and (_directory/'best.keras').is_file():
        _metadata=json.loads((_directory/'report.json').read_text())
        register_model(ModelSpec(_run,f'{_name} · synthetic experiment',
                                 loader=lambda run=_run:experimental_network(run),
                                 prepare=lambda raw,run=_run:prepare_network(raw,run),
                                 evaluation_provider=lambda run=_run:network_evaluation(run),
                                 artifact_path=_directory/'best.keras',version=_run,
                                 parameter_count=_metadata.get('parameterCounts',{}).get('total'),
                                 preprocessing_metadata=_metadata['normalization'],
                                 evaluation_kind='held-out-synthetic-experiment',
                                 training_metadata={'status':'trained-experimental','reason':'Real checkpoint from an eight-epoch bounded synthetic experiment; requires optional TensorFlow for online inference. Not validated on original-domain independent satellite data.'}))
