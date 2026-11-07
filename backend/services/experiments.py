"""Read only explicitly published experiment reports, never arbitrary user paths."""
import json
from functools import lru_cache
from backend.core.config import ROOT

PUBLISHED_RUNS = ('rf-control-v1','rf-awgn-v1','resnet-synthetic-v1','cldnn-synthetic-v1')


def experiment_reports() -> list[dict]:
    result=[]
    for run_id in PUBLISHED_RUNS:
        file=ROOT/'experiments/runs'/run_id/'report.json'
        if file.is_file():
            report=json.loads(file.read_text())
            result.append({'id':run_id,**report})
    return result


@lru_cache(maxsize=1)
def experimental_rf():
    """Trusted versioned local artifact only. This path is never supplied by uploads."""
    import hashlib,joblib,sklearn
    directory=ROOT/'experiments/runs/rf-control-v1'
    report=json.loads((directory/'report.json').read_text())
    trained=report['environment'].get('scikitLearn')
    if trained != sklearn.__version__:
        raise RuntimeError(f'Experimental RF requires its recorded scikit-learn {trained}; this host uses {sklearn.__version__}. Retrain the experiment in this environment.')
    path=directory/'random-forest.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['artifact']['sha256']:
        raise RuntimeError('Experimental RF artifact hash does not match the recorded experiment.')
    model=joblib.load(path)  # trusted repository artifact; uploads never reach this loader
    import numpy as np
    if not np.array_equal(model.classes_, np.arange(3)):
        raise RuntimeError('Experimental classifier class order does not match its metadata.')
    class RF:
        backend_name='scikit-learn Random Forest — synthetic experiment'
        def predict(self,value):return model.predict_proba(value[None])[0]
    return RF()


@lru_cache(maxsize=1)
def rf_normalization():
    import numpy as np
    directory=ROOT/'experiments/runs/rf-control-v1'
    report=json.loads((directory/'report.json').read_text())
    with np.load(directory/'normalization.npz',allow_pickle=False) as stats:
        mean,std=float(stats['mean']),float(stats['std'])
    if mean != report['normalization']['mean'] or std != report['normalization']['std']:
        raise RuntimeError('Experimental preprocessing does not match its recorded training configuration.')
    return mean,std


def prepare_experimental_rf(raw):
    from training.models import expert_features
    from utils.preprocessing import prepare_model_input
    mean,std=rf_normalization()
    return expert_features([prepare_model_input(raw,mean,std)])[0]


def experimental_rf_evaluation():
    report=next(r for r in experiment_reports() if r['id']=='rf-control-v1')
    return {'metrics':report['test'],'source':'experiments/runs/rf-control-v1/report.json','datasetId':report['dataset']['id']}


@lru_cache(maxsize=4)
def experimental_network(run_id: str):
    """Load only a named, locally trained, hash-checked synthetic experiment."""
    if run_id not in ['resnet-synthetic-v1','cldnn-synthetic-v1']:
        raise ValueError('Unpublished neural experiment.')
    try: import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('This experiment is trained, but this runtime lacks optional TensorFlow. Use a compatible requirements-training environment; predictions are unavailable here.') from exc
    import hashlib
    import numpy as np
    from scipy.special import softmax
    directory=ROOT/'experiments/runs'/run_id
    report=json.loads((directory/'report.json').read_text())
    path=directory/'best.keras'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['artifact']['sha256']:
        raise RuntimeError('Experimental neural checkpoint hash does not match its report.')
    network=tf.keras.models.load_model(path,compile=False,safe_mode=True)
    if network.input_shape!=(None,128,2) or network.output_shape!=(None,3) or network.layers[-1].get_config().get('activation')!='linear':
        raise RuntimeError('Experimental neural input/logit contract does not match its report.')
    class Network:
        backend_name=f'TensorFlow {tf.__version__} — synthetic experiment'
        _session=network
        def predict(self,value):return softmax(network(value[None],training=False).numpy()[0]).astype(np.float32)
    return Network()


@lru_cache(maxsize=4)
def network_normalization(run_id: str):
    report=json.loads((ROOT/'experiments/runs'/run_id/'report.json').read_text())
    return report['normalization']['mean'],report['normalization']['std']


def prepare_network(raw,run_id):
    from utils.preprocessing import prepare_model_input
    mean,std=network_normalization(run_id)
    return prepare_model_input(raw,mean,std)


def network_evaluation(run_id):
    report=next(r for r in experiment_reports() if r['id']==run_id)
    return {'metrics':report['test'],'source':f'experiments/runs/{run_id}/report.json','datasetId':report['dataset']['id']}
