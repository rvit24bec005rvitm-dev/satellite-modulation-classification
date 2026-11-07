"""Replay a completed candidate on its exact untouched test split; never refits it."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.data import load_dataset,split_dataset
from training.models import expert_features
from training.metrics import evaluate_probabilities
from utils.preprocessing import CLASS_LABELS,prepare_model_input


def evaluate_run(run_dir,dataset_path):
    root=Path(run_dir);report=json.loads((root/'report.json').read_text());data=load_dataset(dataset_path,allow_small=True)
    if data['datasetSha256']!=report['dataset']['sha256']:raise ValueError('Dataset identity differs from the recorded experiment.')
    indices=np.asarray(report['splits']['test']['indices'])
    explicit=np.full(len(data['signals']),'',dtype='<U10')
    for name,value in report['splits'].items():explicit[value['indices']]=name
    split_dataset({**data,'splits':explicit})
    artifact=root/report['artifact']['filename']
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=report['artifact']['sha256']:raise ValueError('Checkpoint hash differs from the recorded experiment.')
    norm=report['normalization'];frames=np.stack([prepare_model_input(frame,norm['mean'],norm['std']) for frame in data['signals'][indices]])
    if report['model']=='random-forest':
        import joblib,sklearn
        if report['environment']['scikitLearn']!=sklearn.__version__:raise ValueError('Use the recorded scikit-learn version or retrain in a new experiment directory.')
        model=joblib.load(artifact)  # explicit trusted offline checkpoint, never an upload
        probabilities=model.predict_proba(expert_features(frames))
    else:
        import tensorflow as tf
        model=tf.keras.models.load_model(artifact,compile=False)
        probabilities=softmax(model(frames,training=False).numpy(),axis=1)
    with np.load(root/'test-predictions.npz',allow_pickle=False) as saved:
        error=float(np.max(abs(probabilities-saved['probabilities'])))
        np.testing.assert_allclose(probabilities,saved['probabilities'],atol=1e-6)
    return {'protocolId':report['protocolId'],'artifactSha256':report['artifact']['sha256'],'datasetSha256':data['datasetSha256'],
            'maxReplayProbabilityError':error,'test':evaluate_probabilities(probabilities,data['labels'][indices],data['snrs'][indices],CLASS_LABELS)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=evaluate_run(args.run,args.dataset);Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'maxReplayProbabilityError':report['maxReplayProbabilityError'],'accuracy':report['test']['accuracy']},indent=2))
