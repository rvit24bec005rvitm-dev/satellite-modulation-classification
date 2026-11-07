"""Fresh descriptive inference on the demo subset; never substitutes for held-out data."""
from functools import lru_cache
import hashlib
import numpy as np
from backend.services import classifier
from backend.core.config import ROOT, MODEL_PATH
from training.metrics import evaluate_probabilities


@lru_cache(maxsize=1)
def demo_diagnostics() -> dict:
    data = classifier.dataset(); raw=data['signals']; mean,std,classes=classifier.stats()
    labels=np.array([classes.index(str(label).replace('QAM16','16-QAM')) for label in data['labels']])
    engine=classifier.model()
    canonical=np.stack([classifier.preprocess(frame) for frame in raw])
    power=np.sqrt(np.mean(np.sum(raw.astype(float)**2,axis=2),axis=1))
    paths={'training-zscore-only':canonical, 'legacy-unit-power-then-zscore':((raw/np.maximum(power[:,None,None],1e-30)-mean)/std).astype(np.float32), 'no-transform':raw}
    outputs={name:engine.predict_batch(frames) for name,frames in paths.items()}
    results={name:evaluate_probabilities(probs,labels,data['snrs'],classes) for name,probs in outputs.items()}
    return {'scope':'Descriptive 300-frame demo subset; training/validation/test membership is unknown. Do not fit thresholds, calibrate or train on these frames.',
            'independentTest':False,'productionChanged':False,'classes':classes,'samples':len(raw),
            'datasetSha256':hashlib.sha256((ROOT/'classifier_samples.npz').read_bytes()).hexdigest(),
            'modelSha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            'preprocessingEvidence':{'savedMean':mean,'savedStd':std,'demoMean':float(raw.mean()),'demoStd':float(raw.std()),
                                     'trainingSource':'Training source and split definitions absent; saved scalar stats, Keras input shape and original saved-sample inference path agree.'},
            'comparisons':results,
            'warning':'Historical 72.44% / 900-observation results are separate stored aggregates and have not been independently reproduced.'}
