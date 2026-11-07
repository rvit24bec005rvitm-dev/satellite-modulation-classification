"""Shared finite classification/calibration metrics for held-out experiments."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from backend.services.uncertainty import calibration_metrics


def evaluate_probabilities(probabilities, labels, snrs, classes) -> dict:
    p, y, snr = np.asarray(probabilities), np.asarray(labels), np.asarray(snrs)
    reliability = calibration_metrics(p, y)
    if snr.shape != y.shape or not np.isfinite(snr).all(): raise ValueError('One finite SNR per labeled frame is required.')
    predicted = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=np.arange(len(classes)), zero_division=0)
    return {'samples':len(y), 'accuracy':float(np.mean(predicted==y)), 'precision':float(precision.mean()),
            'recall':float(recall.mean()), 'macroF1':float(f1.mean()), 'confusionMatrix':confusion_matrix(y,predicted,labels=np.arange(len(classes))).tolist(),
            'perClass':[{'className':name,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(classes)],
            'snrAccuracy':[{'snrDb':float(level),'samples':int(np.sum(snr==level)),
                            'accuracy':float(np.mean(predicted[snr==level]==y[snr==level])),
                            'perClass':{name:float(np.mean(predicted[(snr==level)&(y==i)]==i)) if np.any((snr==level)&(y==i)) else None for i,name in enumerate(classes)}}
                           for level in np.unique(snr)], 'calibration':reliability}
