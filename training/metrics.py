"""Shared finite classification/calibration metrics for held-out experiments."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from backend.services.uncertainty import calibration_metrics


def evaluate_probabilities(probabilities, labels, snrs, classes, snr_known=None) -> dict:
    p, y, snr = np.asarray(probabilities), np.asarray(labels), np.asarray(snrs,dtype=float)
    reliability = calibration_metrics(p, y)
    if snr.shape != y.shape:raise ValueError('One SNR slot per labeled frame is required.')
    if snr_known is None:
        if not np.isfinite(snr).all():raise ValueError('One finite SNR per labeled frame is required, or an explicit snrKnown mask for missing values.')
        known=np.ones(len(y),dtype=bool)
    else:
        known=np.asarray(snr_known)
        if known.shape!=y.shape or known.dtype.kind!='b' or not np.isfinite(snr[known]).all() or not np.isnan(snr[~known]).all():
            raise ValueError('snrKnown must be boolean and match SNR slots; only explicitly missing slots may contain NaN.')
    predicted = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=np.arange(len(classes)), zero_division=0)
    return {'samples':len(y), 'accuracy':float(np.mean(predicted==y)), 'precision':float(precision.mean()),
            'recall':float(recall.mean()), 'macroF1':float(f1.mean()), 'confusionMatrix':confusion_matrix(y,predicted,labels=np.arange(len(classes))).tolist(),
            'perClass':[{'className':name,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(classes)],
            'snrAccuracy':[{'snrDb':float(level),'samples':int(np.sum(snr==level)),
                            'accuracy':float(np.mean(predicted[snr==level]==y[snr==level])),
                            'perClass':{name:float(np.mean(predicted[(snr==level)&(y==i)]==i)) if np.any((snr==level)&(y==i)) else None for i,name in enumerate(classes)}}
                           for level in np.unique(snr[known])], 'calibration':reliability,
            'snrAvailability':{'knownSamples':int(known.sum()),'unavailableSamples':int((~known).sum()),
                               'source':'Supplied dataset metadata; measurement provenance is the importer’s responsibility.'}}
