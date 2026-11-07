import csv
import json
import numpy as np
from backend.core.config import ROOT
from backend.services.classifier import stats


def snr_results() -> list[dict]:
    with (ROOT/'snr_accuracy_results.csv').open() as f:
        return [{'snrDb':float(r['SNR_dB']), 'samples':int(float(r['Samples'])),
                 'accuracy':float(r['Overall_Accuracy_Percent'])/100,
                 'BPSK':float(r['BPSK_Accuracy_Percent'])/100,
                 'QPSK':float(r['QPSK_Accuracy_Percent'])/100,
                 '16-QAM':float(r['QAM16_Accuracy_Percent'])/100} for r in csv.DictReader(f)]


def evaluation() -> dict:
    raw = json.loads((ROOT/'evaluation_results.json').read_text())
    cm = np.asarray(raw['confusion_matrix'],dtype=int)
    if cm.shape != (3,3) or np.any(cm < 0) or cm.sum() <= 0:
        raise ValueError('Stored confusion matrix must be a nonempty nonnegative 3×3 count matrix.')
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(cm.diagonal(),predicted,out=np.zeros(3),where=predicted!=0)
    recall = np.divide(cm.diagonal(),support,out=np.zeros(3),where=support!=0)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall!=0)
    accuracy = float(cm.trace()/cm.sum())
    per_class = [{'className':label,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,label in enumerate(stats()[2])]
    errors = sorted([{'actual':stats()[2][i],'predicted':stats()[2][j],'count':int(cm[i,j])} for i in range(3) for j in range(3) if i!=j],key=lambda e:-e['count'])
    return {'metrics':{'accuracy':accuracy,'precision':float(precision.mean()),'recall':float(recall.mean()),'f1':float(f1.mean()),'samples':int(cm.sum())},
            'classes':stats()[2],'confusionMatrix':cm.tolist(),'perClass':per_class,'misclassifications':errors,
            'snrAccuracy':snr_results(),'source':'evaluation_results.json + snr_accuracy_results.csv',
            'notes':['Metrics are recomputed from the stored 900-observation confusion matrix.','The included demo library has 300 frames and is a separate subset; it cannot reproduce the complete evaluation.','Training split provenance is not recorded in the supplied artifacts.','SNR percentages were rounded in the source CSV.'], 'storedReport':raw}
