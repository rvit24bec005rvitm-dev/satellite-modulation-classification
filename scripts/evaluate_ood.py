"""Validate an OOD score experiment with a frozen validation-only threshold.

Input NPZ requires ID/OOD validation/test score arrays and corresponding group
arrays. Higher scores must mean more OOD. Scores must come from a documented
real scoring method; this script does not claim the production model is validated.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services.uncertainty import detection_metrics


def evaluate_scores(path):
    with np.load(path,allow_pickle=False) as d:
        required=['idValidation','oodValidation','idTest','oodTest']
        for key in required:
            if key not in d or key+'Groups' not in d or d[key].shape!=d[key+'Groups'].shape or d[key+'Groups'].dtype.kind not in 'iuUS':
                raise ValueError('Every score vector requires matching source-group IDs.')
        validation_groups=set(d['idValidationGroups'])|set(d['oodValidationGroups'])
        test_groups=set(d['idTestGroups'])|set(d['oodTestGroups'])
        if validation_groups & test_groups:raise ValueError('Validation/test source-group leakage.')
        val=detection_metrics(d['idValidation'],d['oodValidation'])
        test=detection_metrics(d['idTest'],d['oodTest'],threshold=val['threshold'])
        return {'thresholdFitSplit':'validation','validation':val,'test':test,'productionEnabled':False,
                'note':'AUROC and frozen-threshold FPR/TPR apply only to the supplied labeled score sets; provenance must be independently established.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scores',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=evaluate_scores(args.scores);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
