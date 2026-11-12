"""Identity-checked persistent explanations of genuine production-CNN logits."""
import hashlib
import json
import threading
import numpy as np
from scipy.stats import spearmanr
from backend.services import captures, classifier, workspace_store as store
from backend.services.explain import explain
from backend.services.signals import from_rows, validate_iq

SLOTS=threading.BoundedSemaphore(2)


class ExplanationConflict(ValueError):
    """Historical inputs or model identity cannot support this explanation."""


def sample_hash(iq):
    return hashlib.sha256(np.asarray(iq,dtype='<f4').tobytes()).hexdigest()


def load_analysis(analysis_id):
    with store.LOCK:
        store.get('analysis',analysis_id)
        return json.loads(store.file_path(analysis_id,'json').read_text())


def checked_inputs(report, supplied=None):
    if report.get('modelId')!='1d-cnn' or report.get('modelHash')!=store.artifact_hash():
        raise ExplanationConflict('The current production model does not match this historical analysis. Classify it again explicitly.')
    mean,std,_=classifier.stats();saved=report.get('preprocessing',{})
    if saved.get('mean')!=mean or saved.get('std')!=std:
        raise ExplanationConflict('Historical preprocessing does not match the active model; it cannot be silently replaced.')
    if supplied is not None:
        iq=validate_iq(supplied)
    elif report.get('captureId'):
        try:meta=store.get('capture',report['captureId'])
        except KeyError as exc:raise ExplanationConflict('The managed recording was deleted. Supply the original samples matching the stored input hash.') from exc
        captured_position=report.get('capturePosition',{})
        position=captured_position.get('fileSample')
        if position is None and captured_position.get('sourceSample') is not None:
            position=captured_position['sourceSample']-meta.get('sigmfSampleOffset',0)
        if position is None:raise ExplanationConflict('This analysis lacks a reproducible capture position. Supply its original samples.')
        try:iq,_=captures.window(meta['id'],position-meta['sourceOffset'],128)
        except (KeyError,ValueError) as exc:raise ExplanationConflict('The retained recording no longer supplies this analysis window.') from exc
    else:
        raise ExplanationConflict('Original samples were not retained for this legacy analysis. Supply the complete original samples or use a managed capture.')
    if not report.get('inputHash') or sample_hash(iq)!=report['inputHash']:
        raise ExplanationConflict('Supplied or retained samples do not match the historical input hash.')
    start=saved.get('windowStart',0)
    if type(start) is not int or start<0 or start+128>len(iq):
        raise ExplanationConflict('Historical classification window is invalid for these samples.')
    return iq,start


def diagnostics(explanation):
    values=np.array([[p['i'],p['q']] for p in explanation['samples']],dtype=float)
    importance=np.abs(values).sum(axis=1);total=float(importance.sum())
    order=np.argsort(-importance,kind='stable')
    selected=np.zeros(len(values),dtype=bool)
    selected[order[:16]]=importance[order[:16]]>0
    regions=[];start=None
    for i,active in enumerate([*selected,False]):
        if active and start is None:start=i
        if not active and start is not None:
            amount=float(importance[start:i].sum())
            regions.append({'startSample':explanation['windowStart']+start,'endSample':explanation['windowStart']+i,
                            'absoluteImportance':amount,'shareOfAbsoluteImportance':amount/total if total else 0})
            start=None
    channels={name:{'positiveSum':float(np.maximum(values[:,i],0).sum()),
                    'negativeSum':float(np.minimum(values[:,i],0).sum()),'absoluteSum':float(np.abs(values[:,i]).sum())}
              for i,name in enumerate(['i','q'])}
    integrated=explanation['method']=='integrated-gradients'
    return {'channels':channels,'importantRegions':regions,
            'topSamples':[{'sample':explanation['samples'][int(i)]['sample'],'importance':float(importance[i])} for i in order[:8]],
            'quantityDescription':'Signed additive class-logit contributions relative to the recorded baseline.' if integrated else
                                  'Signed local class-logit derivatives per raw sample unit; channel sums are derivative summaries, not additive contributions.',
            'regionRule':'At most 16 highest-absolute-importance samples, joined only when contiguous. These are model-ranked regions, not detected RF packets or statistical significance.'}


def method_comparison(explanations,target=None):
    if target is None and explanations:target=explanations[-1]['targetClass']
    by_method={e['method']:e for e in explanations if e['targetClass']==target}
    if not all(m in by_method for m in ['saliency','integrated-gradients']):
        return {'status':'unavailable','targetClass':target,'spearmanRho':None,'note':'Store both methods for the same class to compare sample rankings.'}
    a=np.array([s['importance'] for s in by_method['saliency']['samples']])
    b=np.array([s['importance'] for s in by_method['integrated-gradients']['samples']])
    rho=float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else None
    return {'status':'available','targetClass':target,'samples':len(a),'spearmanRho':rho,
            'measure':'Spearman correlation of absolute per-sample importance rankings',
            'note':'Descriptive ranking comparison only. Raw derivatives and integrated contributions have different units; their magnitudes are not normalized into equivalent quantities. Constant rankings have no defined correlation.'}


def response(report,target=None):
    explanations=report.get('explanations',[])
    targets=list(dict.fromkeys(e['targetClass'] for e in explanations))
    return {'reportVersion':'satmod-explanations-v1','analysisId':report['id'],'createdAt':report['createdAt'],
            'modelId':report['modelId'],'modelVersion':report.get('modelVersion'),'modelHash':report.get('modelHash'),
            'inputHash':report.get('inputHash'),'prediction':{k:report[k] for k in ['predictedModulation','confidence','probabilities','entropyBits']},
            'captureMetadata':report.get('captureMetadata'),'capturePosition':report.get('capturePosition'),
            'explanations':explanations,'methodComparison':method_comparison(explanations,target),
            'methodComparisons':[method_comparison(explanations,value) for value in targets]}


def create(analysis_id,method='both',target=None,steps=64,supplied=None):
    if not SLOTS.acquire(blocking=False):raise BlockingIOError('Two explanation requests are already running; retry after they finish.')
    try:
        report=load_analysis(analysis_id)
        iq,start=checked_inputs(report,supplied)
        target=target or report['predictedModulation']
        methods=['integrated-gradients','saliency'] if method=='both' else [method]
        generated=[]
        for name in methods:
            result=explain(iq,start,target,name,steps)
            result.update(createdAt=store.now(),modelVersion=report.get('modelVersion'),modelHash=report['modelHash'],inputHash=report['inputHash'],
                          quantityUnit='class-logit contribution' if name=='integrated-gradients' else 'class-logit derivative / raw sample unit')
            result['diagnostics']=diagnostics(result);generated.append(result)
        with store.LOCK:
            current=load_analysis(analysis_id)
            if current.get('inputHash')!=report['inputHash'] or current.get('modelHash')!=report['modelHash']:
                raise ExplanationConflict('Analysis identity changed while explaining it.')
            keep=[e for e in current.get('explanations',[]) if not any(e['method']==g['method'] and e['targetClass']==g['targetClass'] for g in generated)]
            current.update(explanationReportVersion='satmod-explanations-v1',explanations=keep+generated,
                           explanationUpdatedAt=store.now())
            summary=store.get('analysis',analysis_id)
            summary.update(explanationCount=len(current['explanations']),explanationUpdatedAt=current['explanationUpdatedAt'])
            store.save_document('analysis',summary,current)
        return response(current,target)
    finally:SLOTS.release()


def supplied_rows(samples):
    return None if samples is None else from_rows(samples)
