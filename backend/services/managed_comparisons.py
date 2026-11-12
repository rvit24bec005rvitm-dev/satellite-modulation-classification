"""Reproducible managed-window comparison without inferred alignment or resampling."""
import json
from time import perf_counter
from uuid import uuid4
import numpy as np
from backend.services import captures, classifier, workspace_store as store
from backend.services.saved_explanations import load_analysis, sample_hash, ExplanationConflict
from backend.services.signals import analyze


def load(record_id):
    with store.LOCK:
        store.get('comparison',record_id)
        return json.loads(store.file_path(record_id,'json').read_text())


def stored_prediction(analysis_id,iq,model_hash):
    report=load_analysis(analysis_id)
    mean,std,_=classifier.stats();preprocessing=report.get('preprocessing',{})
    if (report.get('modelId')!='1d-cnn' or report.get('modelHash')!=model_hash or
            preprocessing.get('mean')!=mean or preprocessing.get('std')!=std or preprocessing.get('windowStart')!=0 or
            report.get('inputHash')!=sample_hash(iq[:128])):
        raise ExplanationConflict('Selected history analysis does not match this complete model window, active checkpoint and preprocessing.')
    keys=['predictedModulation','confidence','probabilities','entropyBits','processingMs','preprocessingMs','modelId','preprocessing','warnings','trust']
    return {**{k:report[k] for k in keys},'status':'available','predictionSource':'stored matching analysis',
            'analysisId':report['id'],'timingScope':'Stored historical single-window inference, not a new comparison measurement.'},report


def attribution_difference(first,second,rates_compatible):
    if not first or not second:return {'status':'unavailable','comparisons':[],'reason':'Select matching saved analyses with explanations for both windows.'}
    if not rates_compatible:return {'status':'unavailable','comparisons':[],'reason':'Sample clocks differ or their known/unknown status differs. Attribution subtraction is not performed across incompatible time axes.'}
    matches=[]
    for a in first.get('explanations',[]):
        for b in second.get('explanations',[]):
            if a.get('inputHash')!=first.get('inputHash') or b.get('inputHash')!=second.get('inputHash'):continue
            if a.get('modelHash')!=first.get('modelHash') or b.get('modelHash')!=second.get('modelHash'):continue
            if (a['method'],a['targetClass'],a.get('modelHash'))!=(b['method'],b['targetClass'],b.get('modelHash')):continue
            av=np.array([[p['i'],p['q']] for p in a['samples']],dtype=float)
            bv=np.array([[p['i'],p['q']] for p in b['samples']],dtype=float)
            if av.shape!=(128,2) or bv.shape!=av.shape:continue
            delta=av-bv
            matches.append({'method':a['method'],'targetClass':a['targetClass'],'quantityUnit':a['quantityUnit'],
                            'firstAnalysisId':first['id'],'secondAnalysisId':second['id'],
                            'meanAbsoluteDifference':float(np.abs(delta).mean()),
                            'samples':[{'sample':i,'i':float(row[0]),'q':float(row[1])} for i,row in enumerate(delta)]})
    return {'status':'available' if matches else 'unavailable','comparisons':matches,
            'reason':None if matches else 'No same-method, same-target, same-model explanations are stored for these windows.',
            'note':'First minus second by relative model-window sample index only. No physical sample alignment, resampling or causal interpretation is inferred.'}


def create(first,second,first_analysis_id=None,second_analysis_id=None):
    begin=perf_counter();window_cache={};prediction_cache={}
    try:model_hash=store.artifact_hash()
    except (RuntimeError,OSError):model_hash=None

    def side(selection,analysis_id):
        record_id=store.identifier(selection['captureId']);start=selection['start'];length=selection['length']
        key=(record_id,start,length)
        if key not in window_cache:
            iq,meta=captures.window(record_id,start,length)
            window_cache[key]=(iq,meta,analyze(iq,meta['sampleRate']))
        iq,meta,analysis=window_cache[key];saved=None
        if analysis_id:
            if len(iq)<128:raise ExplanationConflict('A saved prediction requires a complete 128-sample selected window.')
            prediction,saved=stored_prediction(analysis_id,iq,model_hash)
        elif len(iq)<128:
            prediction={'status':'unavailable','reason':'A complete 128-sample window is required for classification.'}
        elif model_hash is None:
            prediction={'status':'unavailable','reason':'The production checkpoint or inference runtime is unavailable.'}
        else:
            prediction_key=sample_hash(iq[:128])
            if prediction_key not in prediction_cache:
                prediction_cache[prediction_key]={'status':'available',**classifier.predict_window(iq),
                    'predictionSource':'computed selected window','timingScope':'Actual single-window CNN inference; excludes upload, DSP and report rendering.'}
            prediction=prediction_cache[prediction_key]
        position=captures.timing(meta,start,length)
        return {'captureId':record_id,'start':start,'length':length,'metadata':meta,'position':position,
                'inputHash':sample_hash(iq),'modelWindowHash':sample_hash(iq[:128]) if len(iq)>=128 else None,
                'analysis':analysis,'prediction':prediction,
                'inputRetention':'Raw selected samples remain in the referenced managed recording; retain/export the recording for independent replay.'},iq,saved

    a,aiq,asaved=side(first,first_analysis_id);b,biq,bsaved=side(second,second_analysis_id)
    am,bm=a['metadata'],b['metadata']
    known_a=bool(am.get('sampleRateKnown'));known_b=bool(bm.get('sampleRateKnown'))
    rates_equal=am['sampleRate']==bm['sampleRate']
    clocks_compatible=rates_equal and known_a==known_b
    normalized=not (known_a and known_b and rates_equal)
    def plot_data(value):
        analysis=value['analysis'];fs=analysis['sampleRate']
        return {'waveform':[{'x':p['sample'] if normalized else p['time'],'i':p['i'],'q':p['q']} for p in analysis['waveform']],
                'constellation':analysis['constellation'],
                'spectrum':[{'frequency':p['frequency']/fs if normalized else p['frequency'],
                             'psd':p['psd']+10*np.log10(fs) if normalized else p['psd']} for p in analysis['spectrum']],
                'phase':[{'x':p['sample'] if normalized else p['time'],'phaseRad':p['phaseRad']} for p in analysis['phaseTrace']]}
    amp_a=np.sqrt(np.sum(aiq.astype(float)**2,axis=1));amp_b=np.sqrt(np.sum(biq.astype(float)**2,axis=1))
    low=float(min(amp_a.min(),amp_b.min()));high=float(max(amp_a.max(),amp_b.max()))
    if low==high:low=max(0,low-.5);high+=.5
    edges=np.linspace(low,high,33)
    def histogram(values):
        counts,_=np.histogram(values,bins=edges)
        return [{'value':float((edges[i]+edges[i+1])/2),'count':int(n),'fraction':float(n/len(values))} for i,n in enumerate(counts)]
    predictions_available=all(s['prediction']['status']=='available' for s in [a,b])
    n=min(len(aiq),len(biq))
    report={'reportVersion':'satmod-comparison-v1','id':str(uuid4()),'createdAt':store.now(),'modelId':'1d-cnn',
            'modelVersion':'supplied-cnn-v1','modelHash':model_hash,'first':a,'second':b,
            'processingMs':(perf_counter()-begin)*1000,'ratesEqual':rates_equal,'sampleClocksCompatible':clocks_compatible,
            'sampleRates':{'first':am['sampleRate'] if known_a else None,'second':bm['sampleRate'] if known_b else None},
            'normalizedAxes':normalized,'overlapSamples':n,'lengthsEqual':len(aiq)==len(biq),
            'sampleMse':float(np.mean((aiq[:n].astype(float)-biq[:n])**2)) if clocks_compatible else None,
            'powerDifferenceDb':a['analysis']['powerDb']-b['analysis']['powerDb'] if all(s['analysis']['powerDb'] is not None for s in [a,b]) else None,
            'confidenceDifference':a['prediction']['confidence']-b['prediction']['confidence'] if predictions_available else None,
            'plots':{'first':plot_data(a),'second':plot_data(b),'timeUnit':'Sample index' if normalized else 'Time (s)',
                     'frequencyUnit':'cycles/sample' if normalized else 'Hz',
                     'psdUnit':'dB sample units squared / (cycles/sample)' if normalized else 'dB sample units squared / Hz',
                     'amplitudeDistribution':{'edges':edges.tolist(),'first':histogram(amp_a),'second':histogram(amp_b),
                                              'note':'Common amplitude bins; fractions normalize only bin counts, not input signal amplitudes.'}},
            'attributionDifferences':attribution_difference(asaved,bsaved,clocks_compatible),
            'limitations':['No sample alignment or resampling is inferred. MSE uses overlapping relative sample indices only and is not a modulation-similarity score.',
                           'Unequal or unknown clocks use normalized frequency/sample axes; spectral densities are transformed consistently.',
                           'All powers and amplitudes use relative recorder sample units, not calibrated RF watts.',
                           'Closed-set CNN outputs are uncalibrated; confidence differences do not establish a difference in correctness.',
                           'Phase is independently unwrapped in each window; near-zero samples have unavailable phase.']}
    summary={k:report[k] for k in ['id','createdAt','modelId','modelHash','processingMs']}
    summary.update(source=f"{am['source']} / {bm['source']}",firstCaptureId=a['captureId'],secondCaptureId=b['captureId'])
    store.save_document('comparison',summary,report)
    return report
