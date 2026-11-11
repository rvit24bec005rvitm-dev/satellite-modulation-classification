"""Bounded descriptive reference comparisons; no fitted OOD thresholds."""
from functools import lru_cache
import numpy as np
from backend.services import classifier, captures, workspace_store as store


def summarize(frames):
    x=np.asarray(frames,dtype=float);a=np.sqrt(np.sum(x*x,axis=-1));power=np.mean(a*a,axis=1)
    spectra=np.abs(np.fft.fft(x[:,:,0]+1j*x[:,:,1],axis=1))**2
    weighted=np.sum(spectra*np.fft.fftfreq(128)[None,:],axis=1)
    centers=np.divide(weighted,spectra.sum(axis=1),out=np.zeros(len(x)),where=spectra.sum(axis=1)>0)
    predictions=[classifier.predict_window(frame.astype(np.float32)) for frame in x]
    classes=classifier.stats()[2];confidence=np.array([p['confidence'] for p in predictions])
    hist,edges=np.histogram(confidence,bins=np.linspace(0,1,11))
    amplitudes=np.quantile(a,[.01,.1,.5,.9,.99]).tolist()
    return {'windowCount':len(x),'iMean':float(x[:,:,0].mean()),'qMean':float(x[:,:,1].mean()),
        'iStd':float(x[:,:,0].std()),'qStd':float(x[:,:,1].std()),'power':float(power.mean()),
        'powerQuantiles':np.quantile(power,[.1,.5,.9]).tolist(),'amplitudeQuantiles':amplitudes,
        'meanNormalizedSpectralCentroid':float(centers.mean()),
        'predictionDistribution':{c:sum(p['predictedModulation']==c for p in predictions) for c in classes},
        'confidenceDistribution':{'edges':edges.tolist(),'counts':hist.tolist()},
        'meanRawConfidence':float(confidence.mean())}


@lru_cache(maxsize=2)
def reference(model_hash):return summarize(classifier.dataset()['signals'])


def compare(record_id,start):
    _,meta=captures.window(record_id,start,128)
    offsets=np.unique(np.linspace(start,meta['sampleCount']-128,min(32,(meta['sampleCount']-start)//128),dtype=int))
    frames=[captures.window(record_id,int(i),128)[0] for i in offsets]
    ref=reference(store.artifact_hash());external=summarize(frames)
    return {'reference':ref,'external':external,'sampledOffsets':offsets.tolist(),
            'rmsRatio':float(np.sqrt(external['power']/ref['power'])),
            'note':'Descriptive indicators against 300 demos with unknown historical splits. At most 32 evenly spaced external windows; dependent windows are not independent measurements. Not a validated OOD test, training-distribution reconstruction, or correctness guarantee.'}
