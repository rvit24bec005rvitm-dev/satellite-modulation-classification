"""Reproducible descriptive model/preprocessing audit. Does not train or mutate artifacts."""
import sys,json,hashlib,platform
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.services.diagnostics import demo_diagnostics
from backend.services.classifier import dataset,preprocess
from utils.model_backend import NumPyCNNEngine


def main():
    class ReferenceLoop(NumPyCNNEngine):
        def _conv1d(self,x,w,b):
            k=w.shape[0];padded=np.pad(x,((k//2,k//2),(0,0)))
            return np.stack([np.tensordot(padded[i:i+k],w,axes=([0,1],[0,1]))+b for i in range(len(x))]).astype(np.float32)
    original=ReferenceLoop.from_npz(ROOT/'models/cnn_weights.npz'); optimized=NumPyCNNEngine.from_npz(ROOT/'models/cnn_weights.npz')
    frames=np.stack([preprocess(frame) for frame in dataset()['signals']])
    old=original.predict_batch(frames);new=optimized.predict_batch(frames)
    np.testing.assert_allclose(new,old,atol=2e-5,rtol=2e-5)
    latencies={}
    for name,engine in [('reference-loop',original),('vectorized',optimized)]:
        for _ in range(3):engine.predict(frames[42])
        times=[]
        for _ in range(50):
            start=perf_counter();engine.predict(frames[42]);times.append((perf_counter()-start)*1000)
        latencies[name]={'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),'repetitions':50,'batchSize':1,'latenciesMs':times}
    report=demo_diagnostics()
    report['implementationParity']={'frames':len(frames),'classAgreement':float(np.mean(old.argmax(1)==new.argmax(1))), 'maxProbabilityDifference':float(np.max(abs(old-new)))}
    report['latencies']=latencies
    from backend.services.model_registry import hardware_info
    report['hardware']=hardware_info()
    out=ROOT/'integration/audit/model-diagnostics.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out.relative_to(ROOT)),'accuracyByPreprocessing':{k:v['accuracy'] for k,v in report['comparisons'].items()},'parity':report['implementationParity'],'latencies':{k:{x:y for x,y in v.items() if x!='latenciesMs'} for k,v in latencies.items()}},indent=2))
if __name__=='__main__':main()
