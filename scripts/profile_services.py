"""Measured local service/model timings and bounded visualization sizes, no fake baselines."""
import sys,json,platform,os,tracemalloc
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services import classifier
from backend.services.signals import analyze
from utils.signal_processing import generate_iq_signal
from backend.services.model_registry import hardware_info


def main():
    start=perf_counter();model=classifier.model();load_ms=(perf_counter()-start)*1000
    raw=classifier.dataset()['signals'][42];x=classifier.preprocess(raw)
    for _ in range(3):model.predict(x)
    def measure(function,repeats=20):
        times=[]
        for _ in range(repeats):
            t=perf_counter();function();times.append((perf_counter()-t)*1000)
        return {'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),'repetitions':repeats}
    results={'modelInitializationMs':load_ms,'modelInitializationScope':'First exact-artifact load in this process; excludes Python imports; filesystem caches may already be warm.',
             'preprocessing':measure(lambda:classifier.preprocess(raw)), 'inference':measure(lambda:model.predict(x)),
             'signalGeneration1024':measure(lambda:generate_iq_signal('QPSK',num_samples=1024,rng_seed=42)),
             'analysis128':measure(lambda:analyze(raw)), 'hardware':hardware_info()}
    long=generate_iq_signal('QPSK',num_samples=65536,rng_seed=42)
    t=perf_counter();analysis=analyze(long);untraced_elapsed=(perf_counter()-t)*1000
    tracemalloc.start();analyze(long);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    results['analysis65536']={'elapsedMs':untraced_elapsed,'peakTracedAllocationsBytes':peak,
                               'waveformPoints':len(analysis['waveform']),'spectrumBins':len(analysis['spectrum']),
                               'constellationPoints':len(analysis['constellation']),'spectrogramTimeBins':len(analysis['spectrogram']['times'])}
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results['processPeakRssBytes']=int(rss if platform.system()=='Darwin' else rss*1024)
    except ImportError:results['processPeakRssBytes']=None
    from fastapi.testclient import TestClient
    from backend.main import app
    body={'samples':[{'i':float(i),'q':float(q)} for i,q in raw]}
    with TestClient(app) as client:
        def call():
            response=client.post('/api/classify',json=body)
            assert response.status_code==200,response.text
        results['classificationApi128']=measure(call,10)
        results['classificationApi128']['scope']='ASGI TestClient request, JSON validation, inference, DSP and serialization; no real-network RTT.'
    root=Path(__file__).resolve().parents[1]
    (root/'integration/audit/performance.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
