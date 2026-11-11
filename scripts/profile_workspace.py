"""Reproducible synthetic import/window profile; never uses or fits real RF data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
import json
import tempfile
import tracemalloc
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from backend.services import captures
from backend.services.advanced_dsp import diagnostics


def main():
    previous=os.environ.get('SATMOD_DATA_DIR');measurements=[]
    try:
        with tempfile.TemporaryDirectory(prefix='satmod-workspace-profile-') as directory:
            root=Path(directory);os.environ['SATMOD_DATA_DIR']=str(root/'managed')
            for count in [100000,1000000]:
                file=root/f'{count}.npy';np.save(file,np.random.default_rng(42).normal(0,.006,(count,2)).astype(np.float32))
                tracemalloc.start();begin=perf_counter()
                with file.open('rb') as stream:
                    result=captures.import_capture(SimpleNamespace(file=stream,filename=file.name),captures.ImportOptions(sessionId='synthetic-benchmark',source='Synthetic benchmark; not RF'))
                elapsed=perf_counter()-begin;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                begin=perf_counter();window,_=captures.window(result['id'],count-128,128);window_ms=(perf_counter()-begin)*1000
                begin=perf_counter();quality=diagnostics(window,8000);quality_ms=(perf_counter()-begin)*1000
                measurements.append({'samples':count,'sourceBytes':file.stat().st_size,'importMs':elapsed*1000,'peakTracedAllocationBytes':peak,
                                     'lastWindowMs':window_ms,'diagnostics128Ms':quality_ms,'welchBins':len(quality['welch'])})
    finally:
        if previous is None:os.environ.pop('SATMOD_DATA_DIR',None)
        else:os.environ['SATMOD_DATA_DIR']=previous
    import platform
    print(json.dumps({'scope':'Synthetic NPY import and windowed diagnostics; traced allocations exclude pre-existing arrays, mapped pages and OS cache. Not a real RF benchmark.',
                      'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'measurements':measurements},indent=2))


if __name__=='__main__':main()
