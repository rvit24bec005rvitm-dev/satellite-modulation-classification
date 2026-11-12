import os,json,tempfile,time,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from backend.services import workspace_store as store,captures
result={}
for location,base in [('temporaryFilesystem',None),('workspaceFilesystem',str(Path.cwd()))]:
    measurements_by_call={}
    with tempfile.TemporaryDirectory(prefix='satmod-store-profile-',dir=base) as directory:
        os.environ['SATMOD_DATA_DIR']=str(Path(directory)/'managed')
        record=store.put('capture',{'source':'Synthetic metadata/window microbenchmark','sampleCount':128})
        store.file_path(record['id'],'iq').write_bytes(np.zeros((128,2),dtype='<f4').tobytes())
        calls={'resolvedRoot':lambda:store.root(),'filePath':lambda:store.file_path(record['id'],'iq'),'getRecord':lambda:store.get('capture',record['id']),'window128':lambda:captures.window(record['id'])}
        for name,call in calls.items():
            for _ in range(20):call()
            values=[]
            for _ in range(200):
                begin=time.perf_counter();call();values.append((time.perf_counter()-begin)*1000)
            measurements_by_call[name]={'medianMs':statistics.median(values),'p95Ms':sorted(values)[189],'repetitions':200,'warmup':20}
        store.close_connections()
    result[location]=measurements_by_call
print(json.dumps(result))
