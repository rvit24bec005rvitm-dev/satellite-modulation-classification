"""Exercise managed operations and verify them after container recreation (stdlib only)."""
import argparse
import json
import os
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check(base_url, state_file, verify_existing=False):
    base_url=base_url.rstrip('/')
    token=os.environ.get('SATMOD_ACCESS_TOKEN','')
    if not token:raise ValueError('Set SATMOD_ACCESS_TOKEN for this protected-deployment check.')

    def request(path,data=None,mime='application/json',method=None,authorized=True):
        headers={'X-Satmod-Token':token} if authorized else {}
        if data is not None:
            headers['Content-Type']=mime
            if mime=='application/json':data=json.dumps(data).encode()
        with urlopen(Request(base_url+path,data=data,headers=headers,method=method),timeout=30) as response:return response.read()

    def api(path,data=None,**kwargs):return json.loads(request(path,data,**kwargs))

    # The anonymous health route remains useful, while stored signals are protected.
    for _ in range(90):
        try:
            if api('/api/health')['connected']:break
        except OSError:pass
        time.sleep(.5)
    else:raise AssertionError('Model health did not become ready.')
    try:
        request('/api/workspace/captures',authorized=False)
        raise AssertionError('Protected capture listing accepted an anonymous request.')
    except HTTPError as exc:assert exc.code==401,exc.code

    if not verify_existing:
        generated=api('/api/signal/generate',{'modulation':'BPSK','numSamples':1024,'sampleRate':8000,'seed':847})
        raw=b''.join(struct.pack('<ff',p['i'],p['q']) for p in generated['samples'])
        boundary='satmod-managed-smoke-boundary'
        options={'format':'raw','datatype':'cf32_le','sampleRate':8000,'sessionId':'container-generated-fixture',
                 'source':'Generated BPSK container fixture, not real RF','license':'Synthetic test fixture'}
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="options"\r\n\r\n{json.dumps(options)}\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.bin"\r\n'
              'Content-Type: application/octet-stream\r\n\r\n').encode()+raw+f'\r\n--{boundary}--\r\n'.encode()
        capture=json.loads(request('/api/workspace/captures',body,f'multipart/form-data; boundary={boundary}'))
        result=api(f"/api/workspace/captures/{capture['id']}/classify?start=128",{})
        job=api('/api/workspace/jobs/batch',{'captureId':capture['id'],'windows':4,'stride':128})
        for _ in range(120):
            current=api('/api/workspace/jobs/'+job['id'])
            if current['state'] in ['completed','failed','cancelled']:break
            time.sleep(.25)
        assert current['state']=='completed',current
        report=api(f"/api/workspace/history/{result['id']}/report")
        Path(state_file).write_text(json.dumps({'captureId':capture['id'],'analysisId':result['id'],'jobId':job['id'],
                                              'modelHash':report['modelHash'],'probabilities':result['probabilities']}))
    state=json.loads(Path(state_file).read_text())
    overview=api('/api/workspace/captures/'+state['captureId'])
    assert overview['metadata']['sampleCount']==1024
    window=api(f"/api/workspace/captures/{state['captureId']}/window?start=128")
    assert len(window['samples'])==128
    report=api(f"/api/workspace/history/{state['analysisId']}/report")
    assert report['modelHash']==state['modelHash'] and report['probabilities']==state['probabilities']
    assert request(f"/api/workspace/history/{state['analysisId']}/report?format=pdf").startswith(b'%PDF')
    batch=api(f"/api/workspace/jobs/{state['jobId']}/result")
    assert batch['total']==4 and len(batch['predictions'])==4
    assert sum(batch['classCounts'].values())==4
    assert all(p['preprocessing']['windowLength']==128 for p in batch['predictions'])
    return {'protectedApi':True,'managedCapture':True,'batchWindows':4,'pdf':True,'persistedAfterRecreation':verify_existing,
            'modelHash':state['modelHash']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--state-file',required=True)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(args.url,args.state_file,args.verify_existing),indent=2))
