"""Verify a running deployment using only the standard library and real API calls."""
import argparse
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def check(base_url: str, wait_seconds: float = 0) -> dict:
    base_url = base_url.rstrip('/')
    if urlsplit(base_url).scheme not in ('http', 'https'):
        raise ValueError('Use an HTTP or HTTPS application URL.')

    def request(path, data=None, content_type='application/json'):
        payload=json.dumps(data).encode() if data is not None and content_type=='application/json' else data
        req=Request(base_url+path,data=payload,headers={'Content-Type':content_type} if payload is not None else {})
        with urlopen(req,timeout=30) as response:
            return response.read(),response.headers.get('Content-Type','')

    def api(path, data=None):
        raw,content_type=request(path,data)
        assert 'application/json' in content_type, f'{path}: expected JSON'
        return json.loads(raw)

    deadline=time.monotonic()+wait_seconds
    while True:
        try:
            health=api('/api/health')
            assert health['connected'] is True,health
            break
        except (URLError, OSError, AssertionError):
            if time.monotonic()>=deadline:raise
            time.sleep(.5)

    html,_=request('/')
    assert b'<div id="root">' in html,'Compiled React application is missing.'
    assets=re.findall(r'(?:src|href)="(/assets/[^"]+)"',html.decode())
    assert assets,'Compiled frontend assets are missing.'
    for asset in assets:request(asset)
    for path in ['/classifier','/studio','/evaluation','/arena','/intelligence','/theory','/settings']:
        raw,_=request(path)
        assert b'<div id="root">' in raw,path

    classes={'BPSK','QPSK','16-QAM'}
    def valid_prediction(result):
        assert result['predictedModulation'] in classes
        assert set(result['probabilities'])==classes
        probabilities=list(result['probabilities'].values())
        assert all(math.isfinite(p) and 0<=p<=1 for p in probabilities)
        assert math.isclose(sum(probabilities),1,abs_tol=1e-5)
        assert result['preprocessing']['windowLength']==128
        assert result['analysis']['waveform'] and result['analysis']['spectrum']

    demo=api('/api/samples/42')
    predicted=api('/api/classify',{'samples':demo['samples'],'source':'Deployment smoke: saved demo'})
    valid_prediction(predicted)
    generated=api('/api/signal/generate',{'modulation':'BPSK','numSamples':128,'seed':42,'sampleRate':8000})
    valid_prediction(api('/api/classify',{'samples':generated['samples'],'sampleRate':8000,'source':'Deployment smoke: generated signal'}))

    boundary='satmod-deployment-check-boundary'
    csv='I,Q\n'+'\n'.join(f"{p['i']:.12g},{p['q']:.12g}" for p in demo['samples'])
    def upload(contents):
        payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="check.csv"\r\nContent-Type: text/csv\r\n\r\n{contents}\r\n--{boundary}--\r\n').encode()
        raw,_=request('/api/classify/upload',payload,f'multipart/form-data; boundary={boundary}')
        return json.loads(raw)
    uploaded=upload(csv)
    valid_prediction(uploaded)
    assert max(abs(uploaded['probabilities'][c]-predicted['probabilities'][c]) for c in classes)<1e-5
    try:
        upload('I,Q\nnan,nan\n')
        raise AssertionError('Malformed upload was accepted.')
    except HTTPError as exc:
        assert exc.code==422,exc.code

    evaluation=api('/api/evaluation')
    matrix=evaluation['confusionMatrix']
    count=sum(sum(row) for row in matrix)
    assert count==evaluation['metrics']['samples']
    assert math.isclose(sum(matrix[i][i] for i in range(3))/count,evaluation['metrics']['accuracy'])
    try:
        request('/api/does-not-exist')
        raise AssertionError('Unknown API route returned a successful response.')
    except HTTPError as exc:
        assert exc.code==404,exc.code
    return {'url':base_url,'health':health,'spaRoutes':8,'assets':len(assets),'realClassification':predicted['predictedModulation'],
            'generatedSignalClassification':True,'validUploadMatchesDirect':True,'invalidUpload':422,
            'evaluationSamples':count,'unknownApi':404}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--wait-seconds',type=float,default=0)
    args=parser.parse_args()
    print(json.dumps(check(args.url,args.wait_seconds),indent=2))
