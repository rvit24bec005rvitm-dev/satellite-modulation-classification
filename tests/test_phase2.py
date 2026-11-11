"""Phase 2 scientific/API tests; all generated fixtures are explicitly synthetic."""
import io
import json
import time
import threading
import hashlib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services import workspace_store as store, captures, operations, classifier
from backend.services.advanced_dsp import diagnostics


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR',str(tmp_path/'managed'))
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN',raising=False);monkeypatch.delenv('SATMOD_PUBLIC',raising=False)
    with TestClient(app) as c:yield c
    entry=operations.RUNNERS.pop(str(store.root()),None)
    if entry:entry[0].shutdown(wait=True,cancel_futures=False)


def synthetic(n=2048):
    rng=np.random.default_rng(121)
    return (rng.normal(size=(n,2))*.006).astype(np.float32)


def upload(c,iq=None,options=None,metadata=None,payload=None,name='recording.npy'):
    if payload is None:
        buffer=io.BytesIO();np.save(buffer,synthetic() if iq is None else iq,allow_pickle=False);payload=buffer.getvalue()
    files={'file':(name,payload)}
    if metadata is not None:files['metadata']=('recording.sigmf-meta',json.dumps(metadata).encode())
    return c.post('/api/workspace/captures',data={'options':json.dumps({'sessionId':'synthetic-test-session',**(options or {})})},files=files)


def wait(c,job):
    for _ in range(200):
        current=c.get('/api/workspace/jobs/'+job['id']).json()
        if current['state'] in ['completed','failed','cancelled']:return current
        time.sleep(.02)
    raise AssertionError('Job did not terminate')


@pytest.mark.parametrize('datatype',['ci8','ci16_le','ci16_be','ci32_le','ci32_be','cf32_le','cf32_be','cf64_le','cf64_be'])
@pytest.mark.parametrize('layout',['interleaved','planar'])
def test_raw_formats_endianness_scaling_offsets(client,datatype,layout):
    raw=np.arange(512).reshape(256,2)%13
    stream=raw[:,::-1] if layout=='interleaved' else raw[:,::-1].T
    result=upload(client,payload=stream.astype(captures.DTYPES[datatype]).tobytes(),name='unknown.bin',options={
        'format':'raw','datatype':datatype,'layout':layout,'iqOrder':'QI','scale':.001,'sampleRate':16000,'sampleOffset':64,'sampleCount':128})
    assert result.status_code==200,result.text
    record=result.json();response=client.get(f"/api/workspace/captures/{record['id']}/window").json()
    np.testing.assert_allclose([[p['i'],p['q']] for p in response['samples']],raw[64:192]*.001,atol=1e-8)
    assert record['sourceOffset']==64 and record['sampleCount']==128


def test_sigmf_timestamps_hash_and_metadata(client):
    x=synthetic();data=x.astype('<f4').tobytes()
    meta={'global':{'core:datatype':'cf32_le','core:sample_rate':8000,'core:description':'Synthetic fixture, not satellite RF','core:sha512':hashlib.sha512(data).hexdigest()},
        'captures':[{'core:sample_start':0,'core:frequency':137e6,'core:datetime':'2025-01-01T00:00:00Z'},{'core:sample_start':128,'core:frequency':138e6,'core:datetime':'2025-01-01T00:00:01Z'}],'annotations':[]}
    response=upload(client,payload=data,name='test.sigmf-data',options={'format':'sigmf'},metadata=meta)
    assert response.status_code==200,response.text
    record=response.json();assert record['sigmf']==meta
    w=client.get(f"/api/workspace/captures/{record['id']}/window?start=256").json()
    assert w['position']['timestamp']=='2025-01-01T00:00:01.016000+00:00'
    assert w['position']['centerFrequency']==138e6


@pytest.mark.parametrize('change',[{'core:datatype':'cu8'},{'core:num_channels':2},{'core:sample_rate':None},{'core:sha512':'bad'},{'core:dataset':'../../secret.bin'}])
def test_sigmf_invalid_metadata_safe_cleanup(client,change):
    meta={'global':{'core:datatype':'cf32_le','core:sample_rate':8000,**change},'captures':[]}
    response=upload(client,payload=synthetic().tobytes(),name='test.sigmf-data',options={'format':'sigmf'},metadata=meta)
    assert response.status_code==422,response.text
    assert not list(store.root().glob('*.iq')) and not list(store.root().glob('*.upload'))


@pytest.mark.parametrize('options',[{'format':'raw'}, {'format':'raw','datatype':'cf32_le'}, {'format':'raw','datatype':'ri16_le','sampleRate':8000}, {'sampleOffset':999999}, {'scale':0}])
def test_import_requires_explicit_valid_metadata(client,options):assert upload(client,options=options).status_code==422


@pytest.mark.parametrize('kind',['complex','transposed','real'])
def test_long_npy_mapped_windowing(client,kind):
    raw=synthetic(100000)
    value=raw[:,0]+1j*raw[:,1] if kind=='complex' else raw.T if kind=='transposed' else raw
    response=upload(client,iq=value)
    assert response.status_code==200,response.text
    rid=response.json()['id'];overview=client.get('/api/workspace/captures/'+rid).json()
    assert len(overview['points'])<=512
    assert client.get(f'/api/workspace/captures/{rid}/window?start=99900').status_code==422
    window=client.get(f'/api/workspace/captures/{rid}/window?start=99872').json()
    np.testing.assert_array_equal([[p['i'],p['q']] for p in window['samples']],raw[-128:])
    assert client.get(f'/api/workspace/captures/{rid}/window?length=65537').status_code==422


def test_batch_real_inference_matches_single_windows(client):
    raw=synthetic();rid=upload(client,iq=raw).json()['id']
    job=client.post('/api/workspace/jobs/batch',json={'captureId':rid,'windows':4,'stride':64}).json()
    assert wait(client,job)['state']=='completed'
    result=client.get(f"/api/workspace/jobs/{job['id']}/result").json()
    assert result['total']==4 and sum(result['classCounts'].values())==4
    for i,p in enumerate(result['predictions']):
        expected=classifier.predict_window(raw,i*64)
        assert p['probabilities']==expected['probabilities']
        assert p['start']==i*64 and p['timestamp'] is None
    assert client.get(f"/api/workspace/jobs/{job['id']}/result?offset=2&limit=1").json()['predictions'][0]['index']==2
    assert client.post('/api/workspace/jobs/batch',json={'captureId':rid,'windows':1000}).status_code==422
    assert client.delete(f"/api/workspace/jobs/{job['id']}").status_code==200
    assert client.get(f"/api/workspace/jobs/{job['id']}/result").status_code==404


def test_cancel_capacity_and_capture_deletion_protection(client,monkeypatch):
    rid=upload(client).json()['id'];entered=threading.Event();release=threading.Event()
    real=classifier.predict_window
    def slow(*args,**kwargs):entered.set();release.wait(3);return real(*args,**kwargs)
    monkeypatch.setattr(classifier,'predict_window',slow)
    jobs=[client.post('/api/workspace/jobs/batch',json={'captureId':rid,'windows':2}).json() for _ in range(4)]
    assert entered.wait(1)
    assert client.post('/api/workspace/jobs/batch',json={'captureId':rid}).status_code in [422,429]
    assert client.delete('/api/workspace/captures/'+rid).status_code==409
    for job in jobs:assert client.post('/api/workspace/jobs/'+job['id']+'/cancel').json()['cancelRequested']
    release.set()
    assert all(wait(client,j)['state']=='cancelled' for j in jobs)
    assert client.delete('/api/workspace/captures/'+rid).status_code==200


def test_job_failure_and_restart_recovery(client,monkeypatch):
    rid=upload(client).json()['id']
    def fail(*args,**kwargs):raise RuntimeError('/private/server/path must not leak')
    monkeypatch.setattr(classifier,'predict_window',fail)
    job=client.post('/api/workspace/jobs/batch',json={'captureId':rid,'windows':1}).json()
    failed=wait(client,job);assert failed['state']=='failed' and '/private' not in failed['error']
    operations.RUNNERS.pop(str(store.root()))[0].shutdown(wait=True)
    interrupted=store.put('job',{'state':'running','params':{},'kind':'batch','progress':.5,'cancelRequested':False})
    recovered=client.get('/api/workspace/jobs/'+interrupted['id']).json()
    assert recovered['state']=='failed' and 'restart' in recovered['error']


def test_analysis_restart_history_exports_and_delete(client,monkeypatch):
    saved=[];real=store.save_analysis
    def save(*args,**kwargs):saved.append((args,kwargs));return real(*args,**kwargs)
    monkeypatch.setattr(store,'save_analysis',save)
    rid=upload(client).json()['id'];classified=client.post('/api/workspace/captures/'+rid+'/classify?start=128').json()
    assert classified['predictedModulation'] in ['BPSK','QPSK','16-QAM']
    assert len(saved)==1 and saved[0][0][1]==rid
    with TestClient(app) as restarted:
        assert restarted.get('/api/workspace/history?search=External%20recording').json()['total']==1
        report=restarted.get(f"/api/workspace/history/{classified['id']}/report").json()
        assert report['modelHash']==store.artifact_hash() and report['capturePosition']['sourceSample']==128
        assert report['reportVersion']=='satmod-analysis-v2'
        assert restarted.get(f"/api/workspace/history/{classified['id']}/report?format=pdf").content.startswith(b'%PDF')
        assert b'predictedModulation' in restarted.get(f"/api/workspace/history/{classified['id']}/report?format=csv").content
    assert client.delete('/api/workspace/history/'+classified['id']).status_code==200
    assert client.get('/api/workspace/history').json()['total']==0
    assert client.get(f"/api/workspace/history/{classified['id']}/report").status_code==404


def test_persistence_in_new_interpreter_and_schema_version(client):
    import subprocess,sys,os,sqlite3
    rid=upload(client).json()['id']
    result=subprocess.run([sys.executable,'-c','from backend.services.workspace_store import listing;print(listing("capture")["total"])'],env=os.environ.copy(),capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='1'
    assert client.delete('/api/workspace/captures/'+rid.upper()).status_code==200
    assert store.listing('capture')['total']==0
    with sqlite3.connect(store.root()/'workspace.sqlite3') as db:
        assert db.execute('PRAGMA user_version').fetchone()[0]==1


def test_oversized_and_renamed_archives_are_rejected(client,monkeypatch):
    monkeypatch.setattr(captures,'IMPORT_LIMIT',100)
    assert upload(client).status_code==413
    monkeypatch.setattr(captures,'IMPORT_LIMIT',128*1024**2)
    buffer=io.BytesIO();np.savez(buffer,iq=synthetic())
    assert upload(client,payload=buffer.getvalue(),name='renamed.npy').status_code==422


def test_dataset_evaluation_no_inferred_labels_or_snr(client):
    rid=upload(client).json()['id']
    manifest={'name':'Synthetic labeled fixture','source':'Unit-test generator, not actual RF','license':'Test only',
              'entries':[{'captureId':rid,'start':i*128,'label':label,'split':'test'} for i,label in enumerate(['BPSK','QPSK','16-QAM'])]}
    response=client.post('/api/workspace/datasets',json=manifest);assert response.status_code==200,response.text
    dataset=response.json();job=client.post('/api/workspace/datasets/'+dataset['id']+'/evaluate').json()
    assert wait(client,job)['state']=='completed'
    result=client.get('/api/workspace/jobs/'+job['id']+'/result').json()
    assert result['metrics']['samples']==3 and result['metrics']['snrAccuracy']==[]
    assert result['dataset']['hash']==dataset['hash']
    assert [p['groundTruth'] for p in result['predictions']]==['BPSK','QPSK','16-QAM']
    assert result['classOrder']==['BPSK','QPSK','16-QAM']
    exported=client.get('/api/workspace/jobs/'+job['id']+'/export').json()
    assert exported['dataset']['entries']==dataset['entries']
    assert b'groundTruth' in client.get('/api/workspace/jobs/'+job['id']+'/export?format=csv').content
    assert sum(map(sum,result['metrics']['confusionMatrix']))==3
    assert client.delete('/api/workspace/captures/'+rid).status_code==409
    assert client.delete('/api/workspace/datasets/'+dataset['id']).status_code==200
    assert client.delete('/api/workspace/captures/'+rid).status_code==200


def test_dataset_leakage_and_demo_restriction(client):
    rid=upload(client).json()['id']
    base={'name':'Leaks','source':'Fixture','license':'Test'}
    entries=[{'captureId':rid,'start':0,'label':'BPSK','split':'train'},{'captureId':rid,'start':64,'label':'BPSK','split':'test'}]
    assert client.post('/api/workspace/datasets',json={**base,'entries':entries}).status_code==422
    # Rewrapped duplicate capture with a different declared session still cannot cross splits.
    other=upload(client,options={'sessionId':'other'}).json()['id']
    entries[1].update(captureId=other,start=0)
    assert client.post('/api/workspace/datasets',json={**base,'entries':entries}).status_code==422
    demo=upload(client,iq=classifier.dataset()['signals'][42]).json()['id']
    assert client.post('/api/workspace/datasets',json={**base,'entries':[{'captureId':demo,'start':0,'label':'BPSK','split':'test'}]}).status_code==422


def test_scientific_diagnostics_cfo_clipping_and_activity(client):
    fs=8000;n=np.arange(1024);z=.1*np.exp(2j*np.pi*125*n/fs);iq=np.column_stack((z.real,z.imag))
    d=diagnostics(iq,fs,'BPSK',.09)
    assert d['frequencyOffset']['estimateHz']==pytest.approx(125,abs=1e-6)
    assert d['clippedFraction']>0 and d['iqGainImbalanceDb']==pytest.approx(0,abs=1e-6)
    assert np.sum([p['density'] for p in d['welch']])*fs/len(d['welch'])==pytest.approx(.01,rel=.02)
    assert diagnostics(iq,fs)['frequencyOffset'] is None
    rid=upload(client,iq=np.vstack([np.zeros((256,2)),iq[:256],np.zeros((256,2))]),options={'sampleRate':fs}).json()['id']
    activity=client.get(f'/api/workspace/captures/{rid}/activity?threshold=.005').json()
    assert activity['segments'][0]['startSample']==256 and activity['segments'][0]['endSample']==512
    assert client.get(f'/api/workspace/captures/{rid}/domain-shift').status_code==200


def test_access_limits_paths_and_cleanup(client,monkeypatch):
    monkeypatch.setenv('SATMOD_PUBLIC','1')
    assert client.get('/api/workspace/captures').status_code==503
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN','test-only-key')
    assert client.get('/api/workspace/captures').status_code==401
    assert client.get('/api/health').status_code==200
    assert client.get('/api/workspace/captures',headers={'X-Satmod-Token':'test-only-key'}).status_code==200
    denied=client.delete('/api/workspace/history',headers={'Origin':'http://localhost:5173'})
    assert denied.status_code==401 and denied.headers['access-control-allow-origin']=='http://localhost:5173'
    monkeypatch.delenv('SATMOD_PUBLIC');monkeypatch.delenv('SATMOD_ACCESS_TOKEN')
    assert client.get('/api/workspace/captures/not-a-uuid/window').status_code==422
    assert client.get('/api/workspace/captures/00000000-0000-0000-0000-000000000000/window').status_code==404
    monkeypatch.setenv('SATMOD_STORAGE_BYTES','32')
    assert upload(client).status_code==422
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))


def test_reencoded_overlapping_windows_cannot_cross_splits(client):
    raw=synthetic();a=upload(client,iq=raw,options={'sessionId':'declared-session-a'}).json()['id']
    b=upload(client,payload=raw.astype('<f4').tobytes(),name='reencoded.bin',options={'format':'raw','datatype':'cf32_le','sampleRate':8000,'sessionId':'declared-session-b'}).json()['id']
    manifest={'name':'Contaminated','source':'Synthetic fixture','license':'Test','entries':[{'captureId':a,'start':0,'split':'train','label':'BPSK'},{'captureId':b,'start':64,'split':'test','label':'BPSK'}]}
    response=client.post('/api/workspace/datasets',json=manifest)
    assert response.status_code==422 and 'Overlapping' in response.text


def test_capture_delete_checks_all_dataset_pages_and_uuid_case(client):
    rid=upload(client).json()['id']
    reference=store.put('dataset',{'createdAt':'2000-01-01T00:00:00Z','entries':[{'captureId':rid.upper()}]})
    for _ in range(200):store.put('dataset',{'entries':[]})
    assert reference['id'] not in [d['id'] for d in store.listing('dataset',200)['items']]
    assert client.delete('/api/workspace/captures/'+rid.upper()).status_code==409
    assert store.file_path(rid,'iq').exists()


def test_non_npy_import_rejects_before_unbounded_parser_read(client,monkeypatch):
    import backend.core.config as config
    monkeypatch.setattr(config,'MAX_UPLOAD_BYTES',64)
    response=upload(client,payload=b'I,Q\n'+b'0.1,0.2\n'*128,name='large.csv')
    assert response.status_code==422 and 'bounded upload limit' in response.text
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))


def test_chunked_import_size_limit_and_request_id(client,monkeypatch):
    import asyncio,httpx
    monkeypatch.setenv('SATMOD_CAPTURE_UPLOAD_BYTES','4')
    async def run():
        async def chunks():
            yield b'x'*70000
            yield b'x'*70000
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as session:
            return await session.post('/api/workspace/captures',content=chunks(),headers={'Origin':'http://localhost:5173'})
    response=asyncio.run(run())
    assert response.status_code==413
    assert response.headers['access-control-allow-origin']=='http://localhost:5173'
    assert len(response.headers['x-request-id'])==36


def test_slow_request_bodies_time_out_and_release_import_slots(client,monkeypatch):
    import asyncio,httpx
    monkeypatch.setenv('SATMOD_BODY_TIMEOUT_SECONDS','.01')
    async def run(path):
        async def chunks():
            yield b'partial'
            await asyncio.sleep(.2)
            yield b'incomplete'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as session:
            return await session.post(path,content=chunks())
    for path in ['/api/workspace/captures','/api/workspace/captures','/api/classify']:
        response=asyncio.run(run(path))
        assert response.status_code==408 and 'timed out' in response.text
        assert response.headers['x-request-id']
    monkeypatch.setenv('SATMOD_BODY_TIMEOUT_SECONDS','120')
    assert upload(client).status_code==200
    assert not list(store.root().glob('*.upload'))
