"""Genuine explanation/comparison contracts, using explicitly synthetic fixtures."""
import hashlib
import io
import json
from uuid import uuid4
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services import classifier, operations, saved_explanations, workspace_store as store


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR',str(tmp_path/'workspace'))
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN',raising=False)
    monkeypatch.delenv('SATMOD_PUBLIC',raising=False)
    with TestClient(app) as session:yield session
    runner=operations.RUNNERS.pop(str(store.root()),None)
    if runner:runner[0].shutdown(wait=True)
    store.close_connections()


def raw(n=1024,seed=591):return (np.random.default_rng(seed).normal(size=(n,2))*.006).astype('<f4')


def imported(client,values=None,fs=8000,offset=0):
    stream=io.BytesIO();np.save(stream,raw() if values is None else values,allow_pickle=False)
    options={'sessionId':'synthetic-research-fixture','source':'Generated test recording, not satellite RF','sampleOffset':offset}
    if fs is not None:options['sampleRate']=fs
    response=client.post('/api/workspace/captures',data={'options':json.dumps(options)},files={'file':('synthetic.npy',stream.getvalue())})
    assert response.status_code==200,response.text
    return response.json()


def classified(client,capture,start=0):
    response=client.post('/api/workspace/captures/'+capture['id']+f'/classify?start={start}')
    assert response.status_code==200,response.text
    return response.json()


def explained(client,analysis,target='QPSK',method='both'):
    response=client.post('/api/research/analyses/'+analysis['id']+'/explanations',json={'method':method,'targetClass':target,'steps':16})
    assert response.status_code==200,response.text
    return response.json()


def comparison(client,a,b,**kwargs):
    response=client.post('/api/research/comparisons',json={'first':{'captureId':a['id']},'second':{'captureId':b['id']},**kwargs})
    assert response.status_code==200,response.text
    return response.json()


def test_managed_explanation_is_real_signed_persistent_and_in_history_pdf(client):
    capture=imported(client);analysis=classified(client,capture,128)
    result=explained(client,analysis)
    assert result['modelHash']==store.artifact_hash() and result['inputHash']==saved_explanations.sample_hash(raw()[128:256])
    assert result['prediction']['probabilities']==analysis['probabilities']
    assert len(result['explanations'])==2
    for explanation in result['explanations']:
        assert len(explanation['samples'])==128 and explanation['targetClass']=='QPSK'
        assert explanation['targetQuantity']=='pre-softmax class logit'
        values=np.array([[p['i'],p['q']] for p in explanation['samples']])
        np.testing.assert_array_equal([[p['rawI'],p['rawQ']] for p in explanation['samples']],raw()[128:256])
        for i,channel in enumerate(['i','q']):
            totals=explanation['diagnostics']['channels'][channel]
            assert totals['positiveSum']==pytest.approx(np.maximum(values[:,i],0).sum())
            assert totals['negativeSum']==pytest.approx(np.minimum(values[:,i],0).sum())
        assert sum(r['endSample']-r['startSample'] for r in explanation['diagnostics']['importantRegions'])<=16
        if explanation['method']=='integrated-gradients':
            assert explanation['attributionSum']==pytest.approx(values.sum())
            assert explanation['completenessError']==pytest.approx(values.sum()-explanation['logitDifference'])
        else:assert explanation['attributionSum'] is None and explanation['completenessError'] is None
    assert result['methodComparison']['status']=='available'
    assert -1<=result['methodComparison']['spearmanRho']<=1
    qpsk_comparison=result['methodComparison']
    result=explained(client,analysis,target='BPSK')
    assert len(result['explanations'])==4 and len(result['methodComparisons'])==2
    assert result['methodComparison']['targetClass']=='BPSK'
    assert next(c for c in result['methodComparisons'] if c['targetClass']=='QPSK')==qpsk_comparison
    store.close_connections()
    persisted=client.get('/api/research/analyses/'+analysis['id']+'/explanations').json()
    assert persisted==result
    history=client.get('/api/workspace/history').json()['items'][0]
    assert history['explanationCount']==4
    report=client.get('/api/workspace/history/'+analysis['id']+'/report').json()
    assert report['explanations']==result['explanations']
    pdf=client.get('/api/workspace/history/'+analysis['id']+'/report?format=pdf').content
    assert pdf.startswith(b'%PDF') and pdf.rstrip().endswith(b'%%EOF')
    assert b'SIGNED I/Q CLASS-LOGIT ATTRIBUTION' in pdf and b'completenessError' in pdf
    assert result['modelHash'].encode() in pdf
    assert pdf==client.get('/api/workspace/history/'+analysis['id']+'/report?format=pdf').content


def test_legacy_explanation_requires_exact_original_samples(client):
    values=raw(256)
    analysis=classifier.classify(values,'Explicit synthetic legacy fixture',8000,64)
    path='/api/research/analyses/'+analysis['id']+'/explanations'
    assert client.post(path,json={'method':'saliency'}).status_code==409
    wrong=values.copy();wrong[0,0]+=.01
    rows=lambda x:[{'i':float(i),'q':float(q)} for i,q in x]
    assert client.post(path,json={'method':'saliency','samples':rows(wrong)}).status_code==409
    response=client.post(path,json={'method':'saliency','samples':rows(values)})
    assert response.status_code==200,response.text
    explanation=response.json()['explanations'][0]
    assert explanation['windowStart']==64 and explanation['samples'][0]['sample']==64
    assert explanation['method']=='saliency' and explanation['completenessError'] is None


@pytest.mark.parametrize('field,value',[('modelHash','different-checkpoint'),('inputHash','wrong-input'),('preprocessing',{'mean':2,'std':3,'windowStart':0})])
def test_historical_identity_mismatch_never_explains_new_inputs(client,field,value):
    capture=imported(client);analysis=classified(client,capture)
    report=saved_explanations.load_analysis(analysis['id']);report[field]=value
    store.file_path(analysis['id'],'json').write_text(json.dumps(report))
    response=client.post('/api/research/analyses/'+analysis['id']+'/explanations',json={'method':'saliency'})
    assert response.status_code==409
    assert not client.get('/api/research/analyses/'+analysis['id']+'/explanations').json()['explanations']


def test_explanation_options_and_capacity_are_bounded(client):
    capture=imported(client);analysis=classified(client,capture);path='/api/research/analyses/'+analysis['id']+'/explanations'
    for options in [{'method':'invented'},{'steps':257},{'targetClass':'8-PSK'},{'samples':[{'i':0.,'q':0.}]*127}]:
        assert client.post(path,json=options).status_code==422
    saved_explanations.SLOTS.acquire();saved_explanations.SLOTS.acquire()
    try:assert client.post(path,json={'method':'saliency'}).status_code==429
    finally:saved_explanations.SLOTS.release();saved_explanations.SLOTS.release()
    assert not list(store.root().glob('*.pending'))


def test_deleted_capture_requires_matching_original_input_and_history_delete_is_safe(client):
    values=raw();capture=imported(client,values);analysis=classified(client,capture)
    assert client.delete('/api/workspace/captures/'+capture['id']).status_code==200
    path='/api/research/analyses/'+analysis['id']+'/explanations'
    assert client.post(path,json={'method':'saliency'}).status_code==409
    assert client.post(path,json={'method':'saliency','samples':[{'i':float(i),'q':float(q)} for i,q in values[:128]]}).status_code==200
    store.file_path(analysis['id'],'pending').write_bytes(b'interrupted report write')
    assert client.delete('/api/workspace/history/'+analysis['id']).status_code==200
    assert not store.file_path(analysis['id'],'json').exists() and not store.file_path(analysis['id'],'pending').exists()


def test_managed_comparison_uses_correct_normalized_psd_and_declares_clock_mismatch(client):
    values=raw();a=imported(client,values,8000);b=imported(client,values,16000)
    report=comparison(client,a,b)
    assert report['ratesEqual'] is False and report['normalizedAxes'] is True
    assert report['sampleMse'] is None and report['sampleClocksCompatible'] is False
    assert report['sampleRates']=={'first':8000.,'second':16000.}
    assert report['plots']['frequencyUnit']=='cycles/sample' and report['plots']['timeUnit']=='Sample index'
    np.testing.assert_allclose([p['frequency'] for p in report['plots']['first']['spectrum']],
                               [p['frequency'] for p in report['plots']['second']['spectrum']])
    np.testing.assert_allclose([p['psd'] for p in report['plots']['first']['spectrum']],
                               [p['psd'] for p in report['plots']['second']['spectrum']],atol=1e-8)
    for name in ['first','second']:
        assert report[name]['prediction']['status']=='available'
        assert report[name]['prediction']['probabilities']==classifier.predict_window(values)['probabilities']
        assert sum(p['fraction'] for p in report['plots']['amplitudeDistribution'][name])==pytest.approx(1)
    assert report['attributionDifferences']['status']=='unavailable'


def test_valid_history_predictions_and_signed_attribution_are_reused(client,monkeypatch):
    capture=imported(client);a=classified(client,capture,0);b=classified(client,capture,128)
    ea=explained(client,a,method='saliency');eb=explained(client,b,method='saliency')
    def no_redundant_inference(*args,**kwargs):raise AssertionError('Identity-checked saved predictions should be reused')
    monkeypatch.setattr(classifier,'predict_window',no_redundant_inference)
    report=comparison(client,capture,capture,second={'captureId':capture['id'],'start':128},firstAnalysisId=a['id'],secondAnalysisId=b['id'])
    assert report['first']['prediction']['predictionSource']=='stored matching analysis'
    assert report['second']['prediction']['probabilities']==b['probabilities']
    difference=report['attributionDifferences']['comparisons'][0]
    assert difference['method']=='saliency' and difference['targetClass']=='QPSK'
    actual=np.array([[s['i'],s['q']] for s in difference['samples']])
    expected=np.array([[s['i'],s['q']] for s in ea['explanations'][0]['samples']])-np.array([[s['i'],s['q']] for s in eb['explanations'][0]['samples']])
    np.testing.assert_array_equal(actual,expected)
    wrong=client.post('/api/research/comparisons',json={'first':{'captureId':capture['id'],'start':256},'second':{'captureId':capture['id']},'firstAnalysisId':a['id']})
    assert wrong.status_code==409


def test_identical_windows_do_not_repeat_inference_and_exports_survive_raw_deletion(client,monkeypatch):
    capture=imported(client);calls=[];original=classifier.predict_window
    def tracked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(classifier,'predict_window',tracked)
    report=comparison(client,capture,capture)
    assert len(calls)==1 and report['sampleMse']==0 and report['confidenceDifference']==0
    assert client.get('/api/research/comparisons?search=Generated&limit=1').json()['total']==1
    assert client.delete('/api/workspace/captures/'+capture['id']).status_code==200
    store.close_connections()
    path='/api/research/comparisons/'+report['id']
    assert client.get(path).json()==report
    assert client.get(path+'/export').json()==report
    pdf=client.get(path+'/export?format=pdf').content
    assert pdf.startswith(b'%PDF') and report['modelHash'].encode() in pdf
    assert b'SPECTRAL DENSITY COMPARISON' in pdf and b'No sample alignment or resampling' in pdf
    assert pdf==client.get(path+'/export?format=pdf').content
    assert client.delete(path).status_code==200
    assert client.get(path).status_code==404


def test_comparison_long_windows_are_bounded_and_raw_files_not_deleted_with_report(client):
    values=raw(65536);capture=imported(client,values,offset=64)
    report=comparison(client,capture,capture,first={'captureId':capture['id'],'start':128,'length':32768})
    assert report['first']['position']['sourceSample']==192
    expected=hashlib.sha256(values[192:192+32768].tobytes()).hexdigest()
    assert report['first']['inputHash']==expected
    for name in ['first','second']:
        assert len(report['plots'][name]['waveform'])<=1024 and len(report['plots'][name]['spectrum'])<=1024
        assert 'samples' not in report[name]
    assert client.delete('/api/research/comparisons/'+report['id']).status_code==200
    assert store.file_path(capture['id'],'iq').exists()
    bad=client.post('/api/research/comparisons',json={'first':{'captureId':capture['id'],'length':65537},'second':{'captureId':capture['id']}})
    assert bad.status_code==422


def test_atomic_document_failure_preserves_original_history_and_cleans_pending(client,monkeypatch):
    capture=imported(client);analysis=classified(client,capture)
    original=store.file_path(analysis['id'],'json').read_bytes();summary=store.get('analysis',analysis['id'])
    def fail(*args,**kwargs):raise RuntimeError('Controlled database failure')
    monkeypatch.setattr(store,'put',fail)
    with pytest.raises(RuntimeError,match='Controlled'):
        store.save_document('analysis',summary,{'id':analysis['id'],'syntheticTest':True})
    assert store.file_path(analysis['id'],'json').read_bytes()==original
    assert not list(store.root().glob('*.pending'))


def test_stale_atomic_temporary_files_are_cleaned_on_first_connection(client):
    capture=imported(client);store.close_connections()
    orphan=store.file_path(str(uuid4()),'pending');orphan.write_bytes(b'partial JSON from prior process')
    store.get('capture',capture['id'])
    assert not orphan.exists() and store.file_path(capture['id'],'iq').exists()


def test_new_research_operations_respect_existing_workspace_access(client,monkeypatch):
    monkeypatch.setenv('SATMOD_PUBLIC','1');monkeypatch.setenv('SATMOD_ACCESS_TOKEN','test-research-key')
    for path in ['/api/research/comparisons','/api/research/analyses/'+str(uuid4())+'/explanations']:
        assert client.get(path).status_code==401
    assert client.get('/api/research/comparisons',headers={'X-Satmod-Token':'test-research-key'}).status_code==200


def test_resolved_root_cache_preserves_dynamic_environment_relative_paths_and_initialization(tmp_path,monkeypatch):
    first=tmp_path/'first';second=tmp_path/'second'
    first.mkdir();second.mkdir()
    monkeypatch.setenv('SATMOD_DATA_DIR','managed')
    monkeypatch.chdir(first);a=store.root()
    assert a==first/'managed' and a.is_dir()
    assert store.root() is a
    monkeypatch.chdir(second);b=store.root()
    assert b==second/'managed' and b!=a and b.is_dir()
    monkeypatch.setenv('SATMOD_DATA_DIR',str(first/'absolute'));c=store.root()
    assert c==first/'absolute' and c.is_dir()
    store.close_connections()
    c.rmdir()
    assert store.root()==c and c.is_dir()


def test_sigmf_absolute_offset_explanations_reconstruct_file_samples_and_comparison_boundaries(client):
    values=raw(512)
    metadata={'global':{'core:datatype':'cf32_le','core:sample_rate':8000,'core:offset':1000},
              'captures':[{'core:sample_start':1000,'core:frequency':137e6},
                          {'core:sample_start':1200,'core:frequency':138e6}], 'annotations':[]}
    response=client.post('/api/workspace/captures',data={'options':json.dumps({'format':'sigmf','sessionId':'synthetic-offset-report','sampleOffset':16})},
                         files={'file':('synthetic.sigmf-data',values.tobytes()),'metadata':('synthetic.sigmf-meta',json.dumps(metadata).encode())})
    assert response.status_code==200,response.text
    capture=response.json();analysis=classified(client,capture,64)
    assert analysis['capturePosition']['sourceSample']==1080
    result=explained(client,analysis,method='saliency')
    np.testing.assert_array_equal([[p['rawI'],p['rawQ']] for p in result['explanations'][0]['samples']],values[80:208])
    compared=comparison(client,capture,capture,first={'captureId':capture['id'],'start':64,'length':256})
    assert compared['first']['position']['captureBoundariesWithinWindow']==[1200]
