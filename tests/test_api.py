"""API integration and safety tests against the real model and DSP artifacts."""
import io
import json
import zipfile
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.io import savemat, wavfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.classifier import dataset, stats, preprocess, model
from backend.services.signals import parse_capture, validate_iq
from backend.core.config import MAX_UPLOAD_BYTES

@pytest.fixture
def client():
    with TestClient(app) as c: yield c

def capture(): return dataset()['signals'][42]
def sample_rows(a): return [{'i':float(i),'q':float(q)} for i,q in a]
def encoded(kind,a):
    buf=io.BytesIO()
    if kind=='npy':np.save(buf,a,allow_pickle=False)
    elif kind=='npz':np.savez_compressed(buf,iq=a)
    elif kind=='csv':buf.write(('I,Q\n'+'\n'.join(f'{i:.12g},{q:.12g}' for i,q in a)).encode())
    elif kind=='mat':savemat(buf,{'iq':a},do_compression=True)
    elif kind=='wav':wavfile.write(buf,8000,a)
    elif kind=='json':buf.write(json.dumps({'samples':sample_rows(a)}).encode())
    return buf.getvalue()

def test_health_and_registry(client):
    assert client.get('/api/health').json()['connected']
    models=client.get('/api/models').json()
    assert models[0]['status']=='available'
    assert models[0]['parameters']==36355
    assert all(m['status']=='not-trained' and 'accuracy' not in m for m in models[1:])
    assert client.get('/api/system/info').json()['classes']==['BPSK','QPSK','16-QAM']

@pytest.mark.parametrize('kind',['npy','npz','csv','mat','wav','json'])
def test_upload_matches_saved_sample_preprocessing(client,kind):
    raw=capture()
    direct=client.post('/api/classify',json={'samples':sample_rows(raw)}).json()
    response=client.post('/api/classify/upload',files={'file':(f'capture.{kind}',encoded(kind,raw))})
    assert response.status_code==200,response.text
    result=response.json()
    np.testing.assert_allclose(list(result['probabilities'].values()),list(direct['probabilities'].values()),atol=1e-5)
    np.testing.assert_allclose([[p['i'],p['q']] for p in result['samples']],raw,atol=1e-9)
    assert abs(sum(result['probabilities'].values())-1)<1e-5
    assert 0<=result['entropyBits']<=np.log2(3)+1e-6
    assert len(result['analysis']['spectrum'])==128
    assert result['sampleRate']==(8000 if kind=='wav' else 1)

@pytest.mark.parametrize('kind',['npy','npz','mat'])
@pytest.mark.parametrize('layout',['transpose','complex'])
def test_alternative_iq_layouts(kind,layout):
    raw=capture()
    a=raw.T if layout=='transpose' else raw[:,0]+1j*raw[:,1]
    result,_=parse_capture(f'signal.{kind}',encoded(kind,a))
    np.testing.assert_array_equal(result,raw)

@pytest.mark.parametrize('mod',['BPSK','QPSK','16-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK'])
def test_generator_and_analysis(client,mod):
    body={'modulation':mod,'snrDb':18,'numSamples':1024,'seed':123,'rrcEnabled':True}
    response=client.post('/api/signal/generate',json=body)
    assert response.status_code==200,response.text
    signal=response.json()
    assert len(signal['samples'])==1024
    assert signal['analysis']['eye'] is not None
    assert signal['classifierSupported']==(mod in ['BPSK','QPSK','16-QAM'])
    assert signal['analysis']['rmsAmplitude']==pytest.approx(.008584335,rel=1e-5)
    for plot in ['waveform','constellation','spectrum','amplitudeDistribution','phaseDistribution']:
        assert signal['analysis'][plot]
    assert client.post('/api/signal/generate',json=body).json()['samples']==signal['samples']
    if signal['classifierSupported']:
        result=client.post('/api/classify',json={'samples':signal['samples']})
        assert result.status_code==200,result.text
        assert result.json()['predictedModulation'] in ['BPSK','QPSK','16-QAM']
        assert result.json()['preprocessing']['amplitudePolicy']=='Preserve samples; training z-score only'


def test_impairment_controls_change_samples(client):
    base=client.post('/api/signal/generate',json={'sampleRate':1000}).json()['samples']
    for change in [{'dopplerHz':12},{'cfoHz':7},{'phaseNoiseStd':.1},{'ricianEnabled':True,'ricianK':0},{'twtaDrive':2.1},{'rrcAlpha':.8},{'rrcEnabled':False},{'snrDb':-10}]:
        r=client.post('/api/signal/generate',json={'sampleRate':1000,**change})
        assert r.status_code==200,r.text
        assert base!=r.json()['samples']


def test_evaluation_mathematically_consistent(client):
    r=client.get('/api/evaluation').json()
    cm=np.array(r['confusionMatrix'])
    assert r['metrics']['accuracy']==pytest.approx(cm.trace()/cm.sum())
    assert r['metrics']['samples']==900
    assert sum(p['support'] for p in r['perClass'])==900
    for i,p in enumerate(r['perClass']):
        assert p['precision']==pytest.approx(cm[i,i]/cm[:,i].sum())
        assert p['recall']==pytest.approx(cm[i,i]/cm[i,:].sum())
    snr=client.get('/api/evaluation/snr').json()
    assert len(snr)==20
    assert sum(p['samples'] for p in snr)==900
    assert np.mean([p['accuracy'] for p in snr])==pytest.approx(r['metrics']['accuracy'],abs=5e-5)


def test_benchmark_is_measured(client):
    r=client.post('/api/benchmark',json={'iterations':5})
    assert r.status_code==200
    result=r.json()
    assert len(result['latenciesMs'])==5
    assert result['meanMs']==pytest.approx(np.mean(result['latenciesMs']))
    assert result['p95Ms']==pytest.approx(np.percentile(result['latenciesMs'],95))
    assert result['throughputFps']==pytest.approx(1000/result['meanMs'])
    assert client.post('/api/benchmark',json={'modelId':'resnet'}).status_code==409


def test_analysis_comparison_history_and_audio(client):
    body={'samples':sample_rows(capture())}
    a=client.post('/api/signal/analyze',json=body)
    assert a.status_code==200
    assert a.json()['noisePower'] is None
    compare=client.post('/api/signal/compare',json={'first':body,'second':body})
    assert compare.status_code==200
    assert compare.json()['sampleMse']==0
    assert compare.json()['powerDifferenceDb']==0
    classified=client.post('/api/classify',json=body).json()
    assert client.get('/api/history').json()[0]['id']==classified['id']
    audio=client.post('/api/signal/audio',json=body)
    assert audio.status_code==200 and audio.content.startswith(b'RIFF')

@pytest.mark.parametrize('value',[[],[1,2],np.zeros((10,2)),np.zeros((128,3)),np.zeros((1,128,2)),np.array([[np.nan,0]]*128),np.array([[np.inf,0]]*128),np.array([['a','b']]*128),np.array([[object(),0]]*128)])
def test_invalid_arrays(value):
    with pytest.raises(ValueError): validate_iq(np.asarray(value))


def test_invalid_requests_and_files(client):
    assert client.post('/api/classify',json={'samples':[]}).status_code==422
    assert client.post('/api/classify',json={'samples':sample_rows(capture()[:16])}).status_code==422
    assert client.post('/api/classify',json={'samples':sample_rows(capture()),'windowStart':1}).status_code==422
    assert client.post('/api/signal/generate',json={'modulation':'UNKNOWN'}).status_code==422
    assert client.post('/api/signal/generate',json={'numSamples':99999999}).status_code==422
    assert client.post('/api/signal/generate',json={'dopplerHz':1}).status_code==422
    for name,payload in [('signal.npy',b'garbage'),('signal.csv',b'I,Q\nNaN,Inf'),('script.py',b'print(1)'),('empty.npy',b''),('signal.wav',b'RIFF'),('signal.npz',b'not a zip')]:
        assert client.post('/api/classify/upload',files={'file':(name,payload)}).status_code==422
    assert client.get('/api/samples/999').status_code==404


def test_upload_size_limit(client):
    response=client.post('/api/classify/upload',files={'file':('signal.npy',b'x'*(MAX_UPLOAD_BYTES+1))})
    assert response.status_code==413


def test_npz_decompression_bomb_rejected(client):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('iq.npy',b'\0'*(MAX_UPLOAD_BYTES+1))
    assert client.post('/api/classify/upload',files={'file':('bomb.npz',buf.getvalue())}).status_code==422


def test_pickle_payload_rejected(client):
    buf=io.BytesIO();np.save(buf,np.array([{'unsafe':True}],dtype=object),allow_pickle=True)
    assert client.post('/api/classify/upload',files={'file':('object.npy',buf.getvalue())}).status_code==422


def test_training_transform_preserves_distribution():
    raw=capture();mean,std,_=stats()
    np.testing.assert_allclose(preprocess(raw),(raw-mean)/std)
    # A second unit-power normalization would change this by ~100x.
    assert .1<np.std(preprocess(raw))<5


def test_zero_signal_analysis_is_finite(client):
    r=client.post('/api/signal/analyze',json={'samples':[{'i':0,'q':0}]*128})
    assert r.status_code==200
    assert all(v is None or np.isfinite(v) for v in r.json()['features'].values())
    assert r.json()['features']['Kurtosis_I'] is None
    assert r.json()['features']['PAPR_dB'] is None


def test_unavailable_model_reports_503(client,monkeypatch):
    from backend.services import classifier
    def missing(): raise RuntimeError('Model artifact is missing for test')
    monkeypatch.setattr(classifier,'model',missing)
    response=client.get('/api/health')
    assert response.status_code==503 and not response.json()['connected']
    assert client.post('/api/classify',json={'samples':sample_rows(capture())}).status_code==503


def test_cors_and_orbit(client):
    response=client.options('/api/classify',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    assert response.headers['access-control-allow-origin']=='http://localhost:5173'
    r=client.get('/api/orbit?altitudeKm=550&carrierGhz=8.4')
    assert r.status_code==200 and len(r.json()['doppler_khz'])==201


def test_unsigned_mono_wav_keeps_zero_quadrature():
    pcm=np.arange(64,192,dtype=np.uint8)
    buf=io.BytesIO();wavfile.write(buf,8000,pcm)
    iq,fs=parse_capture('mono.wav',buf.getvalue())
    assert fs==8000
    np.testing.assert_array_equal(iq[:,1],np.zeros(128))
    np.testing.assert_allclose(iq[:,0],(pcm.astype(float)-128)/128)
