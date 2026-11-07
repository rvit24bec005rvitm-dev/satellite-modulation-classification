"""Scientific regression tests: real artifacts, physical invariants, no manufactured model results."""
from pathlib import Path
import sys,io,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app
from backend.services import classifier
from backend.services.signals import analyze,parse_capture
from backend.services.explain import explain
from backend.services.uncertainty import calibration_metrics,fit_temperature,detection_metrics
from utils.model_backend import ModelBackend,NumPyCNNEngine
from utils.preprocessing import prepare_model_input
from utils.modulation import constellation,map_bits,frequency_modulate
from utils.signal_processing import generate_iq_signal,add_awgn,_rrc_filter,compute_psd
ROOT=Path(__file__).resolve().parents[1]


def frame():return classifier.dataset()['signals'][42]
def rows(a):return [{'i':float(i),'q':float(q)} for i,q in a]


@pytest.mark.parametrize('bad',[np.zeros((128,3)),np.zeros(256),np.zeros((127,2)),np.full((128,2),np.nan),np.ones((128,2),complex)])
def test_canonical_preprocessing_rejects_invalid(bad):
    with pytest.raises(ValueError):prepare_model_input(bad,0,.006)


@pytest.mark.parametrize('start',[-1,.5,True,1])
def test_canonical_invalid_window_offsets(start):
    with pytest.raises(ValueError):classifier.preprocess(frame(),start)


def test_preprocessing_channel_order_determinism_and_no_double_normalization():
    mean,std,_=classifier.stats();raw=frame()
    x=classifier.preprocess(raw)
    np.testing.assert_array_equal(x,(raw-np.float32(mean))/np.float32(std))
    np.testing.assert_array_equal(x,classifier.preprocess(raw.copy()))
    assert x.dtype==np.float32 and x.shape==(128,2)
    np.testing.assert_array_equal(x[:,0],(raw[:,0]-np.float32(mean))/np.float32(std))
    assert not np.allclose(x,classifier.preprocess(raw/np.sqrt(np.mean(np.sum(raw**2,axis=1)))))


def test_exact_artifact_loading_never_substitutes_sibling_weights(tmp_path):
    np.savez(tmp_path/'cnn_weights.npz',**classifier.model()._session.w)
    with pytest.raises(RuntimeError,match='missing'):ModelBackend.load(tmp_path/'absent.keras')
    (tmp_path/'other.onnx').write_bytes(b'not an ONNX model')
    with pytest.raises(RuntimeError) as caught:ModelBackend.load(tmp_path/'other.onnx')
    assert 'NumPy' not in str(caught.value)
    np.savez(tmp_path/'wrong.npz',weights=np.ones(2))
    with pytest.raises(RuntimeError,match='expected production CNN'):ModelBackend.load(tmp_path/'wrong.npz')


def test_batch_class_order_and_determinism():
    b=classifier.model();x=np.stack([classifier.preprocess(v) for v in classifier.dataset()['signals'][:4]])
    predictions=b.predict_batch(x)
    np.testing.assert_array_equal(predictions,np.stack([b.predict(v) for v in x]))
    np.testing.assert_allclose(predictions.sum(1),1,atol=1e-6)
    assert classifier.stats()[2]==['BPSK','QPSK','16-QAM']
    for bad in [np.zeros((0,128,2)),np.zeros((2,128,3)),np.zeros((128,2))]:
        with pytest.raises(ValueError):b.predict_batch(bad)


@pytest.mark.parametrize('target',[0,1,2])
def test_analytic_logit_gradients_match_numerical_direction(target):
    engine=classifier.model()._session;x=classifier.preprocess(frame())
    rng=np.random.default_rng(341);direction=rng.normal(size=x.shape).astype(np.float32);direction/=np.linalg.norm(direction)
    analytic=float(np.sum(engine.logit_gradient(x,target)*direction))
    step=.001
    numeric=float((engine.logits(x+step*direction)[target]-engine.logits(x-step*direction)[target])/(2*step))
    assert analytic==pytest.approx(numeric,abs=.002,rel=.03)


def test_integrated_gradients_real_target_completeness_and_baseline():
    result=explain(frame(),target='BPSK',steps=128)
    assert result['targetClass']=='BPSK' and result['method']=='integrated-gradients'
    assert len(result['samples'])==128
    assert result['attributionSum']==pytest.approx(result['logitDifference'],abs=max(.03,.04*abs(result['logitDifference'])))
    assert sum(p['importance'] for p in result['samples'])>0
    different=explain(frame(),target='QPSK',method='saliency')
    assert different['completenessError'] is None
    assert [p['i'] for p in result['samples']]!=[p['i'] for p in different['samples']]


@pytest.mark.parametrize('mod,size',[('BPSK',2),('QPSK',4),('8-PSK',8),('16-QAM',16),('32-QAM',32),('16-APSK',16),('32-APSK',32)])
def test_bit_mapping_equals_all_ideal_symbols(mod,size):
    width=int(np.log2(size));bits=((np.arange(size)[:,None] >> np.arange(width-1,-1,-1))&1).reshape(-1)
    symbols=map_bits(bits,mod);refs=constellation(mod)
    assert len(np.unique(refs,axis=0))==size
    np.testing.assert_allclose(np.column_stack((symbols.real,symbols.imag)),refs,atol=1e-7)
    assert np.mean(abs(symbols)**2)==pytest.approx(1,rel=1e-6)


@pytest.mark.parametrize('mod,counts,radii',[('16-APSK',[4,12],[1,3.1]),('32-APSK',[4,12,16],[1,2.8,5.2])])
def test_configurable_apsk_ring_geometry(mod,counts,radii):
    refs=constellation(mod,radii,[.1]*len(radii));r=np.linalg.norm(refs,axis=1)
    actual,counts_actual=np.unique(np.round(r,5),return_counts=True)
    np.testing.assert_array_equal(counts_actual,counts)
    np.testing.assert_allclose(actual/actual[0],np.array(radii)/radii[0],rtol=5e-5)
    with pytest.raises(ValueError):constellation(mod,list(reversed(radii)))


def test_bfsk_bit_frequency_states_and_continuous_phase():
    bits=np.array([0,0,1,1,0,1]);sps=16;fs=8000;deviation=125
    z=frequency_modulate(bits,sps,fs,deviation)
    frequency=np.angle(z[1:]*z[:-1].conj())*fs/(2*np.pi)
    expected=np.repeat(2*bits-1,sps)[1:]*deviation
    np.testing.assert_allclose(frequency,expected,atol=1e-9)
    noncontinuous=frequency_modulate(bits,sps,fs,deviation,continuous_phase=False)
    assert not np.allclose(z,noncontinuous)


def test_gmsk_bt_continuity_frequency_and_envelope():
    fs,sps=8000,8
    constant=frequency_modulate(np.ones(100),sps,fs,bt=.3)
    np.testing.assert_allclose(abs(constant),1,atol=1e-12)
    np.testing.assert_allclose(np.angle(constant[1:]*constant[:-1].conj())*fs/(2*np.pi),fs/(4*sps),atol=1e-9)
    bits=np.random.default_rng(12).integers(0,2,400)
    a=frequency_modulate(bits,sps,fs,bt=.15);b=frequency_modulate(bits,sps,fs,bt=.8)
    da=np.angle(a[1:]*a[:-1].conj());db=np.angle(b[1:]*b[:-1].conj())
    assert np.std(np.diff(da))<np.std(np.diff(db))
    assert max(abs(da))<=np.pi/(2*sps)+1e-9


def test_awgn_sample_snr_and_quadrature_power():
    rng=np.random.default_rng(42);z=np.ones(65536,dtype=complex)
    noisy,noise=add_awgn(z,7,rng)
    assert 10*np.log10(np.mean(abs(z)**2)/np.mean(abs(noise)**2))==pytest.approx(7,abs=.08)
    assert np.var(noise.real)/np.var(noise.imag)==pytest.approx(1,abs=.03)
    np.testing.assert_allclose(noisy-z,noise,atol=1e-15)


def test_frequency_offset_and_fft_peak_location():
    z=np.exp(2j*np.pi*.125*np.arange(1024));iq=np.column_stack((z.real,z.imag))
    f,p=compute_psd(iq,8000);assert f[p.argmax()]==1000
    base=generate_iq_signal('QPSK',num_samples=1024,snr_db=150,rng_seed=5,rrc_enabled=False)
    shifted=generate_iq_signal('QPSK',num_samples=1024,snr_db=150,rng_seed=5,rrc_enabled=False,cfo_hz=25,sample_rate=1000)
    a=base[:,0]+1j*base[:,1];b=shifted[:,0]+1j*shifted[:,1]
    np.testing.assert_allclose(b/a,np.exp(2j*np.pi*.025*np.arange(1024)),atol=1e-6)


def test_rician_power_and_disable_semantics():
    _,meta=generate_iq_signal('QPSK',num_samples=65536,snr_db=150,rng_seed=12,rrc_enabled=False,rician_enabled=True,rician_k=3,normalize_output=False,return_metadata=True)
    assert meta['powerStages']['afterFading']==pytest.approx(1,abs=.03)
    a=generate_iq_signal('QPSK',rng_seed=12,rician_k=3,rician_enabled=False)
    b=generate_iq_signal('QPSK',rng_seed=12,rician_k=0,rician_enabled=False)
    np.testing.assert_array_equal(a,b)


def test_twta_power_is_preserved_when_requested_and_documented_when_normalized():
    a,m=generate_iq_signal('BPSK',snr_db=150,rrc_enabled=False,twta_alpha_a=2.1587,twta_drive=.5,normalize_output=False,rng_seed=2,return_metadata=True)
    expected=(2.1587*.5/(1+1.1517*.5**2))**2
    assert np.mean(np.sum(a**2,axis=1))==pytest.approx(expected,rel=1e-5)
    assert m['powerStages']['afterTwta']==pytest.approx(expected,rel=1e-5)
    _,normalized=generate_iq_signal('BPSK',twta_alpha_a=2.1587,return_metadata=True)
    assert normalized['outputPowerPolicy']=='unit-rms'


def test_phase_noise_mode_changes_actual_phase_statistics():
    kw=dict(modulation='QPSK',num_samples=8192,snr_db=150,rng_seed=10,rrc_enabled=False)
    base=generate_iq_signal(**kw);z=base[:,0]+1j*base[:,1]
    values=[]
    for mode in ['iid','wiener']:
        x=generate_iq_signal(**kw,phase_noise_std=.05,phase_noise_mode=mode)
        phi=np.unwrap(np.angle((x[:,0]+1j*x[:,1])/z));values.append(phi)
    assert np.std(values[0])==pytest.approx(.05,abs=.003)
    assert np.std(np.diff(values[1]))==pytest.approx(.05,abs=.003)
    assert np.std(values[1])>.3


@pytest.mark.parametrize('sps',[1,4,8,32])
@pytest.mark.parametrize('alpha',[.1,1/3,.5,1.])
def test_rrc_energy_symmetry_and_finite_taps(sps,alpha):
    h=_rrc_filter(8*sps+1,sps,alpha)
    assert np.isfinite(h).all()
    np.testing.assert_allclose(h,h[::-1],atol=1e-12)
    assert np.sum(h*h)==pytest.approx(1,abs=1e-12)


def test_power_rms_variance_crest_and_unknown_snr():
    iq=np.column_stack((np.arange(128)/128,np.zeros(128)));a=analyze(iq,8000)
    assert a['signalPower']==pytest.approx(np.mean(iq[:,0]**2))
    assert a['rmsAmplitude']**2==pytest.approx(a['signalPower'])
    assert a['features']['I_Variance']==pytest.approx(np.var(iq[:,0]))
    assert a['features']['Crest_Factor']**2==pytest.approx(10**(a['features']['PAPR_dB']/10))
    assert a['estimatedSnrDb'] is None
    assert 'Signal_Power_dBW' not in a['features']
    assert len(a['instantaneousFrequency']) and len(a['phaseTrace'])


def test_calibration_metrics_and_validation_temperature():
    from scipy.special import softmax
    logits=np.array([[5,0,0],[5,0,0],[0,5,0],[0,5,0]],float);y=np.array([0,1,1,0])
    t=fit_temperature(logits,y);before=calibration_metrics(softmax(logits,axis=1),y);after=calibration_metrics(softmax(logits/t,axis=1),y)
    assert t>1 and after['nll']<before['nll']
    assert sum(b['count'] for b in before['reliability'])==4
    assert calibration_metrics(np.eye(3),np.arange(3))['brier']==0
    m=detection_metrics([.1,.2,.3],[.7,.8,.9]);assert m['auroc']==1 and m['fpr']==0
    frozen=detection_metrics([.05,.15],[.8,.9],threshold=m['threshold']);assert frozen['threshold']==m['threshold']


def test_api_explanation_validity_and_comparison_axes():
    with TestClient(app) as c:
        body={'samples':rows(frame())}
        result=c.post('/api/classify',json=body).json()
        assert result['trust']['calibration']['status']=='not-calibrated'
        assert result['trust']['ood']['status']=='not-validated'
        assert result['trust']['inputValidity']['status']=='valid'
        attribution=c.post('/api/explain',json={**body,'method':'saliency','targetClass':'16-QAM'})
        assert attribution.status_code==200,attribution.text
        assert attribution.json()['targetClass']=='16-QAM'
        assert c.post('/api/explain',json={**body,'steps':99999}).status_code==422
        mixed=c.post('/api/signal/compare',json={'first':body,'second':{**body,'sampleRate':8000}}).json()
        assert mixed['sampleMse'] is None
        assert mixed['plots']['frequencyUnit']=='cycles/sample'
        for a,b in zip(mixed['plots']['first']['spectrum'],mixed['plots']['second']['spectrum']):
            assert a['frequency']==pytest.approx(b['frequency'])
            assert a['psd']==pytest.approx(b['psd'],abs=1e-7)
        assert all(v['status']=='available' for v in mixed['predictions'].values())
        registry=c.get('/api/models').json();assert registry[0]['artifactSha256']
        timing=c.post('/api/benchmark',json={'iterations':5}).json()
        assert timing['hardware']['batchSize']==1 and len(timing['preprocessing']['latenciesMs'])==5


def test_cpm_has_no_fake_eye_and_configured_controls_apply():
    with TestClient(app) as c:
        a=c.post('/api/signal/generate',json={'modulation':'GMSK','gaussianBt':.15}).json()
        b=c.post('/api/signal/generate',json={'modulation':'GMSK','gaussianBt':.8}).json()
        assert a['analysis']['eye'] is None and a['reference']==[]
        assert a['samples']!=b['samples']
        assert a['generation']['modulationIndex']==.5
        assert c.post('/api/signal/generate',json={'modulation':'BFSK','bfskDeviationHz':.51}).status_code==422
        qam=c.post('/api/signal/generate',json={'modulation':'32-QAM'}).json()
        assert not qam['classifierSupported'] and len(qam['reference'])==32


def test_malformed_archive_empty_csv_and_invalid_sample_clock_are_rejected():
    import zipfile
    from scipy.io import wavfile
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        with pytest.warns(UserWarning,match='Duplicate'):
            archive.writestr('iq.npy',b'bad');archive.writestr('iq.npy',b'bad')
    with pytest.raises(ValueError,match='Duplicate'):parse_capture('duplicate.npz',buffer.getvalue())
    with pytest.raises(ValueError,match='empty'):parse_capture('empty.csv',b' \n ')
    buffer=io.BytesIO();wavfile.write(buffer,8000,frame());data=bytearray(buffer.getvalue());data[24:28]=b'\0'*4
    with pytest.raises(ValueError,match='Sample rate'):parse_capture('bad-rate.wav',bytes(data))


def test_request_limits_keep_cors_and_json_iq_is_strict():
    with TestClient(app) as c:
        response=c.post('/api/classify',content=b'{}',headers={'Origin':'http://localhost:5173','Content-Length':str(17*1024*1024)})
        assert response.status_code==413
        assert response.headers['access-control-allow-origin']=='http://localhost:5173'
        assert c.post('/api/classify',json={'samples':[{'i':True,'q':False}]*128}).status_code==422


def test_keras_loader_rejects_same_weights_with_different_architecture(tmp_path):
    import zipfile
    with zipfile.ZipFile(ROOT/'satellite_modulation_cnn_all_snr.keras') as source:
        config=json.loads(source.read('config.json'))
        for layer in config['config']['layers']:
            if layer['class_name']=='MaxPooling1D':layer['config']['pool_size']=[4]
        with zipfile.ZipFile(tmp_path/'changed.keras','w') as destination:
            for key in source.namelist():destination.writestr(key,json.dumps(config) if key=='config.json' else source.read(key))
    with pytest.raises(RuntimeError,match='pooling'):ModelBackend.load(tmp_path/'changed.keras')


def test_experimental_registry_and_analysis_only_upload_are_real():
    from backend.services.model_registry import REGISTRY
    with TestClient(app) as c:
        reports=c.get('/api/experiments').json()
        assert len(reports)>=2 and all(r['protocolId']==reports[0]['protocolId'] for r in reports)
        metadata=next(m for m in c.get('/api/models').json() if m['id']=='rf-experimental')
        import sklearn
        if sklearn.__version__ != reports[0]['environment']['scikitLearn']:
            assert metadata['status']=='unavailable' and 'scikit-learn' in metadata['error']
            assert c.post('/api/benchmark',json={'modelId':'rf-experimental','iterations':5}).status_code==503
        else:
            assert metadata['status']=='available'
            assert metadata['evaluationMetadata']['status']=='held-out-synthetic-experiment'
            timing=c.post('/api/benchmark',json={'modelId':'rf-experimental','iterations':5}).json()
            assert len(timing['latenciesMs'])==5 and timing['modelId']=='rf-experimental'
        payload=json.dumps({'samples':rows(frame()[:64])}).encode()
        uploaded=c.post('/api/signal/upload',files={'file':('short.json',payload)})
        assert uploaded.status_code==200 and len(uploaded.json()['samples'])==64
        compared=c.post('/api/signal/compare',json={'first':{'samples':rows(frame())},'second':{'samples':rows(frame()[:64])}}).json()
        assert compared['predictions']['second']['status']=='unavailable'
        assert compared['overlapSamples']==64


def test_orbital_pass_has_one_consistent_geometry_and_valid_coordinates():
    from utils.signal_processing import simulate_orbit_pass
    result=simulate_orbit_pass(550,8.4,78)
    mid=len(result['time_sec'])//2
    assert result['elevation_deg'][mid]==pytest.approx(78,abs=1e-9)
    assert result['doppler_khz'][mid]==pytest.approx(0,abs=1e-8)
    assert np.all(abs(result['track_lats'])<=90) and np.all(abs(result['track_lons'])<=180)
    numerical=np.gradient(result['range_km'],result['time_sec'])
    np.testing.assert_allclose(numerical[2:-2],result['range_rate_kms'][2:-2],atol=.002)
    np.testing.assert_allclose(result['doppler_khz'],-result['range_rate_kms']/299792.458*8.4e6,rtol=1e-12)
    assert result['range_km'][mid]==min(result['range_km'])


def test_undefined_zero_signal_phase_and_scale_invariant_moments():
    from utils.signal_processing import extract_features
    a=analyze(np.zeros((128,2)))
    assert a['features']['Phase_Std_rad'] is None
    assert all(p['phaseRad'] is None for p in a['phaseTrace'])
    assert all(p['frequency'] is None for p in a['instantaneousFrequency'])
    raw=np.random.default_rng(12).normal(size=(128,2))
    normal=extract_features(raw);small=extract_features(raw*1e-20)
    assert normal['Skewness_I']==pytest.approx(small['Skewness_I'],abs=1e-12)
    assert normal['Kurtosis_Q']==pytest.approx(small['Kurtosis_Q'],abs=1e-12)


def test_visual_spectral_downsampling_conserves_contiguous_linear_power():
    iq=generate_iq_signal('QPSK',num_samples=1301,rng_seed=3)
    a=analyze(iq,8000)
    f,p=compute_psd(iq.astype(float),8000);groups=np.array_split(np.arange(len(f)),1024)
    expected=[10*np.log10(np.mean(10**(p[g]/10))+1e-30) for g in groups]
    np.testing.assert_allclose([v['psd'] for v in a['spectrum']],expected,atol=1e-10)
    zero=analyze(np.zeros((128,2)))
    assert zero['bandwidthHz'] is None and zero['powerDb'] is None
    assert sum(p['count'] for p in zero['phaseDistribution'])==0


def test_reversed_csv_headers_and_invalid_entropy_vectors_are_rejected():
    from utils.signal_processing import shannon_entropy
    with pytest.raises(ValueError,match='reversed'):parse_capture('channels.csv',b'Q,I\n'+b'1,2\n'*128)
    for value in [[-.1,.1,1],[.2,.2],[np.nan,0,1]]:
        with pytest.raises(ValueError):shannon_entropy(np.array(value))
    assert shannon_entropy(np.array([1.,0,0]))==0
