"""Numerical/HTTP contracts for actual, bounded complex spectral views."""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy import signal
from backend.api.visualization import router
from backend.services.time_frequency import analyze


def tone(count=1024,fs=8000,frequency=1000,amplitude=.25):
    z=amplitude*np.exp(2j*np.pi*frequency*np.arange(count)/fs)
    return np.column_stack((z.real,z.imag)).astype(np.float32)


@pytest.mark.parametrize('window',['hann','boxcar'])
@pytest.mark.parametrize('frequency',[-1000,0,1000])
def test_complex_psd_matches_scipy_without_detrending(window,frequency):
    iq=tone(frequency=frequency);report=analyze(iq,8000,128,.5,window)
    z=iq[:,0].astype(float)+1j*iq[:,1]
    f,t,power=signal.spectrogram(z,fs=8000,window=window,nperseg=128,noverlap=64,nfft=128,
                               detrend=False,return_onesided=False,scaling='density',mode='psd')
    order=np.argsort(f)
    expected=10*np.log10(np.maximum(power[order],1e-30))
    np.testing.assert_allclose(report['frequencies'],f[order])
    np.testing.assert_allclose(report['times'],t)
    np.testing.assert_allclose(report['db'],expected,atol=1e-9)
    assert report['peaks'][0]['frequency']==frequency
    assert report['frequencyUnit']=='Hz' and report['timeUnit']=='s'


@pytest.mark.parametrize('window',['hann','boxcar'])
def test_frequency_reduction_preserves_integrated_linear_power(window):
    report=analyze(tone(count=4096),8000,777,.37,window)
    assert len(report['frequencies'])==256
    density=10**(np.asarray(report['db'])/10)
    integrated=(density*np.asarray(report['parameters']['frequencyBinWidths'])[:,None]).sum(axis=0)
    np.testing.assert_allclose(integrated,.25**2,rtol=1e-6)
    assert report['parameters']['overlapSamples']==int(.37*777)
    assert not report['parameters']['detrend'] and not report['parameters']['padding']


def test_very_high_overlap_and_long_capture_remain_bounded():
    report=analyze(tone(count=65536),8000,1024,.999999,'hann')
    assert report['parameters']['hopSamples']==1
    assert report['parameters']['availableTimeWindows']==65536-1024+1
    assert report['parameters']['sampledTimeWindows']==128
    assert np.shape(report['db'])==(256,128)
    assert report['parameters']['sampledWindowStarts'][0]==0
    assert report['parameters']['sampledWindowStarts'][-1]==65536-1024
    assert 'unshown times are not analyzed' in report['note']


def test_unknown_sample_rate_and_explicit_one_hz_are_distinct():
    iq=tone(count=128,fs=1,frequency=.125)
    normalized=analyze(iq,1,64,.5,'boxcar')
    physical=analyze(iq,1,64,.5,'boxcar',True)
    assert normalized['frequencyUnit']=='cycles/sample' and normalized['timeUnit']=='samples'
    assert physical['frequencyUnit']=='Hz' and physical['timeUnit']=='s'
    assert normalized['peaks'][0]['frequency']==physical['peaks'][0]['frequency']==.125
    with pytest.raises(ValueError,match='cannot be silently discarded'):
        analyze(iq,8000,64,.5,'hann',False)


def test_zero_signal_has_display_floor_and_no_fabricated_peaks():
    report=analyze(np.zeros((128,2)),8000)
    assert np.all(np.asarray(report['db'])==-300)
    assert report['peaks']==[]
    assert 'not a measured noise floor' in report['note']


@pytest.mark.parametrize('options',[
    {'fft_size':8},{'fft_size':1025},{'fft_size':256},{'fft_size':64.5},
    {'overlap':1},{'overlap':-.1},{'overlap':np.nan},{'window':'unknown'},
    {'sample_rate':0},{'sample_rate':np.inf},{'sample_rate_known':'yes'},
])
def test_invalid_parameters_are_rejected(options):
    with pytest.raises(ValueError):analyze(tone(count=128),**options)


def test_http_schema_errors_and_actual_report():
    app=FastAPI();app.include_router(router)
    with TestClient(app) as client:
        samples=[{'i':float(i),'q':float(q)} for i,q in tone(count=128)]
        base={'samples':samples,'sampleRate':8000,'fftSize':64,'overlap':.5,'window':'hann'}
        response=client.post('/api/research/time-frequency',json=base)
        assert response.status_code==200,response.text
        assert response.json()['peaks'][0]['frequency']==1000
        for override in [{'fftSize':256},{'fftSize':64.5},{'overlap':1},{'overlap':True},
                         {'window':'invalid'},{'sampleRateKnown':'yes'},{'unknown':1}]:
            invalid=client.post('/api/research/time-frequency',json={**base,**override})
            assert invalid.status_code==422,invalid.text
