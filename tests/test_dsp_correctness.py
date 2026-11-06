from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.signal_processing import CONSTELLATION_REFS,generate_iq_signal,compute_psd,saleh_twta
from utils.model_backend import NumPyCNNEngine
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('mod,size',[('BPSK',2),('QPSK',4),('QAM16',16),('8PSK',8),('16APSK',16),('32APSK',32)])
def test_constellation_count_and_energy(mod,size):
    refs=CONSTELLATION_REFS[mod]
    assert len(np.unique(refs,axis=0))==size
    assert np.mean(np.sum(refs**2,axis=1))==pytest.approx(1,rel=1e-6)

@pytest.mark.parametrize('mod',['BFSK','GMSK'])
def test_continuous_phase_frequency_modulations(mod):
    iq=generate_iq_signal(mod,num_samples=8192,sps=8,snr_db=150,rng_seed=42)
    z=iq[:,0]+1j*iq[:,1]
    assert np.std(np.abs(z))<1e-5
    increments=np.angle(z[1:]*np.conj(z[:-1]))
    assert np.max(np.abs(increments))<=np.pi/16+1e-5
    assert np.std(increments)>.05
    assert np.std(iq[:,1])>.3  # They must not degenerate to BPSK.

def test_generator_rejects_unknown_and_aliases_qam():
    with pytest.raises(ValueError):generate_iq_signal('UNKNOWN')
    np.testing.assert_array_equal(generate_iq_signal('16-QAM',rng_seed=42),generate_iq_signal('QAM16',rng_seed=42))

def test_psd_integrates_to_signal_power():
    iq=generate_iq_signal('QPSK',num_samples=4096,rng_seed=4)
    fs=48000
    _,psd=compute_psd(iq,fs)
    integral=np.sum(10**(psd/10))*(fs/len(iq))
    assert integral==pytest.approx(np.mean(np.sum(iq**2,axis=1)),rel=1e-5)

def test_saleh_zero_signal_and_compression():
    assert np.all(saleh_twta(np.zeros((128,2)))==0)
    iq=np.column_stack((np.linspace(.1,4,128),np.zeros(128)))
    output=saleh_twta(iq)
    expected=2.1587*iq[:,0]/(1+1.1517*iq[:,0]**2)
    np.testing.assert_allclose(np.linalg.norm(output,axis=1),expected,rtol=1e-6)

def test_npz_weights_equal_original_keras_archive():
    weights=NumPyCNNEngine.from_npz(ROOT/'models/cnn_weights.npz').w
    original=NumPyCNNEngine.from_keras_zip(ROOT/'satellite_modulation_cnn_all_snr.keras').w
    for k in weights:np.testing.assert_array_equal(weights[k],original[k])

def test_convolution_matches_same_padding_cross_correlation():
    engine=NumPyCNNEngine.from_npz(ROOT/'models/cnn_weights.npz')
    rng=np.random.default_rng(42)
    x=rng.normal(size=(128,2)).astype(np.float32)
    w,b=engine.w['c1_w'],engine.w['c1_b']
    expected=np.stack([sum(np.correlate(x[:,c],w[:,c,j],mode='same') for c in range(2))+b[j] for j in range(32)],axis=1)
    np.testing.assert_allclose(engine._conv1d(x,w,b),expected,atol=1e-5)


def test_eye_diagram_preserves_exact_sample_timing():
    from utils.signal_processing import compute_eye_diagram
    iq=generate_iq_signal('QPSK',sps=8,rng_seed=42)
    time, i_traces, q_traces=compute_eye_diagram(iq,sps=8)
    np.testing.assert_allclose(np.diff(time),1/32)
    np.testing.assert_array_equal(i_traces[0][::4],iq[:16,0])
    np.testing.assert_array_equal(q_traces[0][::4],iq[:16,1])
