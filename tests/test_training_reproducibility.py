"""Reproducibility, leakage prevention, validation-only fitting and real RF checkpoint replay."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.data import load_dataset,split_dataset
from training.runner import run_experiment
from scripts.evaluate import evaluate_run
from scripts.evaluate_ood import evaluate_scores
ROOT=Path(__file__).resolve().parents[1]


def external_fixture(path):
    rng=np.random.default_rng(421)
    x=rng.normal(size=(180,128,2)).astype(np.float32)*.006
    labels=np.tile(['BPSK','QPSK','16-QAM'],60)
    groups=np.repeat(np.arange(60),3)
    np.savez(path,signals=x,labels=labels,snrs=np.tile([-6,0,6],60),groups=groups)
    return path


def test_training_rejects_supplied_demo_even_if_renamed(tmp_path):
    path=tmp_path/'renamed.npz';path.write_bytes((ROOT/'classifier_samples.npz').read_bytes())
    with pytest.raises(ValueError,match='demo subset'):load_dataset(path,allow_small=True)


def test_groups_are_disjoint_deterministic_and_all_frames_assigned(tmp_path):
    data=load_dataset(external_fixture(tmp_path/'external.npz'),allow_small=True)
    a=split_dataset(data,42);b=split_dataset(data,42)
    for name in a:np.testing.assert_array_equal(a[name],b[name])
    assert len(set(np.concatenate(list(a.values()))))==180
    group_sets=[set(data['groups'][indices]) for indices in a.values()]
    assert not group_sets[0]&group_sets[1] and not group_sets[0]&group_sets[2] and not group_sets[1]&group_sets[2]


def test_duplicate_frame_leakage_rejected_with_different_group_ids(tmp_path):
    data=load_dataset(external_fixture(tmp_path/'external.npz'),allow_small=True)
    data['splits']=np.repeat(['train','validation','test'],60)
    data['signals'][60]=data['signals'][0]
    with pytest.raises(ValueError,match='Identical frame leakage'):split_dataset(data)
    data['groups'][60]=data['groups'][0]
    with pytest.raises(ValueError,match='Source group leakage'):split_dataset(data)


def test_missing_group_provenance_and_small_training_rejected(tmp_path):
    path=external_fixture(tmp_path/'external.npz')
    with pytest.raises(ValueError,match='1000'):load_dataset(path)
    np.savez(tmp_path/'no-groups.npz',signals=np.ones((180,128,2)),labels=np.zeros(180),snrs=np.zeros(180))
    with pytest.raises(ValueError,match='groups'):load_dataset(tmp_path/'no-groups.npz',allow_small=True)


def test_real_rf_training_replays_and_never_overwrites_or_promotes(tmp_path):
    path=external_fixture(tmp_path/'external.npz')
    config={'model':'random-forest','seed':42,'trees':6,'temperatureScaling':True}
    report=run_experiment(path,tmp_path/'run',config,allow_small=True)
    replay=evaluate_run(tmp_path/'run',path)
    assert replay['maxReplayProbabilityError']<1e-10
    data=load_dataset(path,allow_small=True)
    train=data['signals'][report['splits']['train']['indices']]
    assert report['normalization']['mean']==pytest.approx(float(train.astype(float).mean()))
    assert report['calibration']['fitSplit']=='validation'
    assert report['normalization']['fitSplit']=='train'
    assert report['productionPromoted'] is False
    with pytest.raises(ValueError,match='never overwritten'):run_experiment(path,tmp_path/'run',config,allow_small=True)
    second=run_experiment(path,tmp_path/'run2',config,allow_small=True)
    assert second['protocolId']==report['protocolId']
    with np.load(tmp_path/'run/test-predictions.npz') as a,np.load(tmp_path/'run2/test-predictions.npz') as b:
        np.testing.assert_array_equal(a['probabilities'],b['probabilities'])


def test_ood_threshold_fitted_on_validation_and_group_leakage_rejected(tmp_path):
    data={'idValidation':np.array([.1,.2,.3]),'oodValidation':np.array([.7,.8,.9]),'idTest':np.array([.15,.6]),'oodTest':np.array([.4,.9])}
    for i,key in enumerate(list(data)):data[key+'Groups']=np.array([f'{i}-{j}' for j in range(len(data[key]))])
    path=tmp_path/'scores.npz';np.savez(path,**data)
    result=evaluate_scores(path)
    assert result['thresholdFitSplit']=='validation'
    assert result['validation']['threshold']==result['test']['threshold']
    assert result['test']['tpr']==.5
    data['idTestGroups'][0]=data['idValidationGroups'][0];np.savez(path,**data)
    with pytest.raises(ValueError,match='leakage'):evaluate_scores(path)


def test_repackaged_demo_arrays_are_still_forbidden(tmp_path):
    with np.load(ROOT/'classifier_samples.npz') as original:
        np.savez(tmp_path/'repacked.npz',signals=original['signals'],labels=original['labels'],snrs=original['snrs'],groups=np.arange(300))
    with pytest.raises(ValueError,match='demo frames'):load_dataset(tmp_path/'repacked.npz',allow_small=True)


@pytest.mark.parametrize('change',[{'epochs':0},{'cfoMax':1},{'phaseRotation':4},{'awgnSnr':[10,-10]},{'seed':True},{'learningRate':float('nan')}])
def test_training_rejects_invalid_or_no_effect_augmentation_config(change):
    from training.runner import validate_config
    with pytest.raises(ValueError):validate_config(change)


def test_byte_string_labels_preserve_canonical_class_order(tmp_path):
    path=external_fixture(tmp_path/'external.npz')
    with np.load(path) as data:values={key:data[key] for key in data.files}
    values['labels']=values['labels'].astype('S8');np.savez(path,**values)
    loaded=load_dataset(path,allow_small=True)
    np.testing.assert_array_equal(loaded['labels'][:3],[0,1,2])
