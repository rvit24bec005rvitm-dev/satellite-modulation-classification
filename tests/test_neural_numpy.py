"""Exact fixed-checkpoint inference against independently exported TensorFlow data."""
import copy
import hashlib
import json
import zipfile

import numpy as np
import pytest

from backend.core.config import ROOT
from backend.services.neural_numpy import CHECKPOINTS, FixedNeuralEngine, validate_configuration
from utils.preprocessing import prepare_model_input


GOLDEN = ROOT / 'integration/completion/neural-tf-golden.npz'
GOLDEN_HASH = 'bf29443193b42c6d7b78ded35360aae2923f6f1dd6cd75a61467735d3044d7a4'


@pytest.fixture(scope='module')
def models():
    values = {}
    for run_id, spec in CHECKPOINTS.items():
        directory = ROOT / 'experiments/runs' / run_id
        report = json.loads((directory / 'report.json').read_text())
        with zipfile.ZipFile(directory / 'best.keras') as archive:
            config = json.loads(archive.read('config.json'))
        engine = FixedNeuralEngine.from_checkpoint(run_id, directory / 'best.keras', report['artifact']['sha256'])
        assert engine.count_params() == spec['parameters']
        values[run_id] = (engine, config, report)
    return values


@pytest.mark.parametrize('run_id,prefix', [('resnet-synthetic-v1', 'resnet'), ('cldnn-synthetic-v1', 'cldnn')])
def test_all_360_heldout_probability_vectors_and_classes_match_independent_tensorflow(models, run_id, prefix):
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_HASH
    metadata = json.loads(GOLDEN.with_suffix('.json').read_text())
    assert metadata['models'][run_id]['modelHash'] == CHECKPOINTS[run_id]['artifact']
    assert 'Direct tf.keras.models.load_model' in metadata['referencePath']
    engine, _, report = models[run_id]
    norm = report['normalization']
    with np.load(GOLDEN, allow_pickle=False) as golden:
        assert list(golden['classLabels']) == ['BPSK', 'QPSK', '16-QAM']
        assert len(golden['sourceIndices']) == len(set(golden['sourceIndices'])) == 360
        frames = np.stack([prepare_model_input(frame, norm['mean'], norm['std']) for frame in golden['signals']])
        actual = engine.predict_batch(frames)
        expected = golden[prefix + '_probabilities']
        assert actual.shape == (360, 3)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        assert np.isfinite(actual).all() and np.all(actual >= 0)
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
        # Check logits independently as well as the full probability vectors.
        np.testing.assert_allclose(engine.logits(frames[0]), golden[prefix + '_logits'][0], atol=2e-5, rtol=2e-5)
    assert engine.backend_name.startswith('NumPy exact ')
    assert not hasattr(engine, 'logit_gradient')


@pytest.mark.parametrize('change', ['input', 'convolution', 'batchnorm', 'graph', 'layer-kind', 'lstm'])
def test_unsupported_graph_semantics_are_rejected(models, change):
    _, original, _ = models['cldnn-synthetic-v1']
    config = copy.deepcopy(original)
    layers = config['config']['layers']
    if change == 'input':
        layers[0]['config']['batch_shape'] = [None, 64, 2]
    elif change == 'convolution':
        next(layer for layer in layers if layer['class_name'] == 'Conv1D')['config']['strides'] = [2]
    elif change == 'batchnorm':
        next(layer for layer in layers if layer['class_name'] == 'BatchNormalization')['config']['epsilon'] = .01
    elif change == 'graph':
        layers[2]['inbound_nodes'][0]['args'][0]['config']['keras_history'][0] = 'input_layer'
    elif change == 'layer-kind':
        layers[1]['class_name'] = 'Lambda'
    else:
        next(layer for layer in layers if layer['class_name'] == 'LSTM')['config']['go_backwards'] = True
    with pytest.raises(ValueError, match='exact supported published graph'):
        validate_configuration('cldnn-synthetic-v1', config)


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nonfinite', 'negative-variance', 'extra-layer', 'missing-tensor'])
def test_corrupt_or_incompatible_tensors_are_rejected(models, change):
    engine, config, _ = models['cldnn-synthetic-v1']
    weights = {name: [value.copy() for value in values] for name, values in engine.weights.items()}
    if change == 'shape':
        weights['conv1d'][0] = weights['conv1d'][0][:-1]
    elif change == 'dtype':
        weights['conv1d'][0] = weights['conv1d'][0].astype(np.float64)
    elif change == 'nonfinite':
        weights['conv1d'][0][0, 0, 0] = np.nan
    elif change == 'negative-variance':
        weights['batch_normalization'][3][0] = -1
    elif change == 'extra-layer':
        weights['unpublished'] = []
    else:
        weights['lstm'].pop()
    with pytest.raises(ValueError):
        FixedNeuralEngine('cldnn-synthetic-v1', config, weights)


def test_checkpoint_identity_is_immutable_even_if_modified_metadata_agrees(tmp_path):
    run = 'cldnn-synthetic-v1'
    original = ROOT / 'experiments/runs' / run / 'best.keras'
    changed = tmp_path / 'changed.keras'
    changed.write_bytes(original.read_bytes() + b'explicit modified-artifact fixture')
    with pytest.raises(ValueError, match='checkpoint hash'):
        FixedNeuralEngine.from_checkpoint(run, changed, CHECKPOINTS[run]['artifact'])
    with pytest.raises(ValueError, match='supported fixed checkpoint'):
        FixedNeuralEngine.from_checkpoint(run, changed, hashlib.sha256(changed.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='supported fixed checkpoint'):
        FixedNeuralEngine.from_checkpoint('unknown', original, CHECKPOINTS[run]['artifact'])


@pytest.mark.parametrize('value', [
    np.zeros((127, 2)), np.zeros((128, 3)), np.zeros((128, 2), dtype=complex),
    np.full((128, 2), np.nan), np.full((128, 2), np.inf), np.full((128, 2), 1e300),
])
def test_invalid_inputs_never_receive_substitute_predictions(models, value):
    with pytest.raises(ValueError):
        models['cldnn-synthetic-v1'][0].predict(value)


def test_lstm_state_resets_and_weights_remain_owned_and_readonly(models):
    engine, config, _ = models['cldnn-synthetic-v1']
    random = np.random.default_rng(881)
    first, second = random.normal(size=(2, 128, 2)).astype(np.float32)
    expected = engine.predict(first)
    engine.predict(second)
    np.testing.assert_array_equal(engine.predict(first), expected)
    assert all(not value.flags.writeable for values in engine.weights.values() for value in values)
    supplied = {name: [value.copy() for value in values] for name, values in engine.weights.items()}
    owned = FixedNeuralEngine('cldnn-synthetic-v1', config, supplied)
    supplied['lstm'][0][:] = 0
    np.testing.assert_array_equal(owned.predict(first), expected)
    with pytest.raises(ValueError):
        engine.predict_batch(np.zeros((0, 128, 2), dtype=np.float32))
