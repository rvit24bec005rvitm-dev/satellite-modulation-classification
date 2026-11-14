"""Forward-only NumPy inference for two immutable published neural checkpoints.

This is not a general Keras deserializer. Archive and complete configuration
fingerprints allow only the two independently verified graphs below; no custom
objects, callbacks, training code, replacement weights or gradients are loaded.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from scipy.special import expit, softmax

from utils.model_backend import NumPyCNNEngine


CHECKPOINTS = {
    'resnet-synthetic-v1': {
        'kind': 'ResNet',
        'artifact': '3c8dc0d6f05a69f4c366e48fe48ba7e7c9868d87b0d3dbe883cb4d7490011277',
        'configuration': '99a955d97ad6ff0cf355cc2e090fb8335152ece9f1fb9a2db3dc0a2f11e28926',
        'parameters': 189635,
    },
    'cldnn-synthetic-v1': {
        'kind': 'CLDNN',
        'artifact': '62aa060d52b37797d81aa1b9c06499d92f4e7fdfadf457838a79fa94f9e395c6',
        'configuration': '61371b4d679fbddce8486df1b3cb289ae6d79a4e265233f1145912bf3b3f76d6',
        'parameters': 23747,
    },
}


def _tensor_inputs(value):
    found = []
    if isinstance(value, dict):
        if value.get('class_name') == '__keras_tensor__':
            descriptor = value['config']
            history = descriptor['keras_history']
            if history[1:] != [0, 0]:
                raise ValueError('Unsupported reused or multi-output neural layer.')
            found.append((history[0], descriptor['shape']))
        else:
            for item in value.values():
                found.extend(_tensor_inputs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_tensor_inputs(item))
    return found


def validate_configuration(run_id, config):
    if run_id not in CHECKPOINTS:
        raise ValueError('Unpublished neural checkpoint.')
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != CHECKPOINTS[run_id]['configuration']:
        raise ValueError('Neural configuration is not the exact supported published graph.')


def expected_weight_shapes(layer):
    """Derive checked tensor shapes from an already fingerprinted graph."""
    kind, config = layer['class_name'], layer['config']
    inputs = _tensor_inputs(layer.get('inbound_nodes', []))
    width = inputs[0][1][-1] if inputs else None
    if kind == 'Conv1D':
        shapes = [(config['kernel_size'][0], width, config['filters'])]
        return shapes + ([(config['filters'],)] if config['use_bias'] else [])
    if kind == 'BatchNormalization':
        return [(width,)] * 4
    if kind == 'Dense':
        return [(width, config['units']), (config['units'],)]
    if kind == 'TimeDistributed':
        units = config['layer']['config']['units']
        return [(width, units), (units,)]
    if kind == 'LSTM':
        units = config['units']
        return [(width, units * 4), (units, units * 4), (units * 4,)]
    if kind in ('InputLayer', 'Activation', 'Add', 'MaxPooling1D',
                'GlobalAveragePooling1D', 'Dropout'):
        return []
    raise ValueError('Unsupported neural layer; no approximate fallback is available.')


class FixedNeuralEngine:
    """The exact saved inference graph; inputs are preprocessed 128×2 frames."""
    def __init__(self, run_id, config, weights):
        validate_configuration(run_id, config)
        # Own the graph and each tensor so callers cannot alter a loaded model.
        config = json.loads(json.dumps(config))
        layers = config['config']['layers']
        expected_names = {layer['config']['name'] for layer in layers}
        if set(weights) != expected_names:
            raise ValueError('Neural tensor layer names do not match the saved graph.')
        checked = {}
        for layer in layers:
            name = layer['config']['name']
            shapes = expected_weight_shapes(layer)
            values = weights[name]
            if len(values) != len(shapes):
                raise ValueError('Neural layer has missing or unexpected tensors.')
            tensors = []
            for index, (value, shape) in enumerate(zip(values, shapes)):
                value = np.asarray(value)
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('Neural tensor must have its exact finite float32 shape.')
                if layer['class_name'] == 'BatchNormalization' and index == 3 and np.any(value < 0):
                    raise ValueError('Neural batch-normalization variance cannot be negative.')
                owned = value.copy()
                owned.setflags(write=False)
                tensors.append(owned)
            checked[name] = tuple(tensors)
        self.run_id = run_id
        self.artifact_sha256 = CHECKPOINTS[run_id]['artifact']
        self.layers = tuple(layers)
        self.weights = checked
        self.output = config['config']['output_layers'][0]
        self.backend_name = f"NumPy exact {CHECKPOINTS[run_id]['kind']} — published synthetic checkpoint"
        if self.count_params() != CHECKPOINTS[run_id]['parameters']:
            raise ValueError('Neural parameter count does not match the fixed checkpoint.')

    @classmethod
    def from_checkpoint(cls, run_id, path, recorded_hash):
        if run_id not in CHECKPOINTS or recorded_hash != CHECKPOINTS[run_id]['artifact']:
            raise ValueError('Neural metadata does not identify the supported fixed checkpoint.')
        path = Path(path)
        if path.stat().st_size > 4 * 1024 ** 2:
            raise ValueError('Published neural checkpoint exceeds its bounded archive size.')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != recorded_hash:
            raise ValueError('Neural checkpoint hash does not match the recorded experiment.')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            config = json.loads(archive.read('config.json'))
            validate_configuration(run_id, config)
            weights = {}
            with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as file:
                for group in file['layers'].values():
                    # HDF5 save-group numbering differs from graph-layer names
                    # for ResNet branches. The saved vars name identifies it.
                    name = group['vars'].attrs['name']
                    if isinstance(name, bytes):
                        name = name.decode('utf8')
                    if name in weights:
                        raise ValueError('Duplicate neural layer in the weight archive.')
                    variables = group['vars']
                    if 'cell' in group:
                        variables = group['cell']['vars']
                    elif 'layer' in group:
                        variables = group['layer']['vars']
                    if set(variables) != {str(i) for i in range(len(variables))}:
                        raise ValueError('Unexpected neural variable names.')
                    weights[name] = [variables[str(i)][:] for i in range(len(variables))]
        return cls(run_id, config, weights)

    def count_params(self):
        return sum(value.size for values in self.weights.values() for value in values)

    @staticmethod
    def _conv(value, kernel, bias=None):
        width = kernel.shape[0]
        padded = np.pad(value, ((width // 2, width // 2), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
        result = np.einsum('tck,kco->to', windows, kernel, optimize=True)
        return result if bias is None else result + bias

    @staticmethod
    def _lstm(value, weights):
        kernel, recurrent, bias = weights
        units = recurrent.shape[0]
        hidden = np.zeros(units, dtype=np.float32)
        cell = np.zeros(units, dtype=np.float32)
        # Keras gate order is input, forget, candidate, output. State is reset
        # for every independent frame, exactly as the saved stateless layer.
        for sample in value:
            gates = sample @ kernel + hidden @ recurrent + bias
            i, f, candidate, o = np.split(gates, 4)
            cell = expit(f) * cell + expit(i) * np.tanh(candidate)
            hidden = expit(o) * np.tanh(cell)
        return hidden

    def logits(self, iq):
        frame = NumPyCNNEngine._validate(iq)
        results = {}
        for layer in self.layers:
            kind, config = layer['class_name'], layer['config']
            name = config['name']
            inputs = [results[parent] for parent, _ in _tensor_inputs(layer.get('inbound_nodes', []))]
            value = inputs[0] if inputs else frame
            weights = self.weights[name]
            if kind == 'InputLayer':
                output = frame
            elif kind == 'Conv1D':
                output = self._conv(value, *weights)
                if config['activation'] == 'relu':
                    output = np.maximum(output, 0)
            elif kind == 'BatchNormalization':
                gamma, beta, mean, variance = weights
                output = (value - mean) / np.sqrt(variance + np.float32(config['epsilon'])) * gamma + beta
            elif kind == 'Activation':
                output = np.maximum(value, 0)
            elif kind == 'Add':
                output = inputs[0] + inputs[1]
            elif kind == 'MaxPooling1D':
                output = value.reshape(-1, 2, value.shape[1]).max(axis=1)
            elif kind == 'GlobalAveragePooling1D':
                output = value.mean(axis=0)
            elif kind in ('Dense', 'TimeDistributed'):
                output = value @ weights[0] + weights[1]
                activation = config['layer']['config']['activation'] if kind == 'TimeDistributed' else config['activation']
                if activation == 'relu':
                    output = np.maximum(output, 0)
            elif kind == 'LSTM':
                output = self._lstm(value, weights)
            elif kind == 'Dropout':
                output = value  # Saved inference mode; no random mask.
            else:
                raise ValueError('Unsupported neural operation; inference is unavailable.')
            results[name] = output
        logits = results[self.output]
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise RuntimeError('Neural inference returned invalid class logits.')
        return logits.astype(np.float32)

    def predict(self, iq):
        return softmax(self.logits(iq)).astype(np.float32)

    def predict_batch(self, frames):
        frames = np.asarray(frames)
        if frames.ndim != 3 or frames.shape[1:] != (128, 2) or not 1 <= len(frames) <= 4096:
            raise ValueError('Expected 1–4096 complete neural input frames.')
        return np.stack([self.predict(frame) for frame in frames])
