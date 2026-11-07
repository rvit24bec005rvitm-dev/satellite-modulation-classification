"""Exact-artifact inference with the original CNN and a validated NumPy implementation.

Optional TensorFlow/ONNX libraries are imported only when explicitly needed.
No missing or unsupported model is replaced with a different model's weights.
"""
from __future__ import annotations
from pathlib import Path
import io
import json
import zipfile
import numpy as np


class NumPyCNNEngine:
    """Saved SAME Conv1D → ReLU → BN CNN, with real logit gradients."""
    SHAPES = {'c1_w': (5, 2, 32), 'c1_b': (32,), 'c2_w': (5, 32, 64), 'c2_b': (64,),
              'c3_w': (3, 64, 96), 'c3_b': (96,), 'd1_w': (96, 64), 'd1_b': (64,),
              'd2_w': (64, 3), 'd2_b': (3,),
              **{f'bn{i}_{key}': (width,) for i, width in [(1, 32), (2, 64), (3, 96)]
                 for key in ['g', 'b', 'm', 'v']}}

    def __init__(self, weights: dict[str, np.ndarray]):
        if set(weights) != set(self.SHAPES):
            raise ValueError('The supplied artifact does not contain the expected production CNN weights.')
        for key, shape in self.SHAPES.items():
            value = weights[key]
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError(f'Invalid CNN tensor: {key}; expected finite floating weights {shape}.')
            if key.endswith('_v') and np.any(value < 0):
                raise ValueError('Batch normalization variance cannot be negative.')
        self.w = {key: value.astype(np.float32, copy=True) for key, value in weights.items()}

    @classmethod
    def from_npz(cls, path: str | Path) -> 'NumPyCNNEngine':
        with np.load(path, allow_pickle=False) as data:
            return cls({key: data[key] for key in data.files})

    @classmethod
    def from_keras_zip(cls, path: str | Path) -> 'NumPyCNNEngine':
        import h5py
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read('config.json'))
            layers = config['config']['layers']
            kinds = [layer['class_name'] for layer in layers if layer['class_name'] not in ['InputLayer', 'Dropout']]
            expected = ['Conv1D', 'BatchNormalization', 'MaxPooling1D', 'Conv1D', 'BatchNormalization',
                        'MaxPooling1D', 'Conv1D', 'BatchNormalization', 'GlobalAveragePooling1D', 'Dense', 'Dense']
            if kinds != expected:
                raise ValueError('Keras architecture is not the supported production CNN.')
            convs = [layer['config'] for layer in layers if layer['class_name'] == 'Conv1D']
            for layer, (filters, kernel) in zip(convs, [(32, 5), (64, 5), (96, 3)]):
                if (layer['filters'], layer['kernel_size'], layer['padding'], layer['activation']) != (filters, [kernel], 'same', 'relu'):
                    raise ValueError('Unsupported CNN convolution configuration.')
            dense_layers = [layer['config'] for layer in layers if layer['class_name'] == 'Dense']
            if [(c['units'],c['activation']) for c in dense_layers] != [(64,'relu'),(3,'softmax')]:
                raise ValueError('Unsupported CNN dense configuration.')
            for layer in layers:
                c = layer['config']
                if layer['class_name'] == 'InputLayer' and c.get('batch_shape', c.get('batch_input_shape')) != [None,128,2]:
                    raise ValueError('Unsupported CNN input shape.')
                if layer['class_name'] == 'Conv1D' and (c.get('strides',[1]) != [1] or c.get('dilation_rate',[1]) != [1] or c.get('groups',1) != 1 or c.get('data_format','channels_last') != 'channels_last'):
                    raise ValueError('Unsupported CNN convolution stride, dilation or channel order.')
                if layer['class_name'] == 'MaxPooling1D' and (c.get('pool_size') != [2] or c.get('strides') != [2] or c.get('padding') != 'valid'):
                    raise ValueError('Unsupported CNN pooling configuration.')
                if layer['class_name'] == 'BatchNormalization' and c.get('axis',-1) not in [-1,[-1]]:
                    raise ValueError('Unsupported batch normalization channel axis.')
                if layer['class_name'] == 'BatchNormalization' and (c.get('epsilon', .001) != .001 or c.get('renorm', False)):
                    raise ValueError('Unsupported batch normalization configuration.')
            with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as weights_file:
                weights = {}
                for i, name in [(1, 'conv1d'), (2, 'conv1d_1'), (3, 'conv1d_2')]:
                    for j, key in enumerate(['w', 'b']): weights[f'c{i}_{key}'] = weights_file[f'layers/{name}/vars/{j}'][:]
                for i, name in [(1, 'batch_normalization'), (2, 'batch_normalization_1'), (3, 'batch_normalization_2')]:
                    for j, key in enumerate(['g', 'b', 'm', 'v']): weights[f'bn{i}_{key}'] = weights_file[f'layers/{name}/vars/{j}'][:]
                for i, name in [(1, 'dense'), (2, 'dense_1')]:
                    for j, key in enumerate(['w', 'b']): weights[f'd{i}_{key}'] = weights_file[f'layers/{name}/vars/{j}'][:]
        return cls(weights)

    @staticmethod
    def _validate(iq: np.ndarray) -> np.ndarray:
        a = np.asarray(iq)
        if a.shape != (128, 2) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
            raise ValueError('Expected finite real I/Q samples with shape (128, 2).')
        with np.errstate(over='ignore'):
            value = a.astype(np.float32)
        if not np.isfinite(value).all(): raise ValueError('Input cannot be represented as finite float32.')
        return value

    def _conv1d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        kernel = w.shape[0]
        padded = np.pad(x, ((kernel // 2, kernel // 2), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=0)
        return np.einsum('tck,kco->to', windows, w, optimize=True) + b

    def _bn(self, x, g, b, m, v, eps=1e-3):
        return (x - m) / np.sqrt(v + eps) * g + b

    def _maxpool(self, x, pool=2):
        return x[:len(x) // pool * pool].reshape(-1, pool, x.shape[1]).max(axis=1)

    def _forward(self, iq: np.ndarray):
        w, x, cache = self.w, self._validate(iq), []
        for i in [1, 2, 3]:
            conv = self._conv1d(x, w[f'c{i}_w'], w[f'c{i}_b'])
            bn = self._bn(np.maximum(conv, 0), *(w[f'bn{i}_{k}'] for k in ['g', 'b', 'm', 'v']))
            cache.append((x, conv, bn))
            x = self._maxpool(bn) if i < 3 else bn
        pooled = x.mean(axis=0)
        dense = pooled @ w['d1_w'] + w['d1_b']
        logits = np.maximum(dense, 0) @ w['d2_w'] + w['d2_b']
        return logits.astype(np.float32), (cache, pooled, dense)

    def logits(self, iq: np.ndarray) -> np.ndarray:
        return self._forward(iq)[0]

    def predict(self, iq: np.ndarray) -> np.ndarray:
        logits = self.logits(iq)
        values = np.exp(logits - np.max(logits))
        return (values / values.sum()).astype(np.float32)

    def predict_batch(self, iq: np.ndarray) -> np.ndarray:
        a = np.asarray(iq)
        if a.ndim != 3 or a.shape[1:] != (128, 2) or not 1 <= len(a) <= 4096:
            raise ValueError('Expected a batch of 1–4096 frames with shape (B,128,2).')
        return np.stack([self.predict(frame) for frame in a])

    @staticmethod
    def _conv_backward(dy: np.ndarray, weights: np.ndarray) -> np.ndarray:
        kernel, channels, _ = weights.shape
        dx = np.zeros((len(dy) + kernel - 1, channels), dtype=np.float32)
        for k in range(kernel):
            dx[k:k + len(dy)] += dy @ weights[k].T
        pad = kernel // 2
        return dx[pad:pad + len(dy)]

    def logit_gradient(self, iq: np.ndarray, target: int) -> np.ndarray:
        if isinstance(target, bool) or target not in [0, 1, 2]:
            raise ValueError('Target class must be 0, 1, or 2.')
        _, (cache, _, dense) = self._forward(iq)
        w = self.w
        gradient = (w['d2_w'][:, target] * (dense > 0)) @ w['d1_w'].T
        gradient = np.broadcast_to(gradient / len(cache[2][2]), cache[2][2].shape).copy()
        for i in [3, 2, 1]:
            _, conv, bn = cache[i - 1]
            if i < 3:
                index = bn.reshape(-1, 2, bn.shape[1]).argmax(axis=1)
                unpooled = np.zeros_like(bn)
                unpooled[2 * np.arange(len(index))[:, None] + index, np.arange(bn.shape[1])] = gradient
                gradient = unpooled
            gradient = gradient * w[f'bn{i}_g'] / np.sqrt(w[f'bn{i}_v'] + 1e-3) * (conv > 0)
            gradient = self._conv_backward(gradient, w[f'c{i}_w'])
        return gradient


class ModelBackend:
    def __init__(self, session_or_model, backend_name: str):
        self._session, self.backend_name = session_or_model, backend_name

    @classmethod
    def load(cls, model_path: str | Path) -> 'ModelBackend':
        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f'Model artifact is missing: {path.name}')
        try:
            if path.suffix == '.npz':
                return cls(NumPyCNNEngine.from_npz(path), 'NumPy Native Engine')
            if path.suffix == '.keras':
                return cls(NumPyCNNEngine.from_keras_zip(path), 'NumPy Native Engine (Extracted)')
            if path.suffix == '.onnx':
                import onnxruntime as ort
                session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
                return cls(session, 'ONNX Runtime')
            if path.suffix in ['.h5', '.tflite']:
                import tensorflow as tf
                if path.suffix == '.h5':
                    return cls(tf.keras.models.load_model(path, compile=False), 'TensorFlow/Keras')
                interpreter = tf.lite.Interpreter(model_path=str(path))
                interpreter.allocate_tensors()
                return cls(interpreter, 'TFLite')
        except (ImportError, ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f'Cannot load requested model artifact {path.name}: {exc}') from exc
        raise RuntimeError(f'Unsupported model artifact format: {path.suffix}')

    def predict(self, iq: np.ndarray) -> np.ndarray:
        frame = NumPyCNNEngine._validate(iq)
        if self.backend_name.startswith('NumPy'):
            return self._session.predict(frame)
        x = frame[None]
        if self.backend_name == 'ONNX Runtime':
            return np.asarray(self._session.run(None, {self._session.get_inputs()[0].name: x})[0][0], dtype=np.float32)
        if self.backend_name == 'TFLite':
            details = self._session.get_input_details()[0]
            self._session.set_tensor(details['index'], x)
            self._session.invoke()
            return self._session.get_tensor(self._session.get_output_details()[0]['index'])[0].astype(np.float32)
        return self._session(x, training=False).numpy()[0].astype(np.float32)

    def predict_batch(self, iq: np.ndarray) -> np.ndarray:
        a = np.asarray(iq)
        if a.ndim != 3 or a.shape[1:] != (128, 2) or not 1 <= len(a) <= 4096:
            raise ValueError('Expected a batch of 1–4096 frames with shape (B,128,2).')
        return np.stack([self.predict(frame) for frame in a])

    def logits(self, iq: np.ndarray) -> np.ndarray:
        if not self.backend_name.startswith('NumPy'):
            raise RuntimeError('Exact logits and attributions are available for the supplied NumPy CNN only.')
        return self._session.logits(iq)

    @property
    def input_shape(self) -> tuple:
        if self.backend_name.startswith('NumPy'): return (None, 128, 2)
        if self.backend_name == 'ONNX Runtime': return tuple(self._session.get_inputs()[0].shape)
        if self.backend_name == 'TFLite': return tuple(self._session.get_input_details()[0]['shape'])
        return tuple(self._session.input_shape)
