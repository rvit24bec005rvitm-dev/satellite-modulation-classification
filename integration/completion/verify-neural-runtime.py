"""Execute all genuine registered models and replay frozen neural test evidence.

Run using .venv-neural-runtime/bin/python. Only fresh seeded synthetic API
fixtures enter the isolated temporary workspace. No fitting or model promotion
occurs. Existing model, dataset and experiment evidence files are hashed before
and after verification, and the owned native server always stops in finally.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preservation():
    paths = [ROOT / 'models/cnn_weights.npz', ROOT / 'satellite_modulation_cnn_all_snr.keras',
             ROOT / 'normalization_stats_all_snr.npz', ROOT / 'classifier_samples.npz',
             ROOT / 'evaluation_results.json', ROOT / 'snr_accuracy_results.csv', ROOT / '.venv/pyvenv.cfg']
    paths += list((ROOT / 'experiments/runs').rglob('*'))
    paths += list((ROOT / 'experiments/data').glob('*'))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}


def production_parity():
    """Compare all demo inputs without fitting or overwriting original evidence."""
    import tensorflow as tf
    from backend.services.classifier import dataset, preprocess
    from utils.model_backend import NumPyCNNEngine
    with zipfile.ZipFile(ROOT / 'satellite_modulation_cnn_all_snr.keras') as archive:
        config = json.loads(archive.read('config.json'))
        # Match scripts/verify_keras_parity.py: remove disabled training-only
        # legacy flags in memory; saved config and every tensor stay untouched.
        for layer in config['config']['layers']:
            value = layer['config']
            if layer['class_name'] == 'BatchNormalization':
                assert not value.get('renorm', False)
                for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                    value.pop(key, None)
            for name in ['kernel_initializer', 'bias_initializer']:
                if name in value:
                    for key in ['input_axes', 'output_axes']:
                        value[name]['config'].pop(key, None)
            value.pop('quantization_config', None)
        config.pop('compile_config', None)
        model = tf.keras.models.model_from_json(json.dumps(config))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.weights.h5'
            path.write_bytes(archive.read('model.weights.h5'))
            model.load_weights(path)
    engine = NumPyCNNEngine.from_npz(ROOT / 'models/cnn_weights.npz')
    original = NumPyCNNEngine.from_keras_zip(ROOT / 'satellite_modulation_cnn_all_snr.keras')
    for key in engine.w:
        np.testing.assert_array_equal(engine.w[key], original.w[key])
    frames = np.stack([preprocess(frame) for frame in dataset()['signals']])
    reference = model(frames, training=False).numpy()
    actual = np.stack([engine.predict(frame) for frame in frames])
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
    mismatches = int(np.sum(actual.argmax(1) != reference.argmax(1)))
    assert mismatches == 0
    return {'frames': len(frames), 'maxAbsoluteProbabilityError': float(np.max(abs(actual - reference))),
            'classMismatchCount': mismatches, 'savedWeightTensorsIdentical': True,
            'probabilityTolerance': {'atol': 2e-5, 'rtol': 2e-5}}


def frozen_class_decisions(run):
    import tensorflow as tf
    from scipy.special import softmax
    from training.data import load_dataset
    from utils.preprocessing import prepare_model_input
    root = ROOT / 'experiments/runs' / run
    report = json.loads((root / 'report.json').read_text())
    dataset = load_dataset(ROOT / 'experiments/data/controlled-synthetic-v1.npz')
    indices = report['splits']['test']['indices']
    norm = report['normalization']
    frames = np.stack([prepare_model_input(frame, norm['mean'], norm['std']) for frame in dataset['signals'][indices]])
    model = tf.keras.models.load_model(root / report['artifact']['filename'], compile=False, safe_mode=True)
    actual = softmax(model(frames, training=False).numpy(), axis=1)
    with np.load(root / 'test-predictions.npz', allow_pickle=False) as saved:
        reference = saved['probabilities']
        mismatch = int(np.sum(actual.argmax(1) != reference.argmax(1)))
        error = float(np.max(abs(actual - reference)))
    assert mismatch == 0
    return {'samples': len(actual), 'classMismatchCount': mismatch, 'maxAbsoluteProbabilityError': error}


class Server:
    def __init__(self, data, log, port):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
            self.port = probe.getsockname()[1]
        self.data = data
        self.log = log
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.calls = []

    def request(self, method, path, body=None, *, raw=None, content_type=None, expected=200, authorized=True):
        headers = {'X-Satmod-Token': self.token} if authorized else {}
        if body is not None:
            raw = json.dumps(body, allow_nan=False).encode()
            content_type = 'application/json'
        if content_type:
            headers['Content-Type'] = content_type
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}' + path,
                                         data=raw, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status, payload = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, payload = exc.code, exc.read()
        self.calls.append({'method': method, 'path': path, 'status': status})
        if status != expected:
            raise AssertionError(f'{method} {path}: HTTP {status}; expected {expected}; {payload.decode()[:500]}')
        return json.loads(payload)

    def start(self):
        environment = {**os.environ, 'SATMOD_DATA_DIR': str(self.data), 'SATMOD_PUBLIC': '1',
                       'SATMOD_ACCESS_TOKEN': self.token, 'SATMOD_MODEL_PATH': str(ROOT / 'models/cnn_weights.npz'),
                       'TF_CPP_MIN_LOG_LEVEL': '2', 'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        self.process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1',
                                         '--port', str(self.port), '--no-access-log', '--log-level', 'warning'],
                                        cwd=ROOT, env=environment, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError('Owned neural API exited before becoming healthy; inspect its adjacent server log.')
            try:
                self.request('GET', '/api/health')
                return
            except urllib.error.URLError:
                time.sleep(.1)
        raise AssertionError('Owned neural API startup exceeded 90 seconds.')

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def fixture(self):
        entries = []
        frames = []
        for index, label in enumerate(['BPSK', 'QPSK', '16-QAM']):
            generated = self.request('POST', '/api/signal/generate', {
                'modulation': label, 'numSamples': 128, 'sampleRate': 8000,
                'seed': 99101 + index, 'snrDb': 6., 'amplitudeRms': .008584335,
            })
            frame = np.array([[point['i'], point['q']] for point in generated['samples']], dtype='<f4')
            stream = io.BytesIO()
            np.save(stream, frame, allow_pickle=False)
            boundary = 'satmod-neural-fixture-boundary'
            options = json.dumps({'sessionId': f'neural-runtime-synthetic-{index}', 'sampleRate': 8000,
                                  'source': f'Fresh seeded synthetic {label} runtime fixture, not real RF',
                                  'license': 'Generated integration fixture'})
            multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="options"\r\n\r\n{options}\r\n'
                         f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic-{index}.npy"\r\n'
                         'Content-Type: application/octet-stream\r\n\r\n').encode() + stream.getvalue() + f'\r\n--{boundary}--\r\n'.encode()
            capture = self.request('POST', '/api/workspace/captures', raw=multipart,
                                   content_type=f'multipart/form-data; boundary={boundary}')
            entries.append({'captureId': capture['id'], 'start': 0, 'label': label, 'split': 'test'})
            frames.append(frame)
        dataset = self.request('POST', '/api/workspace/datasets', {
            'name': 'Fresh synthetic optional neural runtime verification',
            'source': 'Three seeded generated educational frames, not independent satellite observations',
            'license': 'Generated integration fixture', 'entries': entries,
        })
        return dataset, frames

    def evaluation(self, dataset, frames, model):
        from backend.services.research_evaluation import resolve_model
        submitted = self.request('POST', '/api/research/evaluations', {
            'datasetId': dataset['id'], 'modelId': model['id'], 'modelVersion': model['version'],
        }, expected=202)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = self.request('GET', f'/api/workspace/jobs/{submitted["id"]}')
            if job['state'] == 'completed':
                break
            if job['state'] in ['failed', 'cancelled']:
                raise AssertionError(f'{model["id"]}: actual held-out inference {job["state"]}: {job.get("error")}')
            time.sleep(.1)
        else:
            raise AssertionError(f'{model["id"]}: held-out inference exceeded its bounded verification deadline.')
        report = self.request('GET', f'/api/workspace/jobs/{job["id"]}/result')
        assert report['dataset']['hash'] == dataset['hash']
        assert report['modelHash'] == model['artifactSha256']
        assert report['modelVersion'] == model['version']
        assert report['classOrder'] == ['BPSK', 'QPSK', '16-QAM']
        assert report['evaluationProtocol']['split'] == 'test'
        assert report['metrics']['samples'] == len(frames) == 3
        assert report['metrics']['snrAccuracy'] == []  # Generation metadata was not declared as a capture SNR measurement.
        context = resolve_model(model['id'], model['version'])
        errors = []
        for frame, prediction in zip(frames, report['predictions']):
            actual = np.array([prediction['probabilities'][label] for label in report['classOrder']])
            reference = context.predict(frame)
            expected = np.array([reference['probabilities'][label] for label in report['classOrder']])
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
            errors.append(float(np.max(abs(actual - expected))))
        return {'jobId': job['id'], 'modelId': report['modelId'], 'modelVersion': report['modelVersion'],
                'modelHash': report['modelHash'], 'backend': report['backend'], 'datasetHash': dataset['hash'],
                'protocol': report['evaluationProtocol'], 'preprocessing': report['preprocessingConfiguration'],
                'samples': len(report['predictions']), 'maxApiVersusTrustedLoaderProbabilityError': max(errors),
                'predictions': [{key: prediction[key] for key in ['groundTruth', 'predictedModulation', 'probabilities']}
                                for prediction in report['predictions']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'integration/completion/neural-runtime-verification.json')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ROOT / 'integration/completion'):
        raise ValueError('Verification evidence must remain under integration/completion.')
    log_path = output.with_suffix('.log')
    if output.exists() or log_path.exists():
        raise ValueError('Refusing to overwrite existing verification evidence.')
    for key in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    before = preservation()
    result = {'schemaVersion': 'satmod-completion-neural-runtime-v1', 'passed': False,
              'python': platform.python_version(), 'platform': platform.platform(),
              'scope': 'Existing trusted checkpoints only; no training, fitting, promotion, benchmark, or independent RF accuracy claim.',
              'preservedBefore': before}
    code = 0
    with log_path.open('x') as log, tempfile.TemporaryDirectory(prefix='satmod-neural-verification-') as data:
        server = Server(Path(data), log, args.port)
        try:
            from scripts.evaluate import evaluate_run
            import tensorflow as tf
            import keras
            import sklearn
            import scipy
            result['versions'] = {'tensorflow': tf.__version__, 'keras': keras.__version__, 'numpy': np.__version__,
                                  'scikitLearn': sklearn.__version__, 'scipy': scipy.__version__}
            result['originalProductionCnnParity'] = production_parity()
            print('Production CNN original-weight parity:', result['originalProductionCnnParity'], flush=True)
            result['originalRecordedReplays'] = {}
            for model in ['resnet-synthetic-v1', 'cldnn-synthetic-v1']:
                replay = evaluate_run(ROOT / 'experiments/runs' / model, ROOT / 'experiments/data/controlled-synthetic-v1.npz')
                replay['individualClassDecisions'] = frozen_class_decisions(model)
                replay['probabilityTolerance'] = {'atol': 1e-6, 'rtol': 1e-7}
                result['originalRecordedReplays'][model] = replay
                print(model, 'frozen test replay maximum probability error:', replay['maxReplayProbabilityError'], flush=True)
            server.start()
            server.request('GET', '/api/research/models', expected=401, authorized=False)
            catalog = server.request('GET', '/api/research/models')
            selected = [model for model in catalog if model['id'] in ['1d-cnn', 'rf-experimental', 'resnet-synthetic-v1', 'cldnn-synthetic-v1']]
            assert len(selected) == 4 and all(model['available'] for model in selected), catalog
            result['catalog'] = selected
            dataset, frames = server.fixture()
            result['actualApiEvaluations'] = [server.evaluation(dataset, frames, model) for model in selected]
            compared = server.request('POST', '/api/research/evaluations/compare', {
                'jobIds': [item['jobId'] for item in result['actualApiEvaluations']]})
            assert len(compared['evaluations']) == 4
            result['matchedComparison'] = {'evaluationProtocol': compared['evaluationProtocol'],
                                         'models': [item['modelId'] for item in compared['evaluations']]}
            result['passed'] = True
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            print('FAILED:', type(exc).__name__, str(exc), flush=True)
            code = 1
        finally:
            server.stop()
            result['ownedServerStopped'] = server.process is None or server.process.poll() is not None
            result['httpCalls'] = server.calls
    after = preservation()
    result['preservedAfter'] = after
    result['originalArtifactsAndDefaultEnvironmentUnchanged'] = before == after
    result['temporaryDataRemoved'] = True
    if before != after:
        result['passed'] = False
        code = 1
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'output': str(output.relative_to(ROOT)),
                      'originalArtifactsAndDefaultEnvironmentUnchanged': before == after}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
