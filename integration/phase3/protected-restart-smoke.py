#!/usr/bin/env python3
"""Real protected HTTP/restart smoke on synthetic fixtures and a private temp store.

Run from any directory with the repository environment's Python. This starts and
stops only its own uvicorn subprocess; it never touches the usual workspace or a
browser. The generated access token stays in process memory/environment only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'integration' / 'phase3'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def available_port(requested):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', requested))
        return listener.getsockname()[1]


class NativeSmoke:
    def __init__(self, port, data_dir, token, log, result):
        self.port = port
        self.data_dir = data_dir
        self.token = token
        self.log = log
        self.result = result
        self.process = None
        self.stage = 'initial'
        self.environment = {
            **os.environ,
            'SATMOD_DATA_DIR': str(data_dir),
            'SATMOD_PUBLIC': '1',
            'SATMOD_ACCESS_TOKEN': token,
            'SATMOD_MODEL_PATH': str(ROOT / 'models' / 'cnn_weights.npz'),
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'PYTHONUNBUFFERED': '1',
        }
        self.http = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=90, trust_env=False)

    def note(self, message):
        self.log.write(f'{datetime.now(timezone.utc).isoformat()} smoke: {message}\n')
        self.log.flush()

    def start(self):
        require(self.process is None, 'A smoke-owned server is already running.')
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1',
             '--port', str(self.port), '--no-access-log', '--log-level', 'warning'],
            cwd=ROOT, env=self.environment, stdout=self.log, stderr=subprocess.STDOUT,
        )
        self.result['processes'].append({'pid': self.process.pid, 'stage': self.stage, 'stopped': False})
        self.note(f'started owned server PID {self.process.pid}, stage {self.stage}')
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            require(self.process.poll() is None, 'Owned server exited before becoming healthy; see the adjacent log.')
            try:
                response = self.http.get('/api/health', timeout=1)
                if response.status_code == 200:
                    self.note('owned server became healthy')
                    return
            except httpx.TransportError:
                pass
            time.sleep(.1)
        raise AssertionError('Owned server did not become healthy within 60 seconds.')

    def stop(self):
        if self.process is None:
            return
        process = self.process
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self.result['processes'][-1].update(stopped=True, exitCode=process.returncode)
            self.note(f'stopped owned server PID {process.pid}, exit {process.returncode}')
        finally:
            self.process = None

    def request(self, method, path, *, authorized=True, expected=200, **kwargs):
        response = self.http.request(
            method, path, headers={'X-Satmod-Token': self.token} if authorized else {}, **kwargs)
        self.result['requests'].append({
            'stage': self.stage, 'method': method, 'path': path,
            'authorized': authorized, 'status': response.status_code,
        })
        require(response.status_code == expected,
                f'{method} {path}: expected HTTP {expected}, received {response.status_code}.')
        return response

    def authorize(self):
        self.request('GET', '/api/research/models', authorized=False, expected=401)
        # Middleware protects writes too; this must never execute explanation work.
        self.request('POST', '/api/research/comparisons', authorized=False, expected=401, json={})
        catalog = self.request('GET', '/api/research/models').json()
        production = next(model for model in catalog if model['id'] == '1d-cnn')
        require(production['available'] is True, 'The genuine production checkpoint is unavailable.')
        self.result['catalog'] = {
            'productionId': production['id'], 'productionVersion': production['version'],
            'productionArtifactSha256': production['artifactSha256'],
            'modelCount': len(catalog),
        }

    def fixture(self, seed, name, start):
        signal = self.request('POST', '/api/signal/generate', json={
            'modulation': 'QPSK', 'numSamples': 1024, 'sampleRate': 8000, 'seed': seed,
        }).json()
        values = np.asarray([[p['i'], p['q']] for p in signal['samples']], dtype='<f4')
        imported = self.request('POST', '/api/workspace/captures', files={
            'file': (f'synthetic-restart-{name}.bin', values.tobytes(), 'application/octet-stream'),
        }, data={'options': json.dumps({
            'format': 'raw', 'datatype': 'cf32_le', 'sampleRate': 8000,
            'sessionId': f'phase3-native-restart-{name}-{seed}',
            'source': f'Synthetic native restart fixture {name}, seed {seed}; not independent RF',
            'license': 'Generated smoke-test fixture',
        })}).json()
        classified = self.request('POST', f'/api/workspace/captures/{imported["id"]}/classify?start={start}').json()
        explained = self.request('POST', f'/api/research/analyses/{classified["id"]}/explanations', json={
            'method': 'both', 'targetClass': 'QPSK', 'steps': 16,
        }).json()
        require({item['method'] for item in explained['explanations']} == {'saliency', 'integrated-gradients'},
                'The server did not save both actual explanation methods.')
        require(explained['prediction']['probabilities'] == classified['probabilities'],
                'Saved explanation prediction differs from the actual classification.')
        require(all(len(item['samples']) == 128 for item in explained['explanations']),
                'Explanation samples were not retained at full resolution.')
        for item in explained['explanations']:
            require(all(math.isfinite(point[key]) for point in item['samples'] for key in ['i', 'q', 'rawI', 'rawQ']),
                    'An actual explanation contains a non-finite signed channel value.')
            require(item['modelHash'] == explained['modelHash'] and item['inputHash'] == explained['inputHash'],
                    'Explanation model/input identity differs from the retained analysis.')
        self.note(f'created synthetic managed capture {name}, real classification and both methods at 16 steps')
        return {'capture': imported, 'analysis': classified, 'explanations': explained, 'samples': signal['samples']}

    def exports(self, fixtures, comparison):
        saved = {}
        for name, fixture in fixtures.items():
            analysis_id = fixture['analysis']['id']
            stored = self.request('GET', f'/api/research/analyses/{analysis_id}/explanations').json()
            require(stored == fixture['explanations'], 'Saved full explanation data changed.')
            for kind in ['json', 'pdf']:
                raw = self.request('GET', f'/api/workspace/history/{analysis_id}/report?format={kind}').content
                if kind == 'pdf':
                    require(raw.startswith(b'%PDF'), 'History PDF is not a real PDF document.')
                else:
                    report = json.loads(raw)
                    require(report['explanations'] == stored['explanations'], 'Full history JSON lost signed explanation data.')
                    require(report['modelHash'] == stored['modelHash'], 'Full history JSON changed model identity.')
                saved[f'history{name}.{kind}'] = {'sha256': digest(raw), 'bytes': len(raw)}
        persisted = self.request('GET', f'/api/research/comparisons/{comparison["id"]}').json()
        require(persisted == comparison, 'Saved managed A/B comparison changed.')
        for kind in ['json', 'pdf']:
            raw = self.request('GET', f'/api/research/comparisons/{comparison["id"]}/export?format={kind}').content
            require(raw.startswith(b'%PDF') if kind == 'pdf' else json.loads(raw) == comparison,
                    'Managed comparison export is invalid or differs from the saved comparison.')
            saved[f'comparison.{kind}'] = {'sha256': digest(raw), 'bytes': len(raw)}
        listing = self.request('GET', '/api/workspace/history').json()
        require(listing['total'] == 2, 'Expected exactly two isolated saved analyses.')
        comparisons = self.request('GET', '/api/research/comparisons').json()
        require(comparisons['total'] == 1, 'Expected exactly one isolated saved managed comparison.')
        return saved

    def run(self):
        self.start()
        self.authorize()
        fixtures = {'A': self.fixture(971, 'A', 128), 'B': self.fixture(972, 'B', 256)}
        a, b = fixtures['A'], fixtures['B']
        comparison = self.request('POST', '/api/research/comparisons', json={
            'first': {'captureId': a['capture']['id'], 'start': 128, 'length': 128},
            'second': {'captureId': b['capture']['id'], 'start': 256, 'length': 128},
            'firstAnalysisId': a['analysis']['id'], 'secondAnalysisId': b['analysis']['id'],
        }).json()
        require(comparison['first']['prediction']['probabilities'] == a['analysis']['probabilities'],
                'Managed comparison did not reuse recording A actual saved prediction.')
        require(comparison['second']['prediction']['probabilities'] == b['analysis']['probabilities'],
                'Managed comparison did not reuse recording B actual saved prediction.')
        require(comparison['attributionDifferences']['status'] == 'available',
                'Matching stored explanation methods/targets did not support genuine attribution differences.')
        require(len(comparison['attributionDifferences']['comparisons']) == 2,
                'Both saved methods were not retained in the managed comparison.')
        tf = self.request('POST', '/api/research/time-frequency', json={
            'samples': a['samples'], 'sampleRate': 8000, 'sampleRateKnown': True,
            'fftSize': 128, 'overlap': .5, 'window': 'hann',
        }).json()
        require(tf['frequencyUnit'] == 'Hz' and tf['timeUnit'] == 's', 'Time-frequency clock units are invalid.')
        require(len(tf['db']) == 128 and len(tf['times']) == 15, 'Time-frequency output dimensions differ from complete-frame STFT.')
        require(all(math.isfinite(point) for row in tf['db'] for point in row), 'Time-frequency density is not finite.')
        self.result['timeFrequency'] = {
            'reportVersion': tf['reportVersion'], 'inputHash': tf['inputHash'],
            'frequencyBins': len(tf['db']), 'timeWindows': len(tf['times']), 'parameters': tf['parameters'],
            'responseSha256': digest(json.dumps(tf, sort_keys=True, allow_nan=False).encode()),
        }
        before = self.exports(fixtures, comparison)
        self.result['beforeRestart'] = before
        self.result['fixtureIds'] = {
            name: {'captureId': fixture['capture']['id'], 'analysisId': fixture['analysis']['id'],
                   'inputHash': fixture['explanations']['inputHash'], 'modelHash': fixture['explanations']['modelHash']}
            for name, fixture in fixtures.items()
        }
        self.result['comparisonId'] = comparison['id']
        self.note('captured full-resolution history and managed comparison JSON/PDF hashes before restart')
        self.stop()
        self.stage = 'afterRestart'
        self.start()
        self.request('GET', '/api/research/models', authorized=False, expected=401)
        after = self.exports(fixtures, comparison)
        require(before == after, 'Saved JSON/PDF hashes or lengths changed across a native server restart.')
        require(all(item['method'] == 'GET' for item in self.result['requests'] if item['stage'] == 'afterRestart'),
                'The restart verification unexpectedly issued a write or inference request.')
        self.result.update(afterRestart=after, byteIdenticalAcrossRestart=True, onlyReadRequestsAfterRestart=True,
                           savedExplanationMethods=['integrated-gradients', 'saliency'], integrationSteps=16)
        self.note('restart verified: all six JSON/PDF exports byte-identical; only GET requests after restart')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18591, help='Unused localhost TCP port; 0 chooses a free port.')
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'protected-restart-smoke.json')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    require(EVIDENCE == output.parent or EVIDENCE in output.parents, 'Evidence must remain inside integration/phase3.')
    log_path = output.with_suffix('.log')
    require(not output.exists() and not log_path.exists(), 'Refusing to overwrite an existing evidence result/log.')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'schemaVersion': 'satmod-phase3-native-protected-restart-v1',
        'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
        'fixture': 'Two seeded generated QPSK recordings; synthetic integration fixtures, not independently acquired RF.',
        'scope': 'Actual protected native HTTP server, managed storage, production CNN inference, real explanations and DSP; no browser, training or promotion.',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'SATMOD_PUBLIC': '1', 'tokenConfigured': True, 'tokenLogged': False,
                        'temporaryIsolatedDataDirectory': True, 'inferenceThreads': 1},
        'requests': [], 'processes': [],
    }
    code = 0
    with log_path.open('x') as log:
        with tempfile.TemporaryDirectory(prefix='satmod-phase3-native-restart-') as folder:
            smoke = NativeSmoke(available_port(args.port), Path(folder), secrets.token_urlsafe(32), log, result)
            result['environment']['port'] = smoke.port
            try:
                smoke.run()
                result['passed'] = True
            except Exception as exc:
                result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                smoke.note(f'FAILED: {type(exc).__name__}: {exc}')
                code = 1
            finally:
                smoke.stop()
                smoke.http.close()
        result['temporaryDataCleaned'] = True
        result['ownedProcessesStopped'] = all(item['stopped'] for item in result['processes'])
        result['finishedAt'] = datetime.now(timezone.utc).isoformat()
        log.write(f'Result: {"PASS" if result["passed"] else "FAIL"}; owned processes stopped and isolated data cleaned.\n')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'output': str(output.relative_to(ROOT)),
                      'log': str(log_path.relative_to(ROOT)), 'ownedProcessesStopped': result['ownedProcessesStopped']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
