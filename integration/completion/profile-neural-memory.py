"""Bounded resource observation: one real API worker, four models, synthetic data.

This measures process RSS and child peak RSS, not throughput or RF performance.
No fitting, persistent user workspace or other running process is modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('neural_verification', Path(__file__).with_name('verify-neural-runtime.py'))
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def rss(process):
    status = Path(f'/proc/{process.pid}/status')
    if status.is_file():
        value = next(line.split()[1] for line in status.read_text().splitlines() if line.startswith('VmRSS:'))
        return int(value) * 1024
    raw = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                         check=True, capture_output=True, text=True).stdout.strip()
    return int(raw) * 1024  # macOS and Linux ps report RSS in KiB.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'integration/completion/neural-memory.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'integration/completion'):
        raise ValueError('Evidence must remain under integration/completion.')
    if output.exists() or output.with_suffix('.log').exists():
        raise ValueError('Refusing to overwrite previous memory evidence.')
    before = VERIFY.preservation()
    result = {'schemaVersion': 'satmod-optional-neural-memory-v1', 'passed': False,
              'platform': platform.platform(), 'python': platform.python_version(), 'apiWorkers': 1,
              'scope': 'Process RSS after health/model loading/actual 3-frame evaluation by all four models; child high-water RSS. No concurrent load or hosting headroom claim.'}
    with output.with_suffix('.log').open('x') as log, tempfile.TemporaryDirectory(prefix='satmod-neural-memory-') as data:
        server = VERIFY.Server(Path(data), log, 0)
        try:
            server.start()
            result['rssAfterHealthBytes'] = rss(server.process)
            catalog = server.request('GET', '/api/research/models')
            selected = [model for model in catalog if model['id'] in ['1d-cnn', 'rf-experimental', 'resnet-synthetic-v1', 'cldnn-synthetic-v1']]
            assert len(selected) == 4 and all(model['available'] for model in selected), selected
            result['rssAfterAllModelsLoadedBytes'] = rss(server.process)
            result['models'] = [model['id'] for model in selected]
            dataset, _ = server.fixture()
            for model in selected:
                job = server.request('POST', '/api/research/evaluations', {
                    'datasetId': dataset['id'], 'modelId': model['id'], 'modelVersion': model['version']}, expected=202)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    state = server.request('GET', f'/api/workspace/jobs/{job["id"]}')
                    if state['state'] == 'completed':
                        break
                    assert state['state'] not in ['failed', 'cancelled'], state
                    time.sleep(.1)
                else:
                    raise AssertionError('A real model did not complete the memory fixture within its deadline.')
            result['rssAfterAllFourEvaluationsBytes'] = rss(server.process)
            result['passed'] = True
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            server.stop()
            result['ownedServerStopped'] = server.process is None or server.process.poll() is not None
            high_water = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            result['apiChildPeakRssBytes'] = int(high_water * (1 if platform.system() == 'Darwin' else 1024))
            result['peakMeasurement'] = 'getrusage(RUSAGE_CHILDREN).ru_maxrss; includes the API child and tiny ps children; API is the dominant child.'
    result['originalArtifactsAndDefaultEnvironmentUnchanged'] = before == VERIFY.preservation()
    result['temporaryDataRemoved'] = True
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] and result['originalArtifactsAndDefaultEnvironmentUnchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
