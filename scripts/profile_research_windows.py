"""Reproduce a real-CNN worker/reader ablation on explicit synthetic fixtures.

This is an offline benchmark, not training or independent RF validation. It uses
a temporary workspace and never changes model/data artifacts or application
history. The baseline invokes the previous captures.window path for each frame;
the candidate invokes the bounded job-local mapping cache in the same worker.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import captures, operations, classifier, workspace_store as store, capture_windows


class PreviousWindowReader:
    """Call the genuine existing window function, including its close behavior."""
    def window(self, capture_id, start):
        return captures.window(capture_id, start, 128)

    def close(self):
        pass


def completed(job):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status = store.get('job', job['id'])
        if status['state'] in ['completed', 'failed', 'cancelled']:
            if status['state'] != 'completed':
                raise RuntimeError('Benchmark worker did not complete: ' + status.get('error', status['state']))
            return status
        time.sleep(.01)
    operations.cancel(job['id'])
    raise TimeoutError('Benchmark job exceeded its bounded wait; cancellation requested.')


def profile():
    actual_reader = capture_windows.CaptureWindows
    previous_directory = os.environ.get('SATMOD_DATA_DIR')
    results = {'previousPerWindowMmap': [], 'boundedJobLru': []}
    probability_pairs = []
    with tempfile.TemporaryDirectory(prefix='satmod-research-reader-profile-') as directory:
        os.environ['SATMOD_DATA_DIR'] = directory
        worker_key = str(store.root())
        try:
            operations.runner()
            raw = np.random.default_rng(8386).normal(0, .006, (65536, 2)).astype(np.float32)
            source = io.BytesIO()
            np.save(source, raw, allow_pickle=False)
            source.seek(0)
            capture = captures.import_capture(
                SimpleNamespace(file=source, filename='explicit-synthetic-profile.npy'),
                captures.ImportOptions(sessionId='synthetic-reader-profile',
                                       source='Seeded synthetic profiling fixture; not satellite RF'),
            )
            for _ in range(8):
                classifier.predict_window(raw)
            for pair in range(3):
                values = {}
                for mode, reader in [('previousPerWindowMmap', PreviousWindowReader), ('boundedJobLru', actual_reader)]:
                    capture_windows.CaptureWindows = reader
                    completed(operations.submit('batch', {'captureId': capture['id'], 'start': 0, 'windows': 32, 'stride': 128}))
                    begin = time.perf_counter()
                    job = completed(operations.submit('batch', {'captureId': capture['id'], 'start': 0, 'windows': 512, 'stride': 128}))
                    elapsed = time.perf_counter() - begin
                    report = json.loads(store.file_path(job['id'], 'json').read_text())
                    probabilities = np.array([[p['probabilities'][label] for label in ['BPSK', 'QPSK', '16-QAM']]
                                              for p in report['predictions']])
                    values[mode] = probabilities
                    results[mode].append({'pair': pair, 'seconds': elapsed, 'windowsPerSecond': 512 / elapsed,
                                          'probabilitiesSha256': hashlib.sha256(probabilities.tobytes()).hexdigest()})
                error = float(np.max(np.abs(values['previousPerWindowMmap'] - values['boundedJobLru'])))
                np.testing.assert_allclose(values['previousPerWindowMmap'], values['boundedJobLru'], atol=1e-7, rtol=0)
                probability_pairs.append(error)
            backend = classifier.model()
            try:
                from threadpoolctl import threadpool_info
                pools = [{key: item[key] for key in ['internal_api', 'num_threads', 'version', 'architecture'] if key in item}
                         for item in threadpool_info()]
            except ImportError:
                pools = None
            return {
                'scope': 'Controlled three-pair ablation using the same current worker, cached metadata, model and seeded synthetic capture. '
                         'Baseline uses the original captures.window per frame; candidate uses the bounded eight-map LRU. '
                         'Includes result serialization; excludes upload, generation and HTTP. Timings vary with host load and are not throughput guarantees or RF accuracy.',
                'configuration': {'samples': 65536, 'windows': 512, 'stride': 128, 'pairs': 3,
                                  'seed': 8386, 'warmupInference': 8, 'warmupWindowsPerMode': 32, 'maxCachedMappings': 8},
                'platform': {'os': platform.platform(), 'machine': platform.machine(),
                             'python': platform.python_version(), 'numpy': np.__version__, 'threadpools': pools},
                'backend': backend.backend_name, 'modelHash': backend.artifact_sha256,
                'measurements': results,
                'medianWindowsPerSecond': {mode: float(np.median([row['windowsPerSecond'] for row in rows]))
                                           for mode, rows in results.items()},
                'maxProbabilityErrorAll512Windows': max(probability_pairs),
            }
        finally:
            capture_windows.CaptureWindows = actual_reader
            for job in store.all_records('job'):
                if job.get('state') in ['queued', 'running']:
                    operations.cancel(job['id'])
            runner = operations.RUNNERS.pop(worker_key, None)
            if runner:
                runner[0].shutdown(wait=True)
            store.close_connections()
            if previous_directory is None:
                os.environ.pop('SATMOD_DATA_DIR', None)
            else:
                os.environ['SATMOD_DATA_DIR'] = previous_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='integration/profiles/research-windows.json',
                        help='New benchmark evidence file; existing evidence is never overwritten.')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Choose a new output path; existing benchmark evidence is preserved.')
    result = profile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'medianWindowsPerSecond': result['medianWindowsPerSecond'],
                      'maxProbabilityErrorAll512Windows': result['maxProbabilityErrorAll512Windows']}))


if __name__ == '__main__':
    main()
