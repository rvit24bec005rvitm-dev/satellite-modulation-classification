"""Real import-worker lifecycle checks using explicitly synthetic recordings."""
import io
import json
import threading
import time
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import captures, classifier, operations, workspace_store as store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR', str(tmp_path / 'workspace'))
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('SATMOD_PUBLIC', raising=False)
    with TestClient(app) as session:
        yield session
    runner = operations.RUNNERS.pop(str(store.root()), None)
    if runner:
        runner[0].shutdown(wait=True)
    store.close_connections()


def recording():
    return (np.random.default_rng(581).normal(size=(150000, 2)) * .006).astype('<f4')


def submit(client, samples=None, options=None):
    stream = io.BytesIO()
    np.save(stream, recording() if samples is None else samples, allow_pickle=False)
    return client.post('/api/workspace/captures?background=true',
                       data={'options': json.dumps({'sessionId': 'synthetic-import-test', **(options or {})})},
                       files={'file': ('synthetic.npy', stream.getvalue())})


def finished(client, job):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = client.get('/api/workspace/jobs/' + job['id']).json()
        if current['state'] in ['completed', 'failed', 'cancelled']:
            return current
        time.sleep(.02)
    raise AssertionError('Import worker did not terminate')


def test_background_import_window_provenance_and_job_deletion(client):
    raw = recording()
    response = submit(client, raw[:, 0] + 1j * raw[:, 1],
                      {'sampleRate': 8000, 'sampleOffset': 100, 'sampleCount': 140000})
    assert response.status_code == 202, response.text
    job = finished(client, response.json())
    assert job['state'] == 'completed' and job['progress'] == 1
    assert job['phase'] == 'complete' and job['id'] != job['captureId']
    report = client.get(f"/api/workspace/jobs/{job['id']}/result").json()
    assert report['metadata']['sourceOffset'] == 100
    window, meta = captures.window(job['captureId'], 139872)
    np.testing.assert_array_equal(window, raw[139972:140100])
    assert classifier.predict_window(window)['probabilities'] == classifier.predict_window(raw, 139972)['probabilities']
    assert not list(store.root().glob('*.upload'))
    assert client.get(f"/api/workspace/jobs/{job['id']}/export").json()['captureId'] == meta['id']
    assert client.get(f"/api/workspace/jobs/{job['id']}/export?format=csv").status_code == 422
    assert client.delete('/api/workspace/jobs/' + job['id']).status_code == 200
    assert store.file_path(meta['id'], 'iq').exists()
    assert client.get('/api/workspace/captures/' + meta['id']).status_code == 200


def test_background_metadata_error_is_failed_with_cleanup(client):
    response = submit(client, options={'format': 'raw'})
    assert response.status_code == 202
    job = finished(client, response.json())
    assert job['state'] == 'failed' and 'datatype' in job['error']
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))
    assert client.get('/api/workspace/captures').json()['total'] == 0


def test_running_import_is_pollable_cancellable_and_cleans_files(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = captures.finite_block
    def slow(block):
        entered.set()
        assert release.wait(10)
        return real(block)
    monkeypatch.setattr(captures, 'finite_block', slow)
    response = submit(client)
    assert response.status_code == 202 and entered.wait(5)
    job = response.json()
    try:
        assert client.get('/api/workspace/jobs/' + job['id']).json()['state'] == 'running'
        assert client.post('/api/workspace/jobs/' + job['id'] + '/cancel').json()['cancelRequested']
    finally:
        release.set()
    assert finished(client, job)['state'] == 'cancelled'
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))
    assert client.get('/api/workspace/captures').json()['total'] == 0


def test_import_capacity_rejects_before_staging_and_queued_cancel_cleans(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = captures.finite_block
    def slow(block):
        entered.set()
        assert release.wait(10)
        return real(block)
    monkeypatch.setattr(captures, 'finite_block', slow)
    jobs = []
    try:
        for _ in range(4):
            response = submit(client, recording()[:128])
            assert response.status_code == 202
            jobs.append(response.json())
        assert entered.wait(5)
        before = set(store.root().glob('*.upload'))
        assert submit(client, recording()[:128]).status_code == 429
        assert set(store.root().glob('*.upload')) == before
        for job in jobs:
            client.post('/api/workspace/jobs/' + job['id'] + '/cancel')
    finally:
        release.set()
    assert all(finished(client, job)['state'] == 'cancelled' for job in jobs)
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))


def test_restart_recovers_partial_import_and_committed_capture(client):
    complete = finished(client, submit(client, recording()[:256]).json())
    assert complete['state'] == 'completed'
    operations.RUNNERS.pop(str(store.root()))[0].shutdown(wait=True)
    staged_id = str(uuid4())
    store.file_path(staged_id, 'upload').write_bytes(b'interrupted source')
    store.file_path(staged_id, 'iq').write_bytes(b'partial canonical samples')
    partial = store.put('job', {'kind': 'capture-import', 'state': 'running', 'progress': .2,
                               'cancelRequested': False, 'params': {'staged': {'id': staged_id}}})
    original = store.get('job', complete['id'])
    store.put('job', {**original, 'state': 'running', 'captureId': None})
    orphan = str(uuid4())
    store.file_path(orphan, 'upload').write_bytes(b'staged before job registration')
    assert client.get('/api/workspace/jobs/' + partial['id']).json()['state'] == 'failed'
    recovered = client.get('/api/workspace/jobs/' + complete['id']).json()
    assert recovered['state'] == 'completed' and recovered['captureId'] == complete['captureId']
    assert store.file_path(complete['captureId'], 'iq').exists()
    assert not store.file_path(staged_id, 'iq').exists()
    assert not list(store.root().glob('*.upload'))


def test_overview_uses_actual_bin_means_and_releases_mapping(client):
    raw = recording()
    job = finished(client, submit(client, raw).json())
    overview = client.get('/api/workspace/captures/' + job['captureId']).json()
    width = int(np.ceil(len(raw) / 512))
    expected = np.sum(raw[:width].astype(float)**2, axis=1)
    assert overview['points'][0]['power'] == pytest.approx(expected.mean())
    assert overview['points'][0]['peakPower'] == pytest.approx(expected.max())
    assert len(overview['points']) <= 512
    assert client.delete('/api/workspace/captures/' + job['captureId']).status_code == 200
    assert not list(store.root().glob('*.iq'))


def test_background_dataset_registers_labels_and_evaluates_real_predictions(client):
    capture = finished(client, submit(client, recording()[:512]).json())['captureId']
    manifest = {'name': 'Synthetic async labels', 'source': 'Generated fixture, not RF', 'license': 'Test',
                'entries': [{'captureId': capture, 'start': i * 128, 'label': label, 'split': 'test', 'snrDb': 6.0}
                            for i, label in enumerate(['BPSK', 'QPSK', '16-QAM'])]}
    response = client.post('/api/workspace/datasets?background=true', json=manifest)
    assert response.status_code == 202, response.text
    imported = finished(client, response.json())
    assert imported['state'] == 'completed' and imported['progress'] == 1
    registered = client.get('/api/workspace/jobs/' + imported['id'] + '/result').json()['dataset']
    assert [e['label'] for e in registered['entries']] == ['BPSK', 'QPSK', '16-QAM']
    evaluated = finished(client, client.post('/api/workspace/datasets/' + registered['id'] + '/evaluate').json())
    assert evaluated['state'] == 'completed'
    result = client.get('/api/workspace/jobs/' + evaluated['id'] + '/result').json()
    assert result['metrics']['snrAccuracy'][0]['snrDb'] == 6.0
    assert result['metrics']['samples'] == 3
    assert client.get('/api/workspace/datasets?search=async&limit=1').json()['total'] == 1
    assert client.get('/api/workspace/datasets?offset=1&limit=1').json()['items'] == []
    # A later registration with contrary split membership is rejected by the
    # same indexed checks as an ordinary synchronous registration.
    manifest['entries'][0]['split'] = 'train'
    manifest['entries'] = manifest['entries'][:1]
    rejected = finished(client, client.post('/api/workspace/datasets?background=true', json=manifest).json())
    assert rejected['state'] == 'failed' and 'split' in rejected['error']
    assert client.get('/api/workspace/datasets').json()['total'] == 1


def test_dataset_cancel_preserves_no_partial_registry_entry(client, monkeypatch):
    capture = finished(client, submit(client, recording()[:256]).json())['captureId']
    entered, release = threading.Event(), threading.Event()
    real = captures.window
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(captures, 'window', slow)
    manifest = {'name': 'Cancelled fixture', 'source': 'Generated fixture', 'license': 'Test',
                'entries': [{'captureId': capture, 'start': i * 128, 'label': 'BPSK', 'split': 'test'} for i in range(2)]}
    response = client.post('/api/workspace/datasets?background=true', json=manifest)
    assert response.status_code == 202 and entered.wait(5)
    try:
        assert client.post('/api/workspace/jobs/' + response.json()['id'] + '/cancel').json()['cancelRequested']
    finally:
        release.set()
    assert finished(client, response.json())['state'] == 'cancelled'
    assert client.get('/api/workspace/datasets').json()['total'] == 0


def test_split_guard_checks_both_sides_of_interval_boundaries():
    def entry(start, split):
        return {'sourceHash': 'same recording', 'sourceStart': start, 'start': start,
                'frameHash': str(start), 'sessionId': str(start), 'split': split}
    guard = operations.SplitGuard()
    guard.add(entry(256, 'test'))
    guard.add(entry(128, 'train'))  # Adjacent fixed-length windows do not overlap.
    with pytest.raises(ValueError, match='Overlapping'):
        guard.add(entry(129, 'train'))
    guard = operations.SplitGuard()
    guard.add(entry(128, 'train'))
    with pytest.raises(ValueError, match='Overlapping'):
        guard.add(entry(255, 'test'))


def test_batch_exports_preserve_probability_vectors_identity_and_pdf(client):
    import csv
    capture = finished(client, submit(client, recording()[:512]).json())['captureId']
    job = finished(client, client.post('/api/workspace/jobs/batch', json={'captureId': capture, 'windows': 3}).json())
    report = client.get('/api/workspace/jobs/' + job['id'] + '/export').json()
    exported = client.get('/api/workspace/jobs/' + job['id'] + '/export?format=csv')
    records = list(csv.DictReader(io.StringIO(exported.text)))
    assert len(records) == 3
    for row, prediction in zip(records, report['predictions']):
        assert json.loads(row['probabilities']) == prediction['probabilities']
        assert json.loads(row['preprocessing']) == prediction['preprocessing']
        assert row['modelHash'] == report['modelHash']
        assert row['captureHash'] == report['captures'][capture]['hash']
        assert row['sampleRateHz'] == ''  # Unknown is never silently reported as 1 Hz.
        assert row['groundTruth'] == '' and row['snrDb'] == ''
    pdf = client.get('/api/workspace/jobs/' + job['id'] + '/export?format=pdf')
    assert pdf.status_code == 200 and pdf.headers['content-type'] == 'application/pdf'
    assert pdf.content.startswith(b'%PDF') and pdf.content.rstrip().endswith(b'%%EOF')
    assert report['modelHash'].encode() in pdf.content and job['id'].encode() in pdf.content
    assert b'satmod-technical-pdf-v1' in pdf.content


def test_normalized_cfo_never_reports_a_hz_interval():
    from backend.services.advanced_dsp import diagnostics
    z = np.exp(2j * np.pi * .04 * np.arange(256))
    result = diagnostics(np.column_stack((z.real, z.imag)), 1, 'BPSK')['frequencyOffset']
    assert result['unit'] == 'cycles/sample'
    assert result['estimateHz'] is None and result['ambiguityIntervalHz'] is None
    assert result['ambiguityInterval'] == [-.25, .25]


def test_missing_model_is_service_unavailable_without_private_paths(client, monkeypatch):
    capture = finished(client, submit(client, recording()[:256]).json())['captureId']
    def missing():
        raise RuntimeError('Missing /private/secret-model-root/checkpoint.npz')
    monkeypatch.setattr(classifier, 'model', missing)
    for path in ['/api/workspace/captures/' + capture + '/classify', '/api/classify']:
        kwargs = {'json': {'samples': [{'i': .001, 'q': .002}] * 128}} if path == '/api/classify' else {}
        response = client.post(path, **kwargs)
        assert response.status_code == 503
        assert '/private/' not in response.text and 'checkpoint.npz' not in response.text
    assert client.get('/api/workspace/history').json()['total'] == 0


def test_slow_upload_times_out_and_releases_import_admission(client, monkeypatch):
    import asyncio
    import httpx
    monkeypatch.setenv('SATMOD_BODY_TIMEOUT_SECONDS', '.01')
    async def request(path):
        async def chunks():
            yield b'partial'
            await asyncio.sleep(.2)
            yield b'incomplete'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as session:
            return await session.post(path, content=chunks())
    for path in ['/api/workspace/captures', '/api/workspace/captures', '/api/classify']:
        response = asyncio.run(request(path))
        assert response.status_code == 408 and 'timed out' in response.text
        assert response.headers['x-request-id']
    monkeypatch.setenv('SATMOD_BODY_TIMEOUT_SECONDS', '120')
    assert finished(client, submit(client, recording()[:128]).json())['state'] == 'completed'
