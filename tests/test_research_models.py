"""Phase 3 real inference/export contracts. Every recording here is synthetic."""
import hashlib
import csv
import io
import json
from dataclasses import replace
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import captures, classifier, operations, workspace_store as store
from backend.services.model_registry import REGISTRY, ModelSpec, register_model
from backend.services.research_evaluation import resolve_model
from backend.services.training_exports import deterministic_npz
from training.data import load_dataset, split_dataset
from training.metrics import evaluate_probabilities
from training.runner import run_experiment
from scripts.evaluate import evaluate_run
from utils.signal_processing import generate_iq_signal


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR', str(tmp_path / 'managed'))
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('SATMOD_PUBLIC', raising=False)
    with TestClient(app) as session:
        yield session
    runner = operations.RUNNERS.pop(str(store.root()), None)
    if runner:
        runner[0].shutdown(wait=True)
    store.close_connections()


def import_session(client, group):
    frames = []
    for index, modulation in enumerate(['BPSK', 'QPSK', 'QAM16']):
        snr = 0 if group == 8 and index == 0 else 6
        signal = generate_iq_signal(modulation, num_samples=128, sps=8, snr_db=snr,
                                    rng_seed=10801 + group * 17 + index) * .008584335
        frames.append(np.asarray(signal, dtype='<f4'))
    raw = np.vstack(frames)
    buffer = io.BytesIO()
    np.save(buffer, raw, allow_pickle=False)
    response = client.post('/api/workspace/captures',
                           data={'options': json.dumps({'sessionId': f'synthetic-session-{group}',
                                                        'source': 'Controlled generated test fixture, not actual RF',
                                                        'sampleRate': 8000})},
                           files={'file': (f'synthetic-session-{group}.npy', buffer.getvalue())})
    assert response.status_code == 200, response.text
    return response.json(), raw


def labeled_dataset(client, full_splits=False):
    entries = []
    for group in range(9 if full_splits else 1):
        meta, _ = import_session(client, group)
        split = ['train', 'validation', 'test'][group // 3] if full_splits else 'test'
        for index, label in enumerate(['BPSK', 'QPSK', '16-QAM']):
            entries.append({'captureId': meta['id'], 'start': index * 128, 'label': label, 'split': split,
                            'snrDb': 0.0 if group == 8 and index == 0 else None})
    response = client.post('/api/workspace/datasets', json={
        'name': 'Synthetic controlled fixture', 'source': 'Independent seeded generated groups, not real satellite observations',
        'license': 'Test fixture only', 'entries': entries,
    })
    assert response.status_code == 200, response.text
    return response.json()


def done(client, job):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        value = client.get('/api/workspace/jobs/' + job['id']).json()
        if value['state'] in ['completed', 'failed', 'cancelled']:
            return value
        time.sleep(.02)
    raise AssertionError('Evaluation did not reach a terminal state.')


def evaluate(client, dataset, model='1d-cnn'):
    response = client.post('/api/research/evaluations', json={
        'datasetId': dataset['id'], 'modelId': model, 'modelVersion': REGISTRY[model].version,
    })
    assert response.status_code == 202, response.text
    job = done(client, response.json())
    assert job['state'] == 'completed', job
    return job, client.get('/api/workspace/jobs/' + job['id'] + '/result').json()


def test_catalog_exposes_actual_models_versions_classes_and_availability(client):
    response = client.get('/api/research/models')
    assert response.status_code == 200
    catalog = {item['id']: item for item in response.json()}
    assert {'1d-cnn', 'rf-experimental', 'resnet-synthetic-v1', 'cldnn-synthetic-v1'} <= catalog.keys()
    production = catalog['1d-cnn']
    assert production['available'] and production['status'] == 'Production'
    assert production['classLabels'] == ['BPSK', 'QPSK', '16-QAM']
    assert production['artifactSha256'] == store.artifact_hash()
    for item in catalog.values():
        assert bool(item['available']) == (item['status'] != 'Unavailable')
        if not item['available']:
            assert 'artifactSha256' not in item and item['reason']


def test_production_and_real_rf_use_their_own_preprocessing_and_preserve_provenance(client):
    catalog = {item['id']: item for item in client.get('/api/research/models').json()}
    if not catalog['rf-experimental']['available']:
        pytest.skip('Genuine RF checkpoint is unavailable in this recorded scikit-learn runtime.')
    dataset = labeled_dataset(client)
    jobs = []
    for model_id in ['1d-cnn', 'rf-experimental']:
        job, result = evaluate(client, dataset, model_id)
        jobs.append(job['id'])
        context = resolve_model(model_id)
        assert result['dataset']['hash'] == dataset['hash']
        assert result['modelHash'] == context.metadata['modelHash']
        assert result['modelVersion'] == context.spec.version
        assert result['classOrder'] == list(context.spec.class_labels)
        assert result['evaluationProtocol']['split'] == 'test'
        assert result['metrics']['samples'] == 3 and result['metrics']['snrAccuracy'] == []
        assert result['metrics']['snrAvailability']['unavailableSamples'] == 3
        assert result['preprocessingConfiguration']['mean'] == context.metadata['preprocessingConfiguration']['mean']
        assert result['latency']['inferenceMedianMs'] >= 0
        for entry, prediction in zip(dataset['entries'], result['predictions']):
            iq, _ = captures.window(entry['captureId'], entry['start'])
            expected = context.predict(iq)
            assert prediction['groundTruth'] == entry['label']
            assert prediction['probabilities'] == expected['probabilities']
            assert prediction['modelId'] == model_id
    comparison = client.post('/api/research/evaluations/compare', json={'jobIds': jobs})
    assert comparison.status_code == 200, comparison.text
    assert len(comparison.json()['evaluations']) == 2
    groups = client.get('/api/research/evaluations?datasetId=' + dataset['id']).json()
    assert groups['total'] == 2 and len(groups['groups']) == 1
    assert {item['modelId'] for item in groups['items']} == {'1d-cnn', 'rf-experimental'}


@pytest.mark.parametrize('model_id', ['1d-cnn', 'rf-experimental'])
def test_csv_retains_cnn_softmax_but_labels_forest_proportions_as_model_scores(client, model_id):
    catalog = {item['id']: item for item in client.get('/api/research/models').json()}
    if not catalog[model_id]['available']:
        pytest.skip('Selected genuine checkpoint requires a compatible optional runtime.')
    dataset = labeled_dataset(client)
    job, report = evaluate(client, dataset, model_id)
    response = client.get('/api/workspace/jobs/' + job['id'] + '/export?format=csv')
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == len(report['predictions']) == 3
    for row, prediction in zip(rows, report['predictions']):
        assert float(row['rawModelScore']) == prediction['confidence']
        assert json.loads(row['probabilities']) == prediction['probabilities']
        assert float(row['rawModelScore']) == max(prediction['probabilities'].values())
        assert row['modelHash'] == report['modelHash']
        if model_id == '1d-cnn':
            assert float(row['rawSoftmax']) == prediction['confidence']
        else:
            assert row['rawSoftmax'] == ''


def test_unknown_unavailable_version_and_class_mapping_are_not_substituted(client, monkeypatch):
    dataset = labeled_dataset(client)
    request = {'datasetId': dataset['id'], 'modelId': '1d-cnn'}
    assert client.post('/api/research/evaluations', json={**request, 'modelVersion': 'invented-version'}).status_code == 409
    assert client.post('/api/research/evaluations', json={**request, 'modelId': 'unregistered'}).status_code == 409
    assert client.post('/api/research/evaluations', json={**request, 'modelId': 'resnet'}).status_code == 503
    production = REGISTRY['1d-cnn']
    monkeypatch.setitem(REGISTRY, '1d-cnn', replace(production, class_labels=('BPSK',)))
    assert client.post('/api/research/evaluations', json=request).status_code == 409
    monkeypatch.setitem(REGISTRY, '1d-cnn', production)
    def absent():
        raise RuntimeError('Missing /private/checkpoint/server/path')
    monkeypatch.setitem(REGISTRY, '1d-cnn', replace(production, loader=absent))
    response = client.post('/api/research/evaluations', json=request)
    assert response.status_code == 503 and '/private/' not in response.text
    assert client.get('/api/workspace/jobs').json()['total'] == 0


def test_dataset_sample_mutation_fails_inference_without_exporting_partial_metrics(client):
    dataset = labeled_dataset(client)
    entry = dataset['entries'][0]
    path = store.file_path(entry['captureId'], 'iq')
    with path.open('r+b') as stream:
        stream.write(np.array([.5], dtype='<f4').tobytes())
    response = client.post('/api/research/evaluations', json={'datasetId': dataset['id']})
    assert response.status_code == 202
    job = done(client, response.json())
    assert job['state'] == 'failed' and 'frame hash' in job['error']
    assert not store.file_path(job['id'], 'json').exists()


def test_arena_refuses_different_registered_dataset_or_holdout_protocol(client):
    dataset = labeled_dataset(client)
    first, _ = evaluate(client, dataset)
    cloned = client.post('/api/workspace/datasets', json={
        'name': 'Different registration', 'source': dataset['source'], 'license': dataset['license'],
        'entries': [{key: entry[key] for key in ['captureId', 'start', 'label', 'split', 'snrDb']} for entry in dataset['entries']],
    }).json()
    second, _ = evaluate(client, cloned)
    assert dataset['hash'] == cloned['hash'] and dataset['id'] != cloned['id']
    response = client.post('/api/research/evaluations/compare', json={'jobIds': [first['id'], second['id']]})
    assert response.status_code == 422 and 'identical dataset' in response.text
    assert client.post('/api/research/evaluations/compare', json={'jobIds': [first['id'], first['id']]}).status_code == 422


def test_training_export_preserves_splits_hashes_missing_snr_and_is_deterministic(client, tmp_path):
    dataset = labeled_dataset(client, full_splits=True)
    url = '/api/research/datasets/' + dataset['id']
    assert client.get(url + '/training-manifest').status_code == 422
    manifest_response = client.get(url + '/training-manifest?allowSmall=true&model=random-forest&seed=13')
    assert manifest_response.status_code == 200, manifest_response.text
    manifest = manifest_response.json()
    npz = client.get(url + '/training.npz?allowSmall=true&model=random-forest&seed=13')
    again = client.get(url + '/training.npz?allowSmall=true&model=random-forest&seed=13')
    assert npz.status_code == 200 and npz.content == again.content
    assert hashlib.sha256(npz.content).hexdigest() == manifest['artifact']['sha256'] == npz.headers['x-artifact-sha256']
    assert manifest['datasetHash'] == dataset['hash'] and manifest['sampleCount'] == 27
    assert manifest['snr']['knownSamples'] == 1 and manifest['snr']['unavailableSamples'] == 26
    assert manifest['experimentConfig']['seed'] == 13
    assert '--allow-small-dataset' in manifest['offlineCommand']
    assert client.get(url + '/experiment-config?allowSmall=true&seed=13').json() == manifest['experimentConfig']
    file = tmp_path / manifest['artifact']['filename']
    file.write_bytes(npz.content)
    loaded = load_dataset(file, allow_small=True)
    split = split_dataset(loaded, 13)
    assert {name: len(indices) for name, indices in split.items()} == {'train': 9, 'validation': 9, 'test': 9}
    assert set(loaded['groups'][split['train']]).isdisjoint(loaded['groups'][split['test']])
    assert np.isnan(loaded['snrs'][~loaded['snrKnown']]).all()
    assert loaded['snrs'][loaded['snrKnown']].tolist() == [0.0]


def test_export_executes_the_existing_offline_training_and_replay_without_promoting(client, tmp_path):
    dataset = labeled_dataset(client, full_splits=True)
    response = client.get('/api/research/datasets/' + dataset['id'] + '/training.npz?allowSmall=true')
    assert response.status_code == 200
    file = tmp_path / 'registered.npz'
    file.write_bytes(response.content)
    production_hash = store.artifact_hash()
    report = run_experiment(file, tmp_path / 'smoke-run',
                            {'model': 'random-forest', 'seed': 19, 'trees': 4, 'temperatureScaling': False}, allow_small=True)
    replay = evaluate_run(tmp_path / 'smoke-run', file)
    assert replay['maxReplayProbabilityError'] < 1e-10
    assert report['productionPromoted'] is False and store.artifact_hash() == production_hash
    assert report['normalization']['fitSplit'] == 'train'
    assert report['registeredDatasetProvenance']['datasetId'] == dataset['id']
    assert report['test']['snrAvailability'] == {
        'knownSamples': 1, 'unavailableSamples': 8,
        'source': 'Supplied dataset metadata; measurement provenance is the importer’s responsibility.',
    }
    assert report['test']['snrAccuracy'][0]['snrDb'] == 0.0
    assert report['splits']['train']['indices'] == list(range(9))
    assert report['splits']['test']['indices'] == list(range(18, 27))


def test_all_test_registry_is_never_silently_repartitioned_for_training(client):
    dataset = labeled_dataset(client)
    response = client.get('/api/research/datasets/' + dataset['id'] + '/training.npz?allowSmall=true')
    assert response.status_code == 422 and 'nine independent' in response.text


@pytest.mark.parametrize('legacy', [False, True])
def test_retained_final_holdouts_survive_registry_deletion_and_cannot_be_exported_for_fitting(client, legacy):
    records = [import_session(client, group) for group in range(9)]
    holdout_entries = [{'captureId': records[0][0]['id'], 'start': index * 128, 'label': label, 'split': 'test'}
                       for index, label in enumerate(['BPSK', 'QPSK', '16-QAM'])]
    response = client.post('/api/workspace/datasets', json={
        'name': 'Frozen synthetic final test', 'source': 'Generated test fixture, not RF', 'license': 'Fixture', 'entries': holdout_entries})
    assert response.status_code == 200
    heldout = response.json()
    job, _ = evaluate(client, heldout)
    assert client.delete('/api/workspace/datasets/' + heldout['id']).status_code == 200
    if legacy:
        path = store.file_path(job['id'], 'json')
        report = json.loads(path.read_text())
        report.pop('dataset')
        path.write_text(json.dumps(report, allow_nan=False))
        store.put('job', {key: value for key, value in store.get('job', job['id']).items() if key != 'evaluationSummary'})
    def register_training(selected, name):
        entries = [{'captureId': record[0]['id'], 'start': index * 128, 'label': label,
                    'split': ['train', 'validation', 'test'][group // 3]}
                   for group, record in enumerate(selected)
                   for index, label in enumerate(['BPSK', 'QPSK', '16-QAM'])]
        created = client.post('/api/workspace/datasets', json={
            'name': name, 'source': 'Separate generated test groups, not actual RF', 'license': 'Fixture', 'entries': entries})
        assert created.status_code == 200, created.text
        return created.json()
    contaminated = register_training(records, 'Reassigned known holdouts')
    exported = client.get('/api/research/datasets/' + contaminated['id'] + '/training.npz?allowSmall=true')
    assert exported.status_code == 422 and 'Final held-out evaluation data' in exported.text
    assert store.file_path(job['id'], 'json').exists()  # Protection never deletes historical evidence.
    # Unrelated valid exports remain available even if another registration has
    # intentionally reassigned a previously observed final test acquisition.
    unrelated = register_training([import_session(client, group + 20) for group in range(9)], 'Independent generated groups')
    valid = client.get('/api/research/datasets/' + unrelated['id'] + '/training-manifest?allowSmall=true')
    assert valid.status_code == 200, valid.text


def test_training_exports_enforce_bounded_concurrency_and_release_slots(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from backend.services import training_exports
    dataset = labeled_dataset(client, full_splits=True)
    url = '/api/research/datasets/' + dataset['id'] + '/training-manifest?allowSmall=true'
    entered, release = threading.Event(), threading.Event()
    gate = threading.Lock()
    count = 0
    real = training_exports._export_training_dataset
    def slow(*args, **kwargs):
        nonlocal count
        with gate:
            count += 1
            if count == 2:entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(training_exports, '_export_training_dataset', slow)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = [workers.submit(client.get, url) for _ in range(2)]
        try:
            assert entered.wait(5)
            rejected = client.get(url)
            assert rejected.status_code == 429 and rejected.headers['retry-after'] == '1'
        finally:release.set()
        assert all(future.result().status_code == 200 for future in pending)
    assert client.get(url).status_code == 200


def test_explicit_snr_mask_is_required_and_known_zero_is_retained(tmp_path):
    p = np.array([[.8, .1, .1], [.2, .7, .1]])
    with pytest.raises(ValueError, match='explicit snrKnown'):
        evaluate_probabilities(p, [0, 1], [np.nan, 0.0], ['BPSK', 'QPSK', '16-QAM'])
    metrics = evaluate_probabilities(p, [0, 1], [np.nan, 0.0], ['BPSK', 'QPSK', '16-QAM'], [False, True])
    assert metrics['snrAccuracy'] == [{'snrDb': 0.0, 'samples': 1, 'accuracy': 1.0,
                                     'perClass': {'BPSK': None, 'QPSK': 1.0, '16-QAM': None}}]
    with pytest.raises(ValueError, match='snrKnown'):
        evaluate_probabilities(p, [0, 1], [np.nan, np.nan], ['BPSK', 'QPSK', '16-QAM'], [False, True])


def test_export_rejects_tampered_registered_arrays_and_demo_frames(client, tmp_path):
    dataset = labeled_dataset(client, full_splits=True)
    response = client.get('/api/research/datasets/' + dataset['id'] + '/training.npz?allowSmall=true')
    with np.load(io.BytesIO(response.content), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['signals'][0, 0, 0] += .05
    file = tmp_path / 'tampered.npz'
    file.write_bytes(deterministic_npz(arrays))
    with pytest.raises(ValueError, match='provenance hashes'):
        load_dataset(file, allow_small=True)


def test_research_exports_use_existing_workspace_access_controls(client, monkeypatch):
    dataset = labeled_dataset(client)
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN', 'local-test-operator-key')
    for path in ['/api/research/models', '/api/research/evaluations',
                 '/api/research/datasets/' + dataset['id'] + '/training-manifest']:
        assert client.get(path).status_code == 401
    response = client.get('/api/research/models', headers={'X-Satmod-Token': 'local-test-operator-key'})
    assert response.status_code == 200


def test_model_registry_rejects_ambiguous_class_order_and_lifecycle():
    with pytest.raises(ValueError, match='unique'):
        register_model(ModelSpec('ambiguous-test-only', 'Ambiguous', class_labels=('BPSK', 'BPSK')))
    with pytest.raises(ValueError, match='lifecycle'):
        register_model(ModelSpec('invalid-test-only', 'Invalid', lifecycle='Automatic production promotion'))


def test_job_window_reader_is_bounded_copies_samples_and_closes_evicted_maps(client):
    from backend.services.capture_windows import CaptureWindows
    records = [import_session(client, index) for index in range(3)]
    reader = CaptureWindows(capacity=2)
    try:
        first, _ = reader.window(records[0][0]['id'], 0)
        first_map = reader.mappings[records[0][0]['id']][0]
        first[0, 0] = .75
        again, _ = reader.window(records[0][0]['id'], 0)
        np.testing.assert_array_equal(again, records[0][1][:128])
        reader.window(records[1][0]['id'], 128)
        reader.window(records[2][0]['id'], 256)
        assert len(reader.mappings) == 2 and first_map._mmap.closed
        with pytest.raises(ValueError, match='outside'):
            reader.window(records[2][0]['id'], 300)
        retained = [value[0] for value in reader.mappings.values()]
    finally:
        reader.close()
    assert not reader.mappings and all(value._mmap.closed for value in retained)
    for metadata, _ in records:
        assert client.delete('/api/workspace/captures/' + metadata['id']).status_code == 200


def test_terminal_job_state_is_published_only_after_mappings_close(client, monkeypatch):
    from backend.services import capture_windows
    metadata, _ = import_session(client, 12)
    instances = []
    real_reader = capture_windows.CaptureWindows
    class TrackedReader(real_reader):
        def __init__(self):
            super().__init__()
            instances.append(self)
    monkeypatch.setattr(capture_windows, 'CaptureWindows', TrackedReader)
    real_update = operations.update
    def observed_update(record_id, **fields):
        if fields.get('state') in ['completed', 'failed', 'cancelled'] and instances:
            assert not instances[-1].mappings
        return real_update(record_id, **fields)
    monkeypatch.setattr(operations, 'update', observed_update)
    job = client.post('/api/workspace/jobs/batch', json={'captureId': metadata['id'].upper(), 'windows': 2}).json()
    assert done(client, job)['state'] == 'completed'
    report = client.get('/api/workspace/jobs/' + job['id'] + '/export').json()
    assert all(prediction['captureId'] == metadata['id'] for prediction in report['predictions'])
    def missing(*args, **kwargs):
        raise RuntimeError('Injected inference failure')
    monkeypatch.setattr(classifier, 'predict_window', missing)
    failed = client.post('/api/workspace/jobs/batch', json={'captureId': metadata['id'], 'windows': 1}).json()
    assert done(client, failed)['state'] == 'failed'
    assert client.delete('/api/workspace/captures/' + metadata['id']).status_code == 200
