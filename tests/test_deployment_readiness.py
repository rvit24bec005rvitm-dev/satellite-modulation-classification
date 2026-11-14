"""Deployment readiness uses real storage/model probes and keeps legacy liveness."""
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.services import deployment_entrypoint, deployment_readiness as readiness
from backend.services import workspace_store as store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR', str(tmp_path / 'workspace'))
    for key in (*readiness.COUNT_SETTINGS, 'SATMOD_ACCESS_TOKEN', 'SATMOD_PUBLIC', 'PORT',
                'GUNICORN_CMD_ARGS', 'UVICORN_CMD_ARGS'):
        monkeypatch.delenv(key, raising=False)
    readiness.clear_cache()
    with TestClient(app) as session:
        yield session
    store.close_connections()
    readiness.clear_cache()


def test_ready_checks_real_model_and_storage_without_changing_saved_records(client):
    record_id = str(uuid4())
    original = {'id': record_id, 'source': 'Synthetic readiness preservation fixture'}
    store.put('capture', original)
    capture = store.file_path(record_id, 'iq')
    capture.write_bytes(bytes(128 * 8))
    previous = store.get('capture', record_id)
    result = client.get('/api/ready')
    assert result.status_code == 200, result.text
    assert result.headers['cache-control'] == 'no-store'
    assert result.json()['ready'] is True
    assert all(result.json()['checks'].values())
    assert store.get('capture', record_id) == previous
    assert capture.read_bytes() == bytes(128 * 8)
    assert store.listing('analysis', 1)['total'] == 0
    assert not list(store.root().glob('.readiness-*'))


def test_public_missing_token_is_unready_while_legacy_health_stays_public(client, monkeypatch):
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    result = client.get('/api/ready')
    assert result.status_code == 503
    assert result.json()['checks']['access'] is False
    assert 'workspace-access-not-configured' in result.json()['errors']
    assert client.get('/api/health').status_code == 200
    assert client.get('/api/system/info').status_code == 503


def test_readiness_probe_needs_no_secret_but_does_not_unlock_api(client, monkeypatch):
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN', 'synthetic-test-only-protected-key')
    response = client.get('/api/ready')
    assert response.status_code == 200
    assert 'synthetic-test-only-protected-key' not in response.text
    assert str(store.root()) not in response.text
    assert client.get('/api/system/info').status_code == 401


@pytest.mark.parametrize('token', ['   ', 'non-ascii-\u00e9-key', 'bad\nheader'])
def test_invalid_header_token_fails_readiness_without_exposing_it(client, monkeypatch, token):
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN', token)
    response = client.get('/api/ready')
    assert response.status_code == 503
    assert response.json()['checks']['access'] is False
    assert 'workspace-access-not-configured' in response.json()['errors']


def test_storage_exception_is_safe_and_does_not_change_liveness(client, monkeypatch):
    def unavailable():
        raise PermissionError('/private/example-credentials must never be exposed')
    monkeypatch.setattr(readiness, 'probe_storage', unavailable)
    response = client.get('/api/ready')
    assert response.status_code == 503
    assert response.json()['checks']['storage'] is False
    assert 'workspace-storage-unavailable' in response.json()['errors']
    assert 'example-credentials' not in response.text
    assert client.get('/api/health').status_code == 200


def test_corrupt_sqlite_is_not_ready_and_probe_files_are_cleaned(client):
    store.root().joinpath('workspace.sqlite3').write_bytes(b'Explicit invalid SQLite test fixture')
    response = client.get('/api/ready')
    assert response.status_code == 503
    assert response.json()['checks']['storage'] is False
    assert not list(store.root().glob('.readiness-*'))


def test_interrupted_owned_readiness_files_are_cleaned_without_deleting_other_data(client):
    directory = store.root()
    marker = b'satmod-ready\n'
    stale = [directory / '.readiness-abcdefgh', directory / '.readiness-abcdefgi.verified']
    stale[0].write_bytes(b'')  # Crash after allocation, before the first write.
    stale[1].write_bytes(marker)
    unrelated = directory / '.readiness-abcdefgj'
    unrelated.write_bytes(b'Unrelated operator file')
    assert client.get('/api/ready').status_code == 200
    assert all(not path.exists() for path in stale)
    assert unrelated.read_bytes() == b'Unrelated operator file'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX write bits differ from Windows directory ACLs.')
@pytest.mark.parametrize('target', ['directory', 'database'])
def test_readonly_workspace_or_database_is_not_ready(client, target):
    client.get('/api/ready')
    location = store.root() if target == 'directory' else store.root() / 'workspace.sqlite3'
    original = location.stat().st_mode
    try:
        location.chmod(0o555 if target == 'directory' else 0o444)
        readiness.clear_cache()
        response = client.get('/api/ready')
        assert response.status_code == 503
        assert response.json()['checks']['storage'] is False
    finally:
        location.chmod(original)


def test_model_forward_failure_is_not_ready_without_exception_details(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('private-model-path or credential must not be returned')
    monkeypatch.setattr(readiness.classifier, 'predict_window', unavailable)
    response = client.get('/api/ready')
    assert response.status_code == 503
    assert response.json()['checks']['model'] is False
    assert 'production-model-unavailable' in response.json()['errors']
    assert 'private-model-path' not in response.text


@pytest.mark.parametrize('key', readiness.COUNT_SETTINGS)
def test_explicit_multiple_workers_or_replicas_are_rejected_before_storage(client, monkeypatch, key):
    monkeypatch.setenv(key, '2')
    def forbidden():
        raise AssertionError('Unsupported process configuration must not touch storage')
    monkeypatch.setattr(readiness, 'probe_storage', forbidden)
    response = client.get('/api/ready')
    assert response.status_code == 503
    assert response.json()['checks']['configuration'] is False
    assert response.json()['errors'] == ['single-process-required']
    assert client.get('/api/health').status_code == 200


@pytest.mark.parametrize('arguments', [
    ['uvicorn', '--workers', '2'], ['gunicorn', '-w', '2'],
    ['uvicorn', '--workers=3'], ['gunicorn', '-w4'], ['uvicorn', '--workers'],
])
def test_worker_launch_arguments_are_checked(arguments):
    assert 'single-process-required' in readiness.configuration_errors({}, arguments)


def test_single_process_arguments_and_bad_worker_configuration():
    assert readiness.configuration_errors({'WEB_CONCURRENCY': '1'}, ['uvicorn', '--workers', '1']) == []
    assert readiness.configuration_errors({'GUNICORN_CMD_ARGS': '-w 2'}, []) == ['single-process-required']
    assert readiness.configuration_errors({'UVICORN_CMD_ARGS': '"'}, []) == ['invalid-worker-arguments']
    assert readiness.configuration_errors({'PORT': '70000'}, []) == ['invalid-port']
    assert readiness.configuration_errors({'SATMOD_PUBLIC': '1'}, []) == ['explicit-public-storage-path-required']
    assert readiness.configuration_errors({'SATMOD_PUBLIC': '1', 'SATMOD_DATA_DIR': 'relative'}, []) == ['explicit-public-storage-path-required']


def test_anonymous_readiness_probes_coalesce_but_config_changes_and_cli_are_fresh(client, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(readiness, 'monotonic', lambda: clock[0])
    calls = []
    original = readiness.probe_storage
    def tracked():
        calls.append(1)
        return original()
    monkeypatch.setattr(readiness, 'probe_storage', tracked)
    for _ in range(5):
        assert client.get('/api/ready').status_code == 200
    assert len(calls) == 1
    clock[0] = 103
    assert client.get('/api/ready').status_code == 200
    assert len(calls) == 2
    assert readiness.report()['ready'] is True
    assert len(calls) == 3
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    assert client.get('/api/ready').status_code == 503


def test_entrypoint_never_starts_an_unready_server(monkeypatch, capsys):
    monkeypatch.setattr(deployment_entrypoint, 'drop_root_for_managed_volume', lambda: None)
    monkeypatch.setattr(readiness, 'report', lambda: {'ready': False, 'checks': {'access': False}, 'errors': ['workspace-access-not-configured']})
    def forbidden(*args):
        raise AssertionError('Uvicorn must not start without readiness')
    monkeypatch.setattr(deployment_entrypoint.os, 'execv', forbidden)
    assert deployment_entrypoint.main() == 1
    assert 'workspace-access-not-configured' in capsys.readouterr().out


def test_entrypoint_executes_one_uvicorn_process(monkeypatch):
    monkeypatch.setattr(deployment_entrypoint, 'drop_root_for_managed_volume', lambda: None)
    monkeypatch.setattr(readiness, 'report', lambda: {'ready': True})
    monkeypatch.setenv('PORT', '18429')
    calls = []
    closed = []
    monkeypatch.setattr(store, 'close_connections', lambda: closed.append(True))
    monkeypatch.setattr(deployment_entrypoint.os, 'execv', lambda *args: calls.append(args))
    deployment_entrypoint.main()
    assert closed == [True]
    arguments = calls[0][1]
    assert arguments[arguments.index('--workers') + 1] == '1'
    assert arguments[arguments.index('--port') + 1] == '18429'
    assert arguments[:3] == [deployment_entrypoint.sys.executable, '-m', 'uvicorn']


@pytest.mark.skipif(os.name == 'nt', reason='Root-owned Linux volume initialization is POSIX-only.')
def test_root_mount_initialization_is_scoped_and_drops_privileges(monkeypatch):
    import pwd
    calls = []
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    monkeypatch.setattr(pwd, 'getpwnam', lambda name: SimpleNamespace(pw_uid=10001, pw_gid=10001, pw_dir='/home/satmod'))
    monkeypatch.setenv('SATMOD_DATA_DIR', '/data')
    monkeypatch.setattr(Path, 'is_symlink', lambda path: False)
    monkeypatch.setattr(Path, 'mkdir', lambda *args, **kwargs: calls.append('mkdir'))
    monkeypatch.setattr(os, 'chown', lambda path, uid, gid: calls.append(('chown', str(path), uid, gid)))
    monkeypatch.setattr(os, 'setgroups', lambda groups: calls.append(('groups', groups)))
    monkeypatch.setattr(os, 'setgid', lambda gid: calls.append(('gid', gid)))
    monkeypatch.setattr(os, 'setuid', lambda uid: calls.append(('uid', uid)))
    deployment_entrypoint.drop_root_for_managed_volume()
    assert calls == ['mkdir', ('chown', '/data', 10001, 10001), ('groups', []), ('gid', 10001), ('uid', 10001)]
    calls.clear()
    monkeypatch.setenv('SATMOD_DATA_DIR', '/data/workspace')
    deployment_entrypoint.drop_root_for_managed_volume()
    assert calls == ['mkdir', ('chown', '/data', 10001, 10001), ('groups', []), ('gid', 10001), ('uid', 10001)]
    monkeypatch.setenv('SATMOD_DATA_DIR', '/')
    with pytest.raises(ValueError):
        deployment_entrypoint.drop_root_for_managed_volume()
