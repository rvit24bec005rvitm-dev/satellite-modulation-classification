"""Anonymous discovery and protected sessions never expose credentials."""
from fastapi.testclient import TestClient
import pytest

from backend.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('SATMOD_DATA_DIR', str(tmp_path / 'workspace'))
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('SATMOD_PUBLIC', raising=False)
    with TestClient(app) as instance:
        yield instance


def test_trusted_local_access_is_discoverable_without_a_login(client):
    response = client.get('/api/access')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json() == {'tokenRequired': False, 'configured': True,
                               'publicMode': False, 'sharedWorkspace': True}
    assert client.get('/api/session').json() == {'authorized': True, 'sharedWorkspace': True}


def test_public_missing_secret_is_fail_closed_and_discovery_is_actionable(client, monkeypatch):
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    result = client.get('/api/access')
    assert result.status_code == 200 and result.json()['configured'] is False
    assert client.get('/api/session').status_code == 503
    assert client.get('/api/models').status_code == 503


def test_discovery_never_returns_token_and_session_requires_exact_header(client, monkeypatch):
    token = 'local-unit-operator-token'
    monkeypatch.setenv('SATMOD_PUBLIC', '1')
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN', token)
    discovery = client.get('/api/access')
    assert discovery.json()['tokenRequired'] and discovery.json()['configured']
    assert token not in discovery.text
    assert client.get('/api/session').status_code == 401
    assert client.get('/api/session', headers={'X-Satmod-Token': 'incorrect'}).status_code == 401
    response = client.get('/api/session', headers={'X-Satmod-Token': token})
    assert response.status_code == 200 and response.json()['authorized'] is True
    assert token not in response.text and response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert client.get('/api/session?token=' + token).status_code == 401
    assert client.get('/api/session', headers=[(b'X-Satmod-Token', b'\xff')]).status_code == 401


@pytest.mark.parametrize('token', ['non-ascii-\u2603', 'contains spaces', 'line\nbreak'])
def test_invalid_server_secret_returns_configuration_error_instead_of_500(client, monkeypatch, token):
    monkeypatch.setenv('SATMOD_ACCESS_TOKEN', token)
    assert client.get('/api/access').json()['configured'] is False
    result = client.get('/api/session')
    assert result.status_code == 503 and token not in result.text
