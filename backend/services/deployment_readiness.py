"""Secret-free deployment probes for the supported single-process workspace."""
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import threading
from time import monotonic

import numpy as np

from backend.core.config import VERSION
from backend.services import classifier, workspace_store as store


COUNT_SETTINGS = (
    'WEB_CONCURRENCY', 'UVICORN_WORKERS', 'SATMOD_WORKERS', 'SATMOD_REPLICAS',
)
_CACHE_LOCK = threading.Lock()
_CACHE = None


def configuration_errors(environ=None, arguments=None):
    """Inspect declared concurrency; independent hosts still require one replica."""
    env = os.environ if environ is None else environ
    errors = []
    for name in COUNT_SETTINGS:
        if name in env and str(env[name]).strip() != '1':
            errors.append('single-process-required')
    groups = [list(sys.argv if arguments is None else arguments)]
    for name in ('GUNICORN_CMD_ARGS', 'UVICORN_CMD_ARGS'):
        try:
            groups.append(shlex.split(env.get(name, '')))
        except ValueError:
            errors.append('invalid-worker-arguments')
    for args in groups:
        for index, value in enumerate(args):
            count = None
            if value in ('--workers', '-w'):
                count = args[index + 1] if index + 1 < len(args) else ''
            elif value.startswith('--workers='):
                count = value.partition('=')[2]
            elif value.startswith('-w') and len(value) > 2 and value[2:].isdigit():
                count = value[2:]
            if count is not None and count != '1':
                errors.append('single-process-required')
    if env.get('SATMOD_PUBLIC') == '1':
        configured = env.get('SATMOD_DATA_DIR', '')
        if not configured or not Path(configured).is_absolute():
            errors.append('explicit-public-storage-path-required')
    try:
        port = int(env.get('PORT', '8000'))
        if not 1 <= port <= 65535:
            raise ValueError()
    except (ValueError, TypeError):
        errors.append('invalid-port')
    return list(dict.fromkeys(errors))


def access_ready():
    token = os.getenv('SATMOD_ACCESS_TOKEN', '')
    if not token:
        return os.getenv('SATMOD_PUBLIC') != '1'
    # HTTP headers and constant-time string comparison require an ASCII token.
    return token.isascii() and all(33 <= ord(char) <= 126 for char in token)


def probe_storage():
    """Exercise file replacement and SQLite write permission without changing data.

    This establishes usable storage, not that a cloud mount is durable. The
    provider must attach the persistent volume declared in deployment config.
    """
    paths = []
    with store.LOCK:
        try:
            directory = store.root()
            if not directory.stat().st_mode & 0o222:
                raise PermissionError('Workspace directory is read-only.')
            # Recover only this probe's private temporary files after a killed
            # process. A single supported process holds LOCK throughout probes.
            marker = b'satmod-ready\n'
            for stale in directory.glob('.readiness-*'):
                if (re.fullmatch(r'\.readiness-[a-z0-9_]{8}(?:\.verified)?', stale.name)
                        and not stale.is_symlink() and stale.is_file()
                        and stale.stat().st_size <= len(marker)
                        and marker.startswith(stale.read_bytes())):
                    stale.unlink(missing_ok=True)
            database = directory / 'workspace.sqlite3'
            if database.exists() and not database.stat().st_mode & 0o222:
                raise PermissionError('Workspace database is read-only.')
            with tempfile.NamedTemporaryFile(prefix='.readiness-', dir=directory, delete=False) as stream:
                original = Path(stream.name)
                paths.append(original)
                stream.write(b'satmod-ready\n')
                stream.flush()
                os.fsync(stream.fileno())
            replaced = original.with_name(original.name + '.verified')
            paths.append(replaced)
            os.replace(original, replaced)
            if replaced.read_bytes() != b'satmod-ready\n':
                raise OSError('Workspace storage probe did not round-trip.')
            with store.connection() as database:
                # SQLite must acquire a write transaction, with no row changed.
                database.execute('UPDATE records SET payload=payload WHERE 0')
        finally:
            for path in paths:
                path.unlink(missing_ok=True)


def report():
    errors = configuration_errors()
    checks = {'configuration': not errors, 'access': access_ready(), 'storage': False, 'model': False}
    if not checks['access']:
        errors.append('workspace-access-not-configured')
    # Do not initialize or touch a workspace when its process/path settings
    # are unsupported. In particular, competing workers must not recover files.
    if checks['configuration']:
        try:
            probe_storage()
            checks['storage'] = True
        except Exception:
            errors.append('workspace-storage-unavailable')
    try:
        # A bounded real forward pass checks runtime output as well as artifacts.
        # No analysis/history record or generated prediction is persisted.
        classifier.predict_window(np.zeros((128, 2), dtype=np.float32))
        checks['model'] = True
    except Exception:
        errors.append('production-model-unavailable')
    return {'ready': all(checks.values()), 'version': VERSION, 'checks': checks, 'errors': errors}


def clear_cache():
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


def cached_report():
    """Coalesce anonymous probes for two seconds; CLI preflight stays fresh.

    Environment and launch arguments key the cache, so changing a token or
    process/storage configuration cannot reuse a previously ready result.
    """
    global _CACHE
    keys = (*COUNT_SETTINGS, 'SATMOD_PUBLIC', 'SATMOD_ACCESS_TOKEN', 'SATMOD_DATA_DIR',
            'PORT', 'GUNICORN_CMD_ARGS', 'UVICORN_CMD_ARGS')
    key = (tuple(os.getenv(name) for name in keys), tuple(sys.argv), os.getcwd())
    with _CACHE_LOCK:
        if _CACHE is not None and _CACHE[0] == key and monotonic() - _CACHE[1] < 2:
            return _CACHE[2]
        result = report()
        _CACHE = (key, monotonic(), result)
        return result
