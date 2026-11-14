"""Validate a singleton deployment, then replace this process with Uvicorn."""
import json
import os
from pathlib import Path
import sys


def drop_root_for_managed_volume():
    """Initialize a provider's root-owned mount, then run as the image user.

    Docker normally starts this image as satmod already. Providers that override
    the runtime UID to root may use this narrowly scoped initialization. Existing
    recording/database files are never recursively chowned or rewritten here.
    """
    if not hasattr(os, 'geteuid') or os.geteuid() != 0:
        return
    import pwd
    user = pwd.getpwnam('satmod')
    configured = Path(os.environ.get('SATMOD_DATA_DIR', '/data'))
    mount = Path('/data')
    if (not configured.is_absolute() or configured.resolve() != configured
            or not (configured == mount or mount in configured.parents)):
        raise ValueError('Privileged initialization supports only non-symlink paths under /data.')
    mount.mkdir(parents=True, exist_ok=True)
    os.chown(mount, user.pw_uid, user.pw_gid)
    os.setgroups([])
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)


def main():
    try:
        drop_root_for_managed_volume()
    except Exception:
        print(json.dumps({'event': 'deployment_not_ready', 'errors': ['workspace-mount-initialization-failed']}), flush=True)
        return 1
    from backend.services import deployment_readiness
    result = deployment_readiness.report()
    if not result['ready']:
        print(json.dumps({'event': 'deployment_not_ready', 'checks': result['checks'], 'errors': result['errors']}), flush=True)
        return 1
    # Do not carry preflight SQLite/WAL handles into the replacement process.
    deployment_readiness.store.close_connections()
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'backend.main:app',
                            '--host', '0.0.0.0', '--port', os.getenv('PORT', '8000'),
                            '--proxy-headers', '--workers', '1'])


if __name__ == '__main__':
    raise SystemExit(main())
