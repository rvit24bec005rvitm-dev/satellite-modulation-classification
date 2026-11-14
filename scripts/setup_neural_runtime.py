"""Create the optional Python 3.12 neural API environment without changing defaults.

Use an existing Python 3.12 with --python, or explicitly request the checksum-
verified standalone Intel macOS interpreter with --download-python. All created
interpreters, installers and environment files remain in ignored .venv-neural-*
directories inside this repository. No models are trained or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
INTERPRETER = ROOT / '.venv-neural-python'
RUNTIME = ROOT / '.venv-neural-runtime'
TOOLS = ROOT / '.venv-neural-tools'
PYTHON_VERSION = '3.12.16'
PYTHON_RELEASE = '20261110'
PYTHON_URL = ('https://github.com/astral-sh/python-build-standalone/releases/download/'
              '20261110/cpython-3.12.16%2B20261110-x86_64-apple-darwin-install_only_stripped.tar.gz')
PYTHON_SHA256 = 'b78a4c479f1964ec484880bee567568266f68da102269fb93a05fe8716b45f91'
UV_VERSION = '0.13.7'


def run(command, *, capture=False, environment=None):
    return subprocess.run([str(value) for value in command], cwd=ROOT,
                          env=environment, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def check_python(path):
    output = run([path, '-I', '-c', 'import json,sys;print(json.dumps(list(sys.version_info[:3])))'], capture=True)
    version = json.loads(output)
    if version[:2] != [3, 12]:
        raise ValueError('The optional recorded neural runtime requires Python 3.12.')
    return version


def standalone_python():
    if platform.system() != 'Darwin' or platform.machine() != 'x86_64':
        raise ValueError('--download-python currently supports Intel macOS only; supply an existing Python 3.12 with --python on this platform.')
    binary = INTERPRETER / 'python' / 'bin' / 'python3.12'
    if binary.is_file():
        check_python(binary)
        return binary
    INTERPRETER.mkdir(exist_ok=True)
    archive = INTERPRETER / 'cpython.tar.gz'
    digest = hashlib.sha256()
    context = ssl.create_default_context()
    # Python.org installations may use certifi rather than a configured system
    # certificate bundle. TLS verification remains enabled in either case.
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    print(f'Downloading standalone CPython {PYTHON_VERSION}, release {PYTHON_RELEASE}; validating SHA-256.', flush=True)
    with urllib.request.urlopen(PYTHON_URL, timeout=60, context=context) as response, archive.open('wb') as output:
        while chunk := response.read(1024**2):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != PYTHON_SHA256:
        archive.unlink(missing_ok=True)
        raise ValueError('Standalone Python checksum differs from the pinned official release metadata.')
    # The stdlib data filter rejects paths escaping the dedicated extraction dir.
    with tarfile.open(archive, 'r:gz') as source:
        source.extractall(INTERPRETER, filter='data')
    archive.unlink()
    check_python(binary)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--python', type=Path, help='Existing Python 3.12 executable; it is read, never modified.')
    source.add_argument('--download-python', action='store_true', help='Download the pinned standalone Intel macOS Python into the ignored repository directory.')
    parser.add_argument('--offline', action='store_true', help='Resolve packages from the existing uv cache only; requires the installer to be present already.')
    parser.add_argument('--profile', choices=['auto', 'cpu', 'intel-compat'], default='auto',
                        help='Current Linux CPU profile or local Intel macOS reproduction; auto selects for this host.')
    args = parser.parse_args()
    for managed in [INTERPRETER, RUNTIME, TOOLS]:
        if managed.is_symlink() or (managed.exists() and not managed.is_dir()):
            raise ValueError(f'The isolated setup path must be its own repository directory: {managed.name}')
    selected = args.profile
    if selected == 'auto':
        selected = 'intel-compat' if platform.system() == 'Darwin' and platform.machine() == 'x86_64' else 'cpu'
    if selected == 'intel-compat' and (platform.system() != 'Darwin' or platform.machine() != 'x86_64'):
        raise ValueError('The Intel compatibility profile is only for local Intel macOS reproduction.')
    if selected == 'cpu' and platform.system() == 'Darwin':
        raise ValueError('The current CPU wheel profile targets Linux x86-64; Intel macOS needs the local intel-compat profile.')
    requirements = ROOT / ('requirements-neural-macos-intel.txt' if selected == 'intel-compat' else 'requirements-neural.txt')
    if args.download_python:
        python = standalone_python()
    elif args.python:
        python = args.python.expanduser().resolve()
        check_python(python)
    else:
        found = shutil.which('python3.12')
        if not found:
            raise ValueError('Python 3.12 is not on PATH. Supply --python or explicitly select --download-python on Intel macOS.')
        python = Path(found)
        check_python(python)
    environment = {**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                   'PIP_CACHE_DIR': str(TOOLS / 'pip-cache'), 'UV_LINK_MODE': 'copy'}
    runtime_python = RUNTIME / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if runtime_python.exists():
        if not (RUNTIME / 'pyvenv.cfg').is_file():
            raise ValueError('Existing runtime path is not a virtual environment; it will not be overwritten.')
        if 'include-system-site-packages = true' in (RUNTIME / 'pyvenv.cfg').read_text().lower():
            raise ValueError('The neural runtime must be isolated from global/system site-packages.')
        check_python(runtime_python)
    else:
        if RUNTIME.exists():
            raise ValueError('An incomplete runtime directory already exists; inspect it explicitly before retrying.')
        run([python, '-m', 'venv', RUNTIME], environment=environment)
    uv = TOOLS / ('bin/uv.exe' if os.name == 'nt' else 'bin/uv')
    if not uv.exists():
        if args.offline:
            raise ValueError('The pinned installer is not installed; run once without --offline.')
        TOOLS.mkdir(exist_ok=True)
        run([runtime_python, '-m', 'pip', 'install', '--no-deps', '--target', TOOLS,
             '--index-url', 'https://pypi.org/simple', f'uv=={UV_VERSION}'], environment=environment)
    if not run([uv, '--version'], capture=True).strip().startswith(f'uv {UV_VERSION}'):
        raise ValueError('The isolated installer does not match its pinned version.')
    # uv copies wheels into the environment; shared cached wheels are never
    # symlinked into site-packages or changed by this environment.
    run([uv, 'pip', 'install', '--python', runtime_python, '--index-url', 'https://pypi.org/simple',
         *(['--offline'] if args.offline else []), '-r', requirements], environment=environment)
    run([uv, 'pip', 'check', '--python', runtime_python], environment=environment)
    frozen = run([uv, 'pip', 'freeze', '--python', runtime_python], capture=True, environment=environment)
    (RUNTIME / 'installed-requirements.txt').write_text(frozen)
    metadata = {'python': check_python(runtime_python), 'profile': selected, 'requirements': requirements.name,
                'requirementsSha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
                'standalonePython': {'version': PYTHON_VERSION, 'release': PYTHON_RELEASE,
                                     'url': PYTHON_URL, 'sha256': PYTHON_SHA256} if args.download_python else None,
                'installer': f'uv {UV_VERSION}', 'defaultEnvironmentModified': False}
    (RUNTIME / 'runtime-manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if selected == 'intel-compat':
        print('Local Intel compatibility profile: protobuf has CVE-2026-0994; keep this reproduction environment private.', flush=True)
    print('Optional runtime ready. Start the whole API with:', flush=True)
    print(f'  "{runtime_python}" scripts/serve.py', flush=True)
    print('The default .venv, global Python and saved model artifacts were not changed.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f'Neural runtime setup failed: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
