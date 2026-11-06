"""Launch both services, selecting free local ports when defaults are occupied."""
from pathlib import Path
import json
import errno
import os
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def available_port(preferred: int, explicit: bool) -> int:
    for port in range(preferred, min(preferred + (1 if explicit else 100), 65536)):
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            if socket.has_ipv6:
                try:
                    with socket.socket(socket.AF_INET6) as probe:
                        probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                        probe.bind(('::1', port))
                except OSError as exc:
                    if exc.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT):
                        raise
            return port
        except OSError:
            continue
    raise RuntimeError(f'No available port starting at {preferred}. Stop the conflicting service or configure another port.')


def main() -> int:
    npm = shutil.which('npm')
    if not npm:
        raise SystemExit('Node.js/npm is missing. Install Node.js 22+ and run npm install in frontend/.')
    if not (ROOT / 'frontend/node_modules').is_dir():
        raise SystemExit('Run npm install in frontend/ first.')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    api_port = available_port(int(os.getenv('SATMOD_API_PORT', '8000')), 'SATMOD_API_PORT' in os.environ)
    ui_port = available_port(int(os.getenv('SATMOD_FRONTEND_PORT', '5173')), 'SATMOD_FRONTEND_PORT' in os.environ)
    if api_port != 8000:
        print(f'Port 8000 is occupied; using backend port {api_port}.', flush=True)
    if ui_port != 5173:
        print(f'Port 5173 is occupied; using frontend port {ui_port}.', flush=True)
    api_url, ui_url = f'http://127.0.0.1:{api_port}', f'http://localhost:{ui_port}'
    env = os.environ.copy()
    env['VITE_API_PROXY_TARGET'] = api_url
    env['VITE_DEV_PORT'] = str(ui_port)
    origins = [v.strip() for v in env.get('SATMOD_CORS_ORIGINS', '').split(',') if v.strip()]
    origins.extend([ui_url, f'http://127.0.0.1:{ui_port}'])
    env['SATMOD_CORS_ORIGINS'] = ','.join(dict.fromkeys(origins))
    children = []
    result = 0
    try:
        children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', str(api_port)], cwd=ROOT, env=env))
        children.append(subprocess.Popen([npm, 'run', 'dev'], cwd=ROOT / 'frontend', env=env))
        print('Starting SatModClass services…', flush=True)
        deadline = time.monotonic() + 60
        ready = False
        while all(p.poll() is None for p in children):
            if not ready:
                try:
                    with urlopen(api_url + '/api/health', timeout=2) as health:
                        connected = json.load(health).get('connected') is True
                    with urlopen(f'http://127.0.0.1:{ui_port}', timeout=2) as page:
                        frontend_ready = page.status == 200
                    if connected and frontend_ready:
                        ready = True
                        print(f'READY — SatModClass: {ui_url}\nBackend documentation: {api_url}/docs', flush=True)
                except (OSError, URLError, ValueError):
                    pass
                if not ready and time.monotonic() > deadline:
                    print('Startup timed out. Check the service errors above.', file=sys.stderr, flush=True)
                    result = 1
                    break
            time.sleep(.5)
        if not ready and result == 0:
            result = max(1, *(p.returncode or 0 for p in children))
            print('A service stopped before startup completed. Check the error above.', file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return result


if __name__ == '__main__':
    sys.exit(main())
