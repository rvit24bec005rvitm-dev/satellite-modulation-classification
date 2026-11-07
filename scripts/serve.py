"""Serve the built frontend and API together, honoring configured/free local ports."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SATMOD_SERVE_FRONTEND']='1'
if __name__=='__main__':
    import uvicorn
    from backend.core.config import ROOT as PROJECT_ROOT  # loads optional .env
    from scripts.dev import available_port
    port=available_port(int(os.getenv('SATMOD_API_PORT','8000')),'SATMOD_API_PORT' in os.environ)
    print(f'SatModClass production: http://localhost:{port}',flush=True)
    uvicorn.run('backend.main:app',host='127.0.0.1',port=port)
