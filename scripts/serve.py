"""Serve the built frontend and FastAPI together: python scripts/serve.py."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SATMOD_SERVE_FRONTEND']='1'
if __name__=='__main__':
    import uvicorn
    uvicorn.run('backend.main:app',host='127.0.0.1',port=8000)
