"""Test-only JSON-lines adapter to the real ASGI app; never used in production."""
import base64
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app
with TestClient(app) as client:
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            call=json.loads(line)
            response=client.request(call['method'],call['path'],headers=call['headers'],content=base64.b64decode(call['body']))
            value={'id':call['id'],'status':response.status_code,'headers':dict(response.headers),'body':base64.b64encode(response.content).decode()}
        except Exception as exc:
            value={'id':call.get('id'),'error':str(exc)}
        print(json.dumps(value,allow_nan=False),flush=True)
