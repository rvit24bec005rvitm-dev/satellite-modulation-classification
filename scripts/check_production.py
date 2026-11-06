"""Check the compiled SPA and real API without opening a local TCP socket."""
import os
from pathlib import Path
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SATMOD_SERVE_FRONTEND']='1'
from fastapi.testclient import TestClient
from backend.main import app
with TestClient(app) as client:
    routes=['/','/classifier','/studio','/evaluation','/arena','/intelligence','/theory','/settings']
    for route in routes:
        response=client.get(route)
        assert response.status_code==200 and '<div id="root">' in response.text,route
    index=client.get('/').text
    assets=re.findall(r'(?:src|href)="(/assets/[^"]+)"',index)
    assert assets
    for asset in assets:assert client.get(asset).status_code==200,asset
    assert client.get('/api/health').json()['connected']
    assert client.get('/api/does-not-exist').status_code==404
    report={'spaRoutesChecked':routes,'indexAssetsChecked':assets,'health':200,'unknownApi':404}
    (ROOT/'integration/production-serving.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
