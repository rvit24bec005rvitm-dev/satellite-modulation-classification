"""Check the compiled SPA and real API without opening a local TCP socket."""
import os
from pathlib import Path
import json
import re
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='integration/profiles/production-serving.json',help='Write new evidence without replacing historical serving reports.')
arguments=parser.parse_args()
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
    assert 'fonts.googleapis.com' not in index
    font_css=client.get('/fonts/fonts.css')
    assert font_css.status_code==200 and 'text/css' in font_css.headers['content-type']
    fonts=sorted(set(re.findall(r'url\((/fonts/[^)]+)\)',font_css.text)))
    assert fonts and 'https://' not in font_css.text
    for font in fonts:
        response=client.get(font)
        assert response.status_code==200 and response.content[:4]==b'wOF2',font
    assert client.get('/fonts/missing.woff2').status_code==404
    assert client.get('/api/health').json()['connected']
    assert client.get('/api/does-not-exist').status_code==404
    report={'spaRoutesChecked':routes,'indexAssetsChecked':assets,'localFontAssetsChecked':fonts,'health':200,'unknownApi':404,'missingFont':404}
    destination=Path(arguments.output)
    if not destination.is_absolute():destination=ROOT/destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
