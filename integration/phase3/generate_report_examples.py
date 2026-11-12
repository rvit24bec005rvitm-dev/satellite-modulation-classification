"""Generate actual report artifacts from explicitly synthetic, seeded samples."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend.services import captures, classifier, managed_comparisons, research_pdf, saved_explanations, technical_reports, workspace_store as store


def main():
    output=Path(__file__).resolve().parent/'examples';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='satmod-research-examples-') as directory:
        previous=os.environ.get('SATMOD_DATA_DIR');os.environ['SATMOD_DATA_DIR']=directory
        try:
            values=(np.random.default_rng(591).normal(size=(512,2))*.006).astype('<f4')
            stream=io.BytesIO();np.save(stream,values,allow_pickle=False);stream.seek(0)
            metadata=captures.import_capture(SimpleNamespace(file=stream,filename='synthetic-report-fixture.npy'),
                captures.ImportOptions(sessionId='synthetic-report-example',sampleRate=8000,
                    source='Generated report fixture, not real satellite RF',license='Synthetic demonstration fixture'))
            analyses=[]
            for offset in [0,128]:
                iq,_=captures.window(metadata['id'],offset,128)
                analysis=classifier.classify(iq,'Generated report fixture, not real RF',8000,
                    capture_metadata=metadata,capture_position=captures.timing(metadata,offset))
                for target in ['QPSK','BPSK']:
                    saved_explanations.create(analysis['id'],'both',target,16)
                analyses.append(analysis)
            payload,_=technical_reports.export(analyses[0]['id'],'json')
            (output/'explanation.json').write_bytes(payload)
            ui=saved_explanations.response(saved_explanations.load_analysis(analyses[0]['id']))
            (output/'explanations-ui.json').write_text(json.dumps(ui,indent=2,allow_nan=False)+'\n')
            pdf,_=technical_reports.export(analyses[0]['id'],'pdf');(output/'explanation.pdf').write_bytes(pdf)
            comparison=managed_comparisons.create({'captureId':metadata['id'],'start':0,'length':128},
                {'captureId':metadata['id'],'start':128,'length':128},analyses[0]['id'],analyses[1]['id'])
            (output/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
            (output/'comparison.pdf').write_bytes(research_pdf.comparison_pdf(comparison))
        finally:
            store.close_connections()
            if previous is None:os.environ.pop('SATMOD_DATA_DIR',None)
            else:os.environ['SATMOD_DATA_DIR']=previous
    print('Generated current explanation and comparison reports from seeded synthetic samples; no real RF evidence or fitting.')


if __name__=='__main__':main()
