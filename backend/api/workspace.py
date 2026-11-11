"""Managed external recordings, jobs, provenance, diagnostics and history API."""
import io
import json
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import Response
from pydantic import Field
from typing import Literal
from backend.api.schemas import StrictModel
from backend.services import workspace_store as store, captures, operations, classifier, technical_reports
from backend.services.advanced_dsp import diagnostics, activities

router=APIRouter(prefix='/api/workspace')


class Batch(StrictModel):
    captureId: str
    start: int=Field(default=0,ge=0)
    windows: int=Field(default=10,ge=1,le=10000)
    stride: int=Field(default=128,ge=1,le=65536)


class Entry(StrictModel):
    captureId: str
    start: int=Field(ge=0)
    label: Literal['BPSK','QPSK','16-QAM']
    split: Literal['train','validation','test']='test'
    snrDb: float | None=None


class Dataset(StrictModel):
    name: str=Field(min_length=1,max_length=200)
    source: str=Field(min_length=1,max_length=500)
    license: str=Field(min_length=1,max_length=500)
    entries: list[Entry]=Field(min_length=1,max_length=1000)


@router.get('/captures')
def list_captures(limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),search: str=Query('',max_length=200)):
    return store.listing('capture',limit,offset,search)


@router.post('/captures')
def import_capture(file: UploadFile=File(...),options: str=Form(...),metadata: UploadFile | None=File(default=None)):
    try:
        if file.size is not None and file.size>captures.IMPORT_LIMIT:raise HTTPException(413,'Capture exceeds external-import size limit.')
        config=captures.ImportOptions.model_validate_json(options)
        sigmf=None
        if metadata:
            raw=metadata.file.read(65537)
            if len(raw)>65536:raise ValueError('SigMF metadata exceeds 64 KiB.')
            sigmf=json.loads(raw)
        return captures.import_capture(file,config,sigmf)
    finally:
        file.file.close()
        if metadata:metadata.file.close()


@router.get('/captures/{record_id}')
def capture_detail(record_id: str):return captures.overview(record_id)


@router.delete('/captures/{record_id}')
def delete_capture(record_id: str):
    with store.LOCK:
        if any(store.listing('job',1,field='state',value=state)['total'] for state in ['queued','running']):raise HTTPException(409,'Wait for active jobs to finish or cancel them before deleting captures.')
        for dataset in store.listing('dataset',200)['items']:
            if any(e['captureId']==record_id for e in dataset.get('entries',[])):raise HTTPException(409,'Remove referencing datasets before deleting this capture.')
        store.delete('capture',record_id)
    return {'deleted':True}


@router.get('/captures/{record_id}/window')
def capture_window(record_id: str,start: int=Query(0,ge=0),length: int=Query(128,ge=16,le=65536)):
    return captures.inspect(record_id,start,length)


@router.post('/captures/{record_id}/classify')
def classify_window(record_id: str,start: int=Query(0,ge=0)):
    iq,meta=captures.window(record_id,start,128)
    result=classifier.classify(iq,f"{meta['source']} @ sample {start}",meta['sampleRate'])
    result['captureMetadata']=meta;result['capturePosition']=captures.timing(meta,start)
    store.save_analysis(result,record_id)
    return result


@router.get('/captures/{record_id}/diagnostics')
def capture_diagnostics(record_id: str,start: int=Query(0,ge=0),length: int=Query(4096,ge=16,le=65536),modulation: Literal['BPSK','QPSK'] | None=None,fullScale: float | None=Query(None,gt=0,le=1e6)):
    iq,meta=captures.window(record_id,start,length)
    return diagnostics(iq,meta['sampleRate'],modulation,fullScale)


@router.get('/captures/{record_id}/activity')
def activity(record_id: str,threshold: float=Query(...,ge=0,le=1e12),minimumBins: int=Query(1,ge=1,le=512)):
    overview=captures.overview(record_id)
    segments=activities(overview['points'],threshold,minimumBins)
    for segment in segments:
        if segment['endSample'] is None:segment['endSample']=overview['metadata']['sampleCount']
    return {'segments':segments,'threshold':threshold,'note':'Activity in bounded power bins, not guaranteed bursts or decoded packets. Threshold is not a noise estimate.'}


@router.get('/captures/{record_id}/segment')
def segment(record_id: str,start: int=Query(0,ge=0),length: int=Query(128,ge=16,le=65536)):
    iq,_=captures.window(record_id,start,length);buffer=io.BytesIO();np.save(buffer,iq,allow_pickle=False)
    return Response(buffer.getvalue(),media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="satmod-segment.npy"'})


@router.get('/captures/{record_id}/domain-shift')
def domain_shift(record_id: str,start: int=Query(0,ge=0)):
    from backend.services.domain_shift import compare
    return compare(record_id,start)


@router.get('/jobs/{record_id}/export')
def export_job(record_id: str,format: Literal['json','csv']='json'):
    if store.get('job',record_id)['state']!='completed':raise HTTPException(409,'Job is not complete.')
    data=store.file_path(record_id,'json').read_bytes()
    if format=='json':return Response(data,media_type='application/json')
    import csv
    output=io.StringIO();writer=csv.writer(output)
    writer.writerow(['window','sample','class','rawSoftmax','entropyBits','processingMs','timestamp'])
    for p in json.loads(data)['predictions']:writer.writerow([p['index'],p['start'],p['predictedModulation'],p['confidence'],p['entropyBits'],p['processingMs'],p['timestamp']])
    return Response(output.getvalue(),media_type='text/csv')


@router.post('/jobs/batch')
def batch(body: Batch):
    try:return operations.submit('batch',body.model_dump())
    except ValueError as exc:
        if 'capacity' in str(exc):raise HTTPException(429,str(exc)) from exc
        raise


@router.get('/jobs')
def jobs(limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0)):
    operations.runner();return store.listing('job',limit,offset)


@router.get('/jobs/{record_id}')
def job(record_id: str):operations.runner();return store.get('job',record_id)


@router.post('/jobs/{record_id}/cancel')
def cancel_job(record_id: str):return operations.cancel(record_id)


@router.delete('/jobs/{record_id}')
def delete_job(record_id: str):
    with store.LOCK:
        if store.get('job',record_id)['state'] in ['queued','running']:raise HTTPException(409,'Cancel and wait for a terminal state before deleting a job.')
        store.delete('job',record_id)
    return {'deleted':True}


@router.get('/jobs/{record_id}/result')
def job_result(record_id: str,offset: int=Query(0,ge=0),limit: int=Query(200,ge=1,le=200)):
    return operations.job_result(record_id,offset,limit)


@router.get('/datasets')
def datasets():
    from backend.services.experiments import experiment_reports
    from backend.core.config import ROOT
    import hashlib
    return {'references':[{'id':'supplied-demos','source':'classifier_samples.npz','hash':hashlib.sha256((ROOT/'classifier_samples.npz').read_bytes()).hexdigest(),
                          'acquisitionType':'provided demonstration subset','samples':300,'sampleRate':None,'sessionId':None,
                          'classLabels':classifier.stats()[2],'license':'Verify original source usage rights','importDate':None,'split':'unknown; prohibited for fitting'},
                          *[{'id':r['id'],**r['dataset'],'acquisitionType':'controlled synthetic experiment','split':r['splitStrategy']} for r in experiment_reports()]],
            **store.listing('dataset',200)}


@router.post('/datasets')
def dataset(body: Dataset):
    with store.LOCK:return operations.register_dataset(body.name,body.entries,body.source,body.license)


@router.delete('/datasets/{record_id}')
def delete_dataset(record_id: str):
    if any(store.listing('job',1,field='state',value=state)['total'] for state in ['queued','running']):raise HTTPException(409,'Wait for active jobs before deleting datasets.')
    store.delete('dataset',record_id);return {'deleted':True}


@router.post('/datasets/{record_id}/evaluate')
def evaluate(record_id: str):return operations.submit('evaluation',{'datasetId':record_id})


@router.get('/history')
def history(limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),search: str=Query('',max_length=200),modulation: str | None=None):
    return store.listing('analysis',limit,offset,search,'predictedModulation' if modulation else None,modulation)


@router.get('/history/{record_id}/report')
def report(record_id: str,format: Literal['json','csv','pdf']='json'):
    data,mime=technical_reports.export(record_id,format)
    return Response(data,media_type=mime,headers={'Content-Disposition':f'attachment; filename="satmod-analysis.{format}"'})


@router.delete('/history/{record_id}')
def delete_analysis(record_id: str):store.delete('analysis',record_id);return {'deleted':True}


@router.delete('/history')
def clear_history():
    with store.LOCK:
        while True:
            records=store.listing('analysis',200)['items']
            if not records:break
            for record in records:store.delete('analysis',record['id'])
    return {'cleared':True}


@router.get('/status')
def status():
    operations.runner()
    jobs=store.listing('job',200)['items'];history=store.listing('analysis',200)['items'];timings=[h['processingMs'] for h in history]
    return {'captures':store.listing('capture',1)['total'],'datasets':store.listing('dataset',1)['total'],
            'analyses':store.listing('analysis',1)['total'],'runningJobs':sum(store.listing('job',1,field='state',value=state)['total'] for state in ['queued','running']),
            'inferenceMedianMs':float(np.median(timings)) if timings else None,'schemaVersion':1,'storageScope':'Single shared workspace; no per-user isolation.'}
