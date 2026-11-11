"""Persistent bounded jobs and independently labeled capture evaluation."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from uuid import uuid4
import json
import threading
import numpy as np
from backend.services import workspace_store as store, captures, classifier

RUNNERS={}


def overlaps(a,b):
    if a['sourceHash']==b['sourceHash']:return abs(a['sourceStart']-b['sourceStart'])<128
    return bool(a.get('canonicalSourceHash') and a['canonicalSourceHash']==b.get('canonicalSourceHash') and abs(a['start']-b['start'])<128)


def register_dataset(name, entries, source, license_text):
    if not entries or len(entries)>5000:raise ValueError('Supply 1–5000 independently labeled windows.')
    from utils.preprocessing import CLASS_LABELS
    from training.data import forbidden_demo_frame_hashes
    import hashlib
    groups={};hashes={};normalized=[]
    for entry in entries:
        if entry.label not in CLASS_LABELS:raise ValueError('Ground truth must be supplied from the active model classes.')
        iq,meta=captures.window(entry.captureId,entry.start,128)
        if meta['split']!='unassigned' and entry.split!=meta['split']:raise ValueError('Window split conflicts with recording split.')
        if meta['sessionId'] in groups and groups[meta['sessionId']]!=entry.split:raise ValueError('Source session crosses dataset splits.')
        groups[meta['sessionId']]=entry.split
        digest=hashlib.sha256(iq.tobytes()).hexdigest()
        if bytes.fromhex(digest) in forbidden_demo_frame_hashes():raise ValueError('Demo frames have unknown historical split membership; external evaluation/fitting is prohibited.')
        if digest in hashes and hashes[digest]!=entry.split:raise ValueError('Duplicate frame crosses dataset splits.')
        hashes[digest]=entry.split
        candidate={**entry.model_dump(),'frameHash':digest,'sessionId':meta['sessionId'],'sourceHash':meta['sourceHash'],'sourceStart':meta['sourceOffset']+entry.start,'canonicalSourceHash':meta['hash']}
        for prior in normalized:
            if prior['split']!=entry.split and overlaps(prior,candidate):
                raise ValueError('Overlapping recording windows cross splits.')
        normalized.append(candidate)
    # Also reject contamination against previously registered datasets.
    offset=0
    while True:
        page=store.listing('dataset',200,offset)
        for dataset in page['items']:
            for previous in dataset.get('entries',[]):
                for entry in normalized:
                    if previous['split']!=entry['split'] and (previous['sessionId']==entry['sessionId'] or previous['frameHash']==entry['frameHash'] or
                        overlaps(previous,entry)):
                        raise ValueError('Dataset conflicts with an existing session/frame/overlap split assignment.')
        offset+=200
        if offset>=page['total']:break
    import hashlib
    return store.put('dataset',{'name':name,'source':source,'license':license_text,'acquisitionType':'user-declared external labels',
        'entries':normalized,'hash':hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest(),
        'preprocessing':'Canonical saved scalar z-score; no fitting on imported test labels.',
        'note':'Labels are supplied by the importer, not inferred. Acquisition independence is declared, not automatically proven.'})


def runner():
    key=str(store.root())
    with store.LOCK:
        if key not in RUNNERS:
            # A previous process cannot resume in-flight Python work safely.
            pending=[*store.listing('job',200,field='state',value='queued')['items'],*store.listing('job',200,field='state',value='running')['items']]
            for job in pending:
                if job['state'] in ['queued','running']:
                    store.put('job',{**job,'state':'failed','error':'Interrupted by application restart; resubmit explicitly.'})
            RUNNERS[key]=(ThreadPoolExecutor(max_workers=2,thread_name_prefix='satmod-jobs'),threading.BoundedSemaphore(4))
        return RUNNERS[key]


def submit(kind, params):
    executor,slots=runner()
    with store.LOCK:
        if not slots.acquire(blocking=False):raise ValueError('Job capacity reached (two workers, four queued/running jobs).')
        try:
            if kind=='batch':
                meta=store.get('capture',params['captureId'])
                if params['start']+(params['windows']-1)*params['stride']+128>meta['sampleCount']:raise ValueError('Batch extends beyond recording.')
            elif kind=='evaluation':store.get('dataset',params['datasetId'])
            else:raise ValueError('Unsupported job kind.')
            job=store.put('job',{'state':'queued','kind':kind,'params':params,'progress':0,'cancelRequested':False})
            executor.submit(execute,job['id'],slots)
            return job
        except BaseException:slots.release();raise


def update(record_id, **fields):
    with store.LOCK:return store.put('job',{**store.get('job',record_id),**fields})


def cancel(record_id):
    with store.LOCK:
        job=store.get('job',record_id)
        if job['state'] not in ['queued','running']:return job
        return update(record_id,cancelRequested=True)


def execute(record_id, slots):
    try:
        job=store.get('job',record_id)
        if job['cancelRequested']:update(record_id,state='cancelled');return
        update(record_id,state='running');params=job['params'];predictions=[];provenance={}
        if job['kind']=='batch':
            entries=[{'captureId':params['captureId'],'start':params['start']+i*params['stride']} for i in range(params['windows'])]
        else:
            entries=[entry for entry in store.get('dataset',params['datasetId'])['entries'] if entry['split']=='test']
            if not entries:raise ValueError('External evaluation requires explicitly labeled test windows.')
        for index,entry in enumerate(entries):
            if store.get('job',record_id)['cancelRequested']:
                update(record_id,state='cancelled',progress=index/len(entries));return
            iq,meta=captures.window(entry['captureId'],entry['start'],128)
            if meta['id'] not in provenance:
                provenance[meta['id']]={k:meta.get(k) for k in ['id','source','hash','sourceHash','sessionId','sampleCount','sampleRate','sampleRateKnown','sourceOffset','centerFrequency','license','createdAt']}
            result=classifier.predict_window(iq)
            predictions.append({'index':index,'start':entry['start'],'captureId':entry['captureId'],**captures.timing(meta,entry['start']),**result})
            if index%16==0:update(record_id,progress=(index+1)/len(entries))
        result={'reportVersion':'satmod-batch-v1','jobId':record_id,'createdAt':job['createdAt'],'parameters':params,'captures':provenance,'modelId':'1d-cnn','modelVersion':'supplied-cnn-v1','modelHash':store.artifact_hash(),
                'predictions':predictions,'classCounts':dict(Counter(p['predictedModulation'] for p in predictions)),
                'note':'Adjacent or overlapping windows are dependent. Vote counts do not establish improved accuracy.'}
        if job['kind']=='evaluation':
            from training.metrics import evaluate_probabilities
            from utils.preprocessing import CLASS_LABELS
            y=[CLASS_LABELS.index(e['label']) for e in entries]
            snrs=[e.get('snrDb') for e in entries]
            metrics=evaluate_probabilities([list(p['probabilities'].values()) for p in predictions],y,[s if s is not None else 0 for s in snrs],CLASS_LABELS)
            # Zero is never substituted as a reported measured SNR.
            known=[i for i,s in enumerate(snrs) if s is not None]
            metrics['snrAccuracy']=evaluate_probabilities([list(predictions[i]['probabilities'].values()) for i in known],[y[i] for i in known],[snrs[i] for i in known],CLASS_LABELS)['snrAccuracy'] if known else []
            result.update(metrics=metrics,datasetId=params['datasetId'],labelSource='Importer-supplied independent labels; not model predictions',calibration='Diagnostic only; production temperature unchanged')
        data=json.dumps(result,allow_nan=False).encode()
        with store.LOCK:
            if store.get('job',record_id)['cancelRequested']:update(record_id,state='cancelled');return
            store.quota(len(data));store.file_path(record_id,'json').write_bytes(data)
            update(record_id,state='completed',progress=1,finishedAt=store.now())
    except Exception as exc:
        import logging
        logging.getLogger('satmod').exception('Background operation failed: %s',record_id)
        store.file_path(record_id,'json').unlink(missing_ok=True)
        update(record_id,state='failed',error=str(exc) if isinstance(exc,(ValueError,KeyError)) else 'Processing failed; inspect the server log.',finishedAt=store.now())
    finally:slots.release()


def job_result(record_id,offset=0,limit=200):
    job=store.get('job',record_id)
    if job['state']!='completed':raise ValueError('Job has no completed result.')
    result=json.loads(store.file_path(record_id,'json').read_text())
    predictions=result.pop('predictions');return {**result,'predictions':predictions[offset:offset+limit],'total':len(predictions),'offset':offset}
