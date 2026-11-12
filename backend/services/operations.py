"""Persistent bounded jobs and independently labeled capture evaluation."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from bisect import bisect_left, insort
from uuid import uuid4
import json
import threading
import numpy as np
from backend.services import workspace_store as store, captures, classifier

RUNNERS={}


class JobCancelled(Exception):
    """Cooperative cancellation at a bounded-work boundary."""


def recover_interrupted_jobs():
    """Recover once per process/workspace, before admitting new background work."""
    for job in store.all_records('job'):
        if job['state'] not in ['queued','running']:continue
        if job['kind']=='capture-import':
            staged_id=job['params']['staged']['id']
            store.file_path(staged_id,'upload').unlink(missing_ok=True)
            try:
                # Commit may have succeeded just before a process stopped. Keep
                # that capture and report its actual state, rather than unlink it.
                capture=store.get('capture',staged_id)
            except KeyError:
                store.file_path(staged_id,'iq').unlink(missing_ok=True)
            else:
                store.put('job',{**job,'state':'completed','progress':1,'phase':'complete',
                                 'captureId':capture['id'],'finishedAt':store.now()})
                continue
        elif job['kind']=='dataset-import':
            try:dataset=store.get('dataset',job['params']['datasetId'])
            except KeyError:pass
            else:
                store.put('job',{**job,'state':'completed','progress':1,'phase':'complete',
                                 'datasetId':dataset['id'],'finishedAt':store.now()})
                continue
        else:store.file_path(job['id'],'json').unlink(missing_ok=True)
        store.put('job',{**job,'state':'failed','error':'Interrupted by application restart; resubmit explicitly.','finishedAt':store.now()})
    # Only UUID-named files in the application's private managed directory are
    # eligible. This also cleans a crash between staging and job registration.
    for path in store.root().glob('*.upload'):
        try:store.identifier(path.stem)
        except ValueError:continue
        path.unlink(missing_ok=True)
    for path in store.root().glob('*.iq'):
        try:
            record_id=store.identifier(path.stem)
            store.get('capture',record_id)
        except KeyError:path.unlink(missing_ok=True)
        except ValueError:continue


def overlaps(a,b):
    if a['sourceHash']==b['sourceHash']:return abs(a['sourceStart']-b['sourceStart'])<128
    return bool(a.get('canonicalSourceHash') and a['canonicalSourceHash']==b.get('canonicalSourceHash') and abs(a['start']-b['start'])<128)


class SplitGuard:
    """Index fixed-length windows by source; avoid quadratic manifest checks."""
    def __init__(self):
        self.groups={};self.hashes={};self.positions={}

    def check(self, entry):
        """Validate against indexed assignments without retaining more rows."""
        split=entry['split']
        for key, index, error in [
            ('sessionId',self.groups,'Source session crosses dataset splits.'),
            ('frameHash',self.hashes,'Duplicate frame crosses dataset splits.')]:
            if entry[key] in index and index[entry[key]]!=split:raise ValueError(error)
        for prefix, digest, start in [('source',entry['sourceHash'],entry['sourceStart']),
                                     ('canonical',entry.get('canonicalSourceHash'),entry['start'])]:
            if not digest:continue
            positions=self.positions.get((prefix,digest),{})
            for prior_split, offsets in positions.items():
                if prior_split==split:continue
                at=bisect_left(offsets,start)
                if any(abs(start-offsets[i])<128 for i in [at-1,at] if 0<=i<len(offsets)):
                    raise ValueError('Overlapping recording windows cross splits.')
    def add(self, entry):
        self.check(entry)
        split=entry['split']
        self.groups[entry['sessionId']]=split;self.hashes[entry['frameHash']]=split
        for prefix,digest,start in [('source',entry['sourceHash'],entry['sourceStart']),
                                   ('canonical',entry.get('canonicalSourceHash'),entry['start'])]:
            if digest:insort(self.positions.setdefault((prefix,digest),{}).setdefault(split,[]),start)


def register_dataset(name, entries, source, license_text, *, progress=None, record_id=None):
    if not entries or len(entries)>5000:raise ValueError('Supply 1–5000 independently labeled windows.')
    from utils.preprocessing import CLASS_LABELS
    from training.data import forbidden_demo_frame_hashes
    import hashlib
    guard=SplitGuard();normalized=[]
    for index,raw_entry in enumerate(entries):
        if progress:progress('validating labeled windows',.85*index/len(entries))
        entry=raw_entry.model_dump() if hasattr(raw_entry,'model_dump') else dict(raw_entry)
        if entry['label'] not in CLASS_LABELS:raise ValueError('Ground truth must be supplied from the active model classes.')
        iq,meta=captures.window(entry['captureId'],entry['start'],128)
        if meta['split']!='unassigned' and entry['split']!=meta['split']:raise ValueError('Window split conflicts with recording split.')
        digest=hashlib.sha256(iq.tobytes()).hexdigest()
        if bytes.fromhex(digest) in forbidden_demo_frame_hashes():raise ValueError('Demo frames have unknown historical split membership; external evaluation/fitting is prohibited.')
        candidate={**entry,'captureId':meta['id'],'frameHash':digest,'sessionId':meta['sessionId'],'sourceHash':meta['sourceHash'],'sourceStart':meta['sourceOffset']+entry['start'],'canonicalSourceHash':meta['hash']}
        guard.add(candidate)
        normalized.append(candidate)
    if progress:progress('checking registered split assignments',.9)
    # Cross-dataset checks and final registration are atomic relative to other
    # registrations. Polling/cancellation stays possible during window parsing.
    with store.LOCK:
        for dataset in store.all_records('dataset'):
            for previous in dataset.get('entries',[]):
                guard.add(previous)
        return store.put('dataset',{**({'id':record_id} if record_id else {}),'name':name,'source':source,'license':license_text,'acquisitionType':'user-declared external labels',
            'entries':normalized,'hash':hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest(),
            'preprocessing':'Canonical saved scalar z-score; no fitting on imported test labels.',
            'note':'Labels are supplied by the importer, not inferred. Acquisition independence is declared, not automatically proven.'})


def runner():
    key=str(store.root())
    with store.LOCK:
        if key not in RUNNERS:
            # A previous process cannot resume in-flight Python work safely.
            recover_interrupted_jobs()
            RUNNERS[key]=(ThreadPoolExecutor(max_workers=2,thread_name_prefix='satmod-jobs'),threading.BoundedSemaphore(4))
        return RUNNERS[key]


def submit_import(upload, options, metadata=None):
    executor,slots=runner()
    if not slots.acquire(blocking=False):raise ValueError('Job capacity reached (two workers, four queued/running jobs).')
    staged=None
    try:
        staged=captures.stage_upload(upload)
        params={'staged':staged,'options':options.model_dump(),'sigmf':metadata}
        with store.LOCK:
            job=store.put('job',{'state':'queued','kind':'capture-import',
                                'params':params,'phase':'queued','progress':0,'cancelRequested':False})
            executor.submit(execute,job['id'],slots)
        return job
    except BaseException:
        if staged:store.file_path(staged['id'],'upload').unlink(missing_ok=True)
        slots.release();raise


def submit(kind, params):
    executor,slots=runner()
    with store.LOCK:
        if not slots.acquire(blocking=False):raise ValueError('Job capacity reached (two workers, four queued/running jobs).')
        try:
            if kind=='batch':
                meta=store.get('capture',params['captureId'])
                if params['start']+(params['windows']-1)*params['stride']+128>meta['sampleCount']:raise ValueError('Batch extends beyond recording.')
                params={**params,'captureId':meta['id']}
            elif kind=='evaluation':
                dataset=store.get('dataset',params['datasetId'])
                from backend.services.research_evaluation import resolve_model
                labels=[entry['label'] for entry in dataset['entries'] if entry['split']=='test']
                if not labels:raise ValueError('External evaluation requires explicitly labeled test windows.')
                context=resolve_model(params.get('modelId','1d-cnn'),params.get('modelVersion'),labels,params.get('modelHash'))
                params={**params,'datasetId':dataset['id'],'modelId':context.spec.id,
                        'modelVersion':context.spec.version,'modelHash':context.metadata['modelHash']}
            elif kind=='dataset-import':params={**params,'datasetId':str(uuid4())}
            else:raise ValueError('Unsupported job kind.')
            job=store.put('job',{'state':'queued','kind':kind,'params':params,'progress':0,'cancelRequested':False})
            executor.submit(execute,job['id'],slots)
            return job
        except BaseException:slots.release();raise


def update(record_id, **fields):
    with store.LOCK:return store.put('job',{**store.get('job',record_id),**fields})


def cancel(record_id):
    runner()
    with store.LOCK:
        job=store.get('job',record_id)
        if job['state'] not in ['queued','running']:return job
        return update(record_id,cancelRequested=True)


def execute(record_id, slots):
    job=None;reader=None
    def release_reader():
        nonlocal reader
        if reader is not None:
            reader.close();reader=None
    try:
        job=store.get('job',record_id)
        if job['cancelRequested']:raise JobCancelled()
        update(record_id,state='running');params=job['params'];predictions=[];provenance={};context=None
        if job['kind'] in ['capture-import','dataset-import']:
            def progress(phase, fraction):
                with store.LOCK:
                    if store.get('job',record_id)['cancelRequested']:raise JobCancelled()
                    update(record_id,phase=phase,progress=fraction)
            if job['kind']=='capture-import':
                capture=captures.import_capture(None,captures.ImportOptions.model_validate(params['options']),
                                                params.get('sigmf'),staged=params['staged'],progress=progress)
                output={'captureId':capture['id']}
            else:
                dataset=register_dataset(params['name'],params['entries'],params['source'],params['license'],
                                         progress=progress,record_id=params['datasetId'])
                output={'datasetId':dataset['id']}
            update(record_id,state='completed',phase='complete',progress=1,**output,finishedAt=store.now())
            return
        if job['kind']=='batch':
            entries=[{'captureId':params['captureId'],'start':params['start']+i*params['stride']} for i in range(params['windows'])]
        else:
            dataset=store.get('dataset',params['datasetId'])
            entries=[entry for entry in dataset['entries'] if entry['split']=='test']
            if not entries:raise ValueError('External evaluation requires explicitly labeled test windows.')
            from backend.services.research_evaluation import resolve_model
            context=resolve_model(params.get('modelId','1d-cnn'),params.get('modelVersion'),
                                  [entry['label'] for entry in entries],params.get('modelHash'))
        from backend.services.capture_windows import CaptureWindows
        reader=CaptureWindows()
        for index,entry in enumerate(entries):
            if store.get('job',record_id)['cancelRequested']:
                update(record_id,progress=index/len(entries));raise JobCancelled()
            iq,meta=reader.window(entry['captureId'],entry['start'])
            if context is not None:
                import hashlib
                if entry.get('frameHash')!=hashlib.sha256(iq.astype('<f4').tobytes()).hexdigest():
                    raise ValueError('Capture window no longer matches the registered dataset frame hash.')
            if meta['id'] not in provenance:
                provenance[meta['id']]=meta
            result=context.predict(iq) if context is not None else classifier.predict_window(iq)
            predictions.append({'index':index,'start':entry['start'],'captureId':entry['captureId'],**captures.timing(meta,entry['start']),**result,
                                **({'groundTruth':entry['label'],'snrDb':entry.get('snrDb')} if job['kind']=='evaluation' else {})})
            if index%16==0:update(record_id,progress=(index+1)/len(entries))
        # Publish a terminal state only after handles have closed. A client may
        # immediately delete a completed capture; Windows forbids unlinking an
        # open mapping even though inference itself has already finished.
        release_reader()
        result={'reportVersion':'satmod-batch-v1','jobId':record_id,'createdAt':job['createdAt'],'parameters':params,'captures':provenance,'modelId':'1d-cnn','modelVersion':'supplied-cnn-v1','modelHash':store.artifact_hash(),
                'predictions':predictions,'classCounts':dict(Counter(p['predictedModulation'] for p in predictions)),
                'note':'Adjacent or overlapping windows are dependent. Vote counts do not establish improved accuracy.'}
        if job['kind']=='evaluation':
            from training.metrics import evaluate_probabilities
            from backend.services.research_evaluation import evaluation_protocol, summarize_evaluation
            classes=context.spec.class_labels
            y=[classes.index(e['label']) for e in entries]
            snrs=[e.get('snrDb') for e in entries]
            known=np.array([value is not None for value in snrs],dtype=bool)
            numeric_snr=np.array([value if value is not None else np.nan for value in snrs],dtype=float)
            metrics=evaluate_probabilities([[p['probabilities'][c] for c in classes] for p in predictions],y,numeric_snr,classes,known)
            timings=[p['processingMs'] for p in predictions];preparation=[p['preprocessingMs'] for p in predictions]
            result.update(**context.metadata,reportVersion='satmod-external-evaluation-v2',
                          metrics=metrics,datasetId=params['datasetId'],dataset=dataset,
                          completedAt=store.now(),evaluationProtocol=evaluation_protocol(dataset,entries,classes),
                          latency={'inferenceMedianMs':float(np.median(timings)),'inferenceP95Ms':float(np.percentile(timings,95)),
                                   'preprocessingMedianMs':float(np.median(preparation)),
                                   'scope':'Actual per-window inference; preprocessing separate; excludes queue/upload/model initialization.'},
                          labelSource='Importer-supplied labels; acquisition independence is declared, not automatically proven',calibration='Diagnostic only; production temperature unchanged')
        data=json.dumps(result,allow_nan=False).encode()
        with store.LOCK:
            if store.get('job',record_id)['cancelRequested']:raise JobCancelled()
            store.quota(len(data));store.file_path(record_id,'json').write_bytes(data)
            update(record_id,state='completed',progress=1,finishedAt=store.now(),
                   **({'evaluationSummary':summarize_evaluation(result),'modelId':result['modelId'],'modelHash':result['modelHash'],
                       'datasetId':result['datasetId'],'datasetHash':result['dataset']['hash']} if job['kind']=='evaluation' else {}))
    except JobCancelled:
        release_reader()
        update(record_id,state='cancelled',finishedAt=store.now())
    except Exception as exc:
        release_reader()
        import logging
        logging.getLogger('satmod').exception('Background operation failed: %s',record_id)
        store.file_path(record_id,'json').unlink(missing_ok=True)
        update(record_id,state='failed',error=str(exc) if isinstance(exc,(ValueError,KeyError)) else 'Processing failed; inspect the server log.',finishedAt=store.now())
    finally:
        release_reader()
        if job and job['kind']=='capture-import':store.file_path(job['params']['staged']['id'],'upload').unlink(missing_ok=True)
        slots.release()


def job_result(record_id,offset=0,limit=200):
    runner()
    with store.LOCK:
        job=store.get('job',record_id)
        if job['state']!='completed':raise ValueError('Job has no completed result.')
        if job['kind']=='capture-import':
            return {'reportVersion':'satmod-import-v1','jobId':job['id'],'captureId':job['captureId'],
                    'metadata':store.get('capture',job['captureId'])}
        if job['kind']=='dataset-import':
            return {'reportVersion':'satmod-dataset-import-v1','jobId':job['id'],
                    'dataset':store.get('dataset',job['datasetId'])}
        result=json.loads(store.file_path(record_id,'json').read_text())
    predictions=result.pop('predictions');return {**result,'predictions':predictions[offset:offset+limit],'total':len(predictions),'offset':offset}
