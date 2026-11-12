"""Single-user SQLite metadata and bounded managed files. Paths never come from clients."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID, uuid4
import json
import os
import sqlite3
import threading
import hashlib
from functools import lru_cache

LOCK = threading.RLock()
KINDS = {'capture','analysis','dataset','job','comparison'}
CONNECTIONS={}


@lru_cache(maxsize=32)
def _resolved_root(configured: str, relative_to: str) -> Path:
    path=(Path(relative_to)/configured if relative_to else Path(configured)).resolve()
    path.mkdir(parents=True,exist_ok=True)
    return path


def root() -> Path:
    from backend.core.config import ROOT
    configured=os.getenv('SATMOD_DATA_DIR',str(ROOT/'.satmod'))
    # Keep environment changes (including relative paths after chdir) explicit,
    # while avoiding repeated resolve/mkdir syscalls for every mapped window.
    return _resolved_root(configured,'' if Path(configured).is_absolute() else os.getcwd())


def identifier(value: str) -> str:
    try: return str(UUID(value))
    except (ValueError,TypeError,AttributeError) as exc:raise ValueError('Invalid managed record identifier.') from exc


def file_path(record_id: str, suffix: str) -> Path:
    if suffix not in ['iq','json','upload','pending']:raise ValueError('Unsupported managed file type.')
    return root()/f'{identifier(record_id)}.{suffix}'


@contextmanager
def connection():
    with LOCK:
        key=str(root())
        db=CONNECTIONS.get(key)
        if db is None:
            # Keep WAL open between window reads; reopening/checkpointing it for
            # every prediction caused avoidable filesystem latency in profiling.
            if len(CONNECTIONS)>=8:CONNECTIONS.pop(next(iter(CONNECTIONS))).close()
            db=sqlite3.connect(root()/'workspace.sqlite3',timeout=10,check_same_thread=False)
            db.row_factory=sqlite3.Row
            db.execute('PRAGMA journal_mode=WAL')
            # These are private atomic-report temporary files, never raw
            # recordings. A first connection precedes every report write, so
            # cleanup cannot race a live writer in this single-process design.
            for pending in root().glob('*.pending'):
                try:identifier(pending.stem)
                except ValueError:continue
                pending.unlink(missing_ok=True)
            CONNECTIONS[key]=db
        try:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version>1:raise RuntimeError('Workspace schema is newer than this application.')
            if version==0:
                db.executescript('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,kind TEXT NOT NULL,created_at TEXT NOT NULL,payload TEXT NOT NULL); CREATE INDEX IF NOT EXISTS records_kind_date ON records(kind,created_at); PRAGMA user_version=1;')
            yield db
            db.commit()
        except BaseException:
            db.rollback();raise


def close_connections():
    with LOCK:
        for db in CONNECTIONS.values():db.close()
        CONNECTIONS.clear()
        _resolved_root.cache_clear()


import atexit
atexit.register(close_connections)


def now():return datetime.now(timezone.utc).isoformat()


def quota(additional: int = 0):
    used=sum(p.stat().st_size for p in root().iterdir() if p.is_file())
    if used+additional>int(os.getenv('SATMOD_STORAGE_BYTES',str(1024**3))):
        raise ValueError('Managed storage quota reached. Delete captures or history before importing more.')


def put(kind: str, value: dict) -> dict:
    if kind not in KINDS:raise ValueError('Invalid record kind.')
    value=dict(value)
    value.setdefault('id',str(uuid4()));value.setdefault('createdAt',now())
    value['id']=identifier(value['id'])
    encoded=json.dumps(value,allow_nan=False)
    with connection() as db:
        if len(encoded)>2*1024**2:raise ValueError('Record metadata exceeds 2 MiB.')
        if not db.execute('SELECT 1 FROM records WHERE id=?',(value['id'],)).fetchone():quota(len(encoded.encode())*2+65536)
        if db.execute('SELECT COUNT(*) FROM records').fetchone()[0]>=10000 and not db.execute('SELECT 1 FROM records WHERE id=?',(value['id'],)).fetchone():raise ValueError('Workspace record limit reached. Delete old records.')
        db.execute('INSERT INTO records VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                   (value['id'],kind,value['createdAt'],encoded))
    return value


def get(kind: str, record_id: str) -> dict:
    with connection() as db:
        row=db.execute('SELECT payload FROM records WHERE kind=? AND id=?',(kind,identifier(record_id))).fetchone()
    if row is None:raise KeyError('Managed record not found.')
    return json.loads(row[0])


def listing(kind: str, limit=50, offset=0, search='', field=None, value=None):
    if not 1<=limit<=200 or offset<0:raise ValueError('Invalid pagination.')
    clause='kind=?';params=[kind]
    if search:
        clause+=' AND payload LIKE ?';params.append('%'+search[:200]+'%')
    if field is not None:
        if field not in ['state','captureId','modelId','predictedModulation']:raise ValueError('Unsupported filter.')
        clause+=f" AND json_extract(payload,'$.{field}')=?";params.append(value)
    with connection() as db:
        total=db.execute('SELECT COUNT(*) FROM records WHERE '+clause,params).fetchone()[0]
        rows=db.execute('SELECT payload FROM records WHERE '+clause+' ORDER BY created_at DESC,id LIMIT ? OFFSET ?',params+[limit,offset]).fetchall()
    return {'items':[json.loads(r[0]) for r in rows],'total':total,'limit':limit,'offset':offset}


def all_records(kind: str):
    """Iterate all pages while callers hold LOCK for reference-sensitive changes."""
    offset=0
    while True:
        page=listing(kind,200,offset)
        yield from page['items'];offset+=200
        if offset>=page['total']:break


def delete(kind: str, record_id: str):
    record_id=identifier(record_id)
    with LOCK:
        get(kind,record_id)
        with connection() as db:db.execute('DELETE FROM records WHERE kind=? AND id=?',(kind,record_id))
        for suffix in ['iq','json','upload','pending']:file_path(record_id,suffix).unlink(missing_ok=True)


def save_document(kind: str, summary: dict, report: dict):
    """Atomically replace a bounded managed report, restoring it on DB failure."""
    record_id=identifier(summary['id'])
    payload=json.dumps(report,allow_nan=False).encode()
    if len(payload)>16*1024**2:raise ValueError('Technical report exceeds the 16 MiB document limit.')
    with LOCK:
        # Initialize SQLite/recover stale pending files before creating ours.
        with connection() as db:
            existing=db.execute('SELECT kind FROM records WHERE id=?',(record_id,)).fetchone()
            if existing and existing['kind']!=kind:raise ValueError('Managed identifier belongs to another record kind.')
        target=file_path(record_id,'json');pending=file_path(record_id,'pending')
        previous=target.read_bytes() if target.exists() else None
        quota(len(payload)+(131072 if existing is None else 0))
        try:
            with pending.open('wb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(pending,target)
            try:put(kind,summary)
            except BaseException:
                if previous is None:target.unlink(missing_ok=True)
                else:
                    pending.write_bytes(previous);os.replace(pending,target)
                raise
        finally:pending.unlink(missing_ok=True)
    return report


def artifact_hash():
    from backend.services.classifier import model
    return model().artifact_sha256


def save_analysis(result: dict, capture_id=None, experiment_id=None):
    import numpy as np
    report={k:v for k,v in result.items() if k!='samples'}
    report.update(reportVersion='satmod-analysis-v2',modelVersion='supplied-cnn-v1',modelHash=artifact_hash(),captureId=capture_id,
                  experimentId=experiment_id,quantityTypes={'probabilities':'uncalibrated softmax','metrics':'computed from samples','snr':'unavailable unless supplied independently'},
                  limitations=['Closed-set three-class classifier; confidence is uncalibrated.','Attribution does not prove physical causality.'])
    report['inputHash']=hashlib.sha256(np.asarray([[p['i'],p['q']] for p in result['samples']],dtype='<f4').tobytes()).hexdigest()
    report['inputRetention']='Managed recording reference' if capture_id else 'Raw samples are not retained for legacy requests. Keep the original capture or import it into managed storage.'
    payload=json.dumps(report,allow_nan=False).encode()
    with LOCK:
        quota(len(payload)+131072);file_path(result['id'],'json').write_bytes(payload)
        summary={k:result[k] for k in ['id','createdAt','source','predictedModulation','confidence','entropyBits','processingMs','modelId']}
        summary.update(captureId=capture_id,modelHash=report['modelHash'],modelVersion=report['modelVersion'],experimentId=experiment_id,reportVersion=report['reportVersion'])
        try:put('analysis',summary)
        except BaseException:
            file_path(result['id'],'json').unlink(missing_ok=True);raise
    return report
