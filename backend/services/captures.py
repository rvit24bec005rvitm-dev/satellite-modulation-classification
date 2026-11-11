"""Bounded capture import: explicit raw decoding and a documented SigMF subset."""
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import os
import numpy as np
from pydantic import Field
from typing import Literal
from backend.api.schemas import StrictModel
from backend.services import workspace_store as store
from backend.services.signals import parse_capture, analyze, rows

IMPORT_LIMIT=int(os.getenv('SATMOD_CAPTURE_UPLOAD_BYTES',str(128*1024**2)))
DTYPES={'ci8':'i1','ci16_le':'<i2','ci16_be':'>i2','ci32_le':'<i4','ci32_be':'>i4',
        'cf32_le':'<f4','cf32_be':'>f4','cf64_le':'<f8','cf64_be':'>f8'}


class ImportOptions(StrictModel):
    format: Literal['application','raw','sigmf']='application'
    datatype: str | None=None
    layout: Literal['interleaved','planar']='interleaved'
    iqOrder: Literal['IQ','QI']='IQ'
    sampleRate: float | None=Field(default=None,ge=1,le=1e9)
    centerFrequency: float | None=Field(default=None,ge=0,le=1e12)
    scale: float=Field(default=1,gt=0,le=1e6)
    sampleOffset: int=Field(default=0,ge=0)
    sampleCount: int | None=Field(default=None,ge=128)
    source: str=Field(default='External recording',max_length=200)
    sessionId: str=Field(min_length=1,max_length=200)
    license: str=Field(default='Unspecified; user must verify usage rights',max_length=500)
    label: Literal['BPSK','QPSK','16-QAM'] | None=None
    split: Literal['unassigned','train','validation','test']='unassigned'


def finite_block(block):
    block=np.asarray(block,dtype=np.float64)
    if not np.isfinite(block).all() or np.max(np.abs(block))>1e6:
        raise ValueError('Capture contains nonfinite or out-of-range samples after explicit scaling.')
    return block.astype(np.float32)


def stage_upload(upload, record_id=None):
    """Spool once, before returning an import job; retain no caller file handle."""
    record_id=store.identifier(record_id) if record_id else str(uuid4())
    temporary=store.file_path(record_id,'upload')
    sha=hashlib.sha256();sha512=hashlib.sha512();size=0
    try:
        with temporary.open('xb') as target:
            while True:
                chunk=upload.file.read(1024**2)
                if not chunk:break
                size+=len(chunk)
                if size>IMPORT_LIMIT:raise ValueError('Capture upload exceeds configured external-import size limit.')
                with store.LOCK:
                    store.quota(len(chunk));target.write(chunk)
                sha.update(chunk);sha512.update(chunk)
        if not size:raise ValueError('Capture is empty.')
        return {'id':record_id,'filename':Path(upload.filename or 'capture').name[:200],
                'size':size,'sourceHash':sha.hexdigest(),'sha512':sha512.hexdigest()}
    except BaseException:
        temporary.unlink(missing_ok=True);raise


def import_capture(upload, options: ImportOptions, sigmf: dict | None=None, *, staged=None, progress=None) -> dict:
    """Decode staged bytes in bounded blocks. Progress may raise to cancel safely."""
    staged=staged if staged is not None else stage_upload(upload)
    record_id=staged['id'];temporary=store.file_path(record_id,'upload');destination=store.file_path(record_id,'iq')
    size=staged['size'];mapped=None
    def notify(phase, fraction):
        if progress:progress(phase,fraction)
    try:
        notify('validating metadata',0)
        metadata=options.model_dump();metadata.update(filename=staged['filename'],acquisitionType='external',sourceHash=staged['sourceHash'],importBytes=size)
        rate=options.sampleRate;dtype=options.datatype;layout=options.layout;order=options.iqOrder
        if options.format=='sigmf':
            if not isinstance(sigmf,dict) or not isinstance(sigmf.get('global'),dict):raise ValueError('A SigMF metadata object with global properties is required.')
            if len(json.dumps(sigmf,allow_nan=False))>65536:raise ValueError('SigMF metadata exceeds 64 KiB.')
            g=sigmf['global'];dtype=g.get('core:datatype');rate=g.get('core:sample_rate')
            if not isinstance(sigmf.get('captures',[]),list) or len(sigmf.get('captures',[]))>1024 or any(not isinstance(c,dict) for c in sigmf.get('captures',[])):
                raise ValueError('SigMF captures must be a bounded array of metadata objects.')
            if not isinstance(g.get('core:extensions',[]),list) or any(not isinstance(e,dict) for e in g.get('core:extensions',[])):
                raise ValueError('Invalid SigMF extension declarations.')
            if g.get('core:num_channels',1)!=1 or g.get('core:trailing_bytes',0) or g.get('core:offset',0):raise ValueError('Only conforming single-channel SigMF datasets without headers/trailers/global offsets are supported.')
            if any(e.get('optional',False) is not True for e in g.get('core:extensions',[])):
                raise ValueError('Required SigMF extensions are not supported.')
            if g.get('core:sha512') and (not isinstance(g['core:sha512'],str) or g['core:sha512'].lower()!=staged['sha512']):raise ValueError('SigMF data SHA-512 does not match metadata.')
            if 'core:dataset' in g and g['core:dataset']!=metadata['filename']:raise ValueError('SigMF dataset must match the uploaded filename; external paths are never opened.')
            layout='interleaved';order='IQ';metadata['sigmf']=sigmf
            metadata['description']=str(g.get('core:description',''))[:2000]
            if options.license.startswith('Unspecified') and isinstance(g.get('core:license'),str):metadata['license']=g['core:license'][:500]
        if options.format=='application':
            if Path(metadata['filename']).suffix.lower()=='.npy':
                loaded=np.load(temporary,mmap_mode='r',allow_pickle=False)
                if not isinstance(loaded,np.memmap):
                    if hasattr(loaded,'close'):loaded.close()
                    raise ValueError('Expected an actual NPY array, not an archive under a renamed extension.')
                mapped=loaded
                if mapped.dtype.kind not in 'fciu':raise ValueError('Numeric non-object NPY arrays only.')
                if np.iscomplexobj(mapped):
                    if mapped.ndim!=1:raise ValueError('Complex NPY recordings require a vector.')
                    source_count=len(mapped)
                    def slice_iq(start,end):return np.column_stack((mapped[start:end].real,mapped[start:end].imag))
                else:
                    if mapped.ndim!=2 or 2 not in mapped.shape:raise ValueError('Real NPY recordings require (N,2) or (2,N).')
                    transposed=mapped.shape[0]==2 and mapped.shape[1]!=2
                    source_count=mapped.shape[1] if transposed else mapped.shape[0]
                    def slice_iq(start,end):return mapped[:,start:end].T if transposed else mapped[start:end]
                detected_rate=1
            else:
                from backend.core.config import MAX_UPLOAD_BYTES
                if size>MAX_UPLOAD_BYTES:raise ValueError('This application format exceeds its bounded upload limit; use NPY or explicit binary for long recordings.')
                iq,detected_rate=parse_capture(metadata['filename'],temporary.read_bytes())
                source_count=len(iq)
                def slice_iq(start,end):return iq[start:end]
            rate=rate if rate is not None else detected_rate
            def blocks(start,end):
                for offset in range(start,end,65536):yield finite_block(slice_iq(offset,min(end,offset+65536)).astype(np.float64)*options.scale)
        else:
            if not isinstance(dtype,str) or dtype not in DTYPES:raise ValueError('Explicit supported complex datatype is required: '+', '.join(DTYPES))
            component=np.dtype(DTYPES[dtype])
            if size%(2*component.itemsize):raise ValueError('Binary length is not a whole number of complex samples.')
            source_count=size//(2*component.itemsize)
            def blocks(start,end):
                with temporary.open('rb') as stream:
                    for offset in range(start,end,65536):
                        count=min(65536,end-offset)
                        if layout=='interleaved':
                            stream.seek(offset*2*component.itemsize)
                            block=np.frombuffer(stream.read(count*2*component.itemsize),dtype=component).reshape(-1,2)
                        else:
                            stream.seek(offset*component.itemsize);i=np.frombuffer(stream.read(count*component.itemsize),dtype=component)
                            stream.seek((source_count+offset)*component.itemsize);q=np.frombuffer(stream.read(count*component.itemsize),dtype=component)
                            block=np.column_stack((i,q))
                        if order=='QI':block=block[:,::-1]
                        yield finite_block(block.astype(np.float64)*options.scale)
        if not isinstance(rate,(int,float)) or isinstance(rate,bool) or not np.isfinite(rate) or not 1<=rate<=1e9:
            raise ValueError('An explicit finite sample rate is required for raw/SigMF recordings.')
        start=options.sampleOffset;end=min(source_count,start+options.sampleCount) if options.sampleCount else source_count
        if end-start<128:raise ValueError('Selected capture must contain at least 128 samples.')
        if options.sampleCount and end-start!=options.sampleCount:raise ValueError('Requested sample range extends beyond the source recording.')
        if sigmf and options.format=='sigmf':
            previous=-1
            for capture in sigmf.get('captures',[]):
                offset=capture.get('core:sample_start')
                if type(offset) is not int or not previous<offset<source_count:raise ValueError('SigMF capture starts must be increasing indices within the recording.')
                previous=offset
                if capture.get('core:header_bytes',0):raise ValueError('SigMF per-capture headers are unsupported.')
                if 'core:frequency' in capture and (not isinstance(capture['core:frequency'],(int,float)) or isinstance(capture['core:frequency'],bool) or not np.isfinite(capture['core:frequency']) or capture['core:frequency']<0):raise ValueError('Invalid SigMF center frequency.')
                if 'core:datetime' in capture:
                    if not isinstance(capture['core:datetime'],str) or not capture['core:datetime'].endswith('Z'):raise ValueError('SigMF timestamps must be UTC with Z suffix.')
                    datetime.fromisoformat(capture['core:datetime'].replace('Z','+00:00'))
        with store.LOCK:store.quota((end-start)*8)
        digest=hashlib.sha256();written=0
        with destination.open('xb') as target:
            for block in blocks(start,end):
                notify('decoding samples',.9*written/(end-start))
                data=block.astype('<f4').tobytes();digest.update(data)
                with store.LOCK:
                    store.quota(len(data));target.write(data)
                written+=len(block)
        metadata.update(id=record_id,sampleCount=end-start,sourceSampleCount=source_count,sampleRate=float(rate),
                            sampleRateKnown=options.format!='application' or options.sampleRate is not None or detected_rate!=1,
                            sourceOffset=start,datatype=dtype or 'parsed application format',layout=layout,iqOrder=order,
                            hash=digest.hexdigest(),preprocessing='Raw calibrated sample units preserved; saved model z-score only at inference.')
        notify('checking provenance',.9)
        if options.split!='unassigned':
            data=np.memmap(destination,dtype='<f4',mode='r',shape=(end-start,2))
            from training.data import forbidden_demo_frame_hashes
            forbidden=forbidden_demo_frame_hashes()
            try:
                for i in range(0,len(data)-127,128):
                    if i%65536==0:notify('checking provenance',.9+.09*i/len(data))
                    if hashlib.sha256(data[i:i+128].tobytes()).digest() in forbidden:
                        raise ValueError('Original demo frames have unknown split membership and cannot be assigned a fitting/evaluation split.')
            finally:data._mmap.close()
        # Provenance/split checks and the record commit are serialized. No
        # cancellation point follows the commit: the completed capture survives.
        with store.LOCK:
            existing=all_captures()
            if options.split!='unassigned':
                for prior in existing:
                    if prior['split'] not in ['unassigned',options.split] and (prior['sessionId']==options.sessionId or prior['sourceHash']==staged['sourceHash'] or prior['hash']==metadata['hash']):
                        raise ValueError('Dataset split contamination: related session/source or duplicate capture crosses splits.')
            return store.put('capture',metadata)
    except BaseException:
        destination.unlink(missing_ok=True);raise
    finally:
        if mapped is not None:mapped._mmap.close()
        temporary.unlink(missing_ok=True)


def all_captures():
    offset=0
    while True:
        page=store.listing('capture',200,offset)
        yield from page['items'];offset+=200
        if offset>=page['total']:break


def window(record_id, start=0, length=128):
    with store.LOCK:
        meta=store.get('capture',record_id)
        if start<0 or not 16<=length<=65536 or start+length>meta['sampleCount']:raise ValueError('Window lies outside the capture or exceeds 65,536 samples.')
        mm=np.memmap(store.file_path(record_id,'iq'),dtype='<f4',mode='r',offset=start*8,shape=(length,2))
        try:iq=np.array(mm,copy=True)
        finally:mm._mmap.close()
    return iq,meta


def timing(meta, start):
    absolute=meta['sourceOffset']+start
    applicable=[c for c in meta.get('sigmf',{}).get('captures',[]) if c['core:sample_start']<=absolute]
    capture=applicable[-1] if applicable else {}
    timestamp=None
    if capture.get('core:datetime'):
        timestamp=(datetime.fromisoformat(capture['core:datetime'].replace('Z','+00:00'))+timedelta(seconds=(absolute-capture['core:sample_start'])/meta['sampleRate'])).isoformat()
    return {'sourceSample':absolute,'seconds':start/meta['sampleRate'] if meta.get('sampleRateKnown') else None,'timestamp':timestamp,
            'centerFrequency':capture.get('core:frequency',meta.get('centerFrequency'))}


def inspect(record_id,start=0,length=128):
    iq,meta=window(record_id,start,length)
    return {'samples':rows(iq),'sampleRate':meta['sampleRate'],'source':f"{meta['source']} @ sample {start}",
            'analysis':analyze(iq,meta['sampleRate']),'metadata':meta,'position':timing(meta,start)}


def overview(record_id):
    # Hold the file reference lock to prevent concurrent deletion, including on
    # Windows. Temporary allocations stay bounded even if the import cap grows.
    with store.LOCK:
        meta=store.get('capture',record_id)
        mm=np.memmap(store.file_path(record_id,'iq'),dtype='<f4',mode='r',shape=(meta['sampleCount'],2))
        try:
            width=max(128,int(np.ceil(len(mm)/512)));points=[]
            for start in range(0,len(mm),width):
                end=min(len(mm),start+width);total=0.;peak=0.
                for offset in range(start,end,65536):
                    segment=np.asarray(mm[offset:min(end,offset+65536)],dtype=np.float64)
                    power=np.sum(segment**2,axis=1);total+=float(power.sum());peak=max(peak,float(power.max()))
                points.append({'sample':start,'seconds':start/meta['sampleRate'] if meta.get('sampleRateKnown') else None,'power':total/(end-start),'peakPower':peak})
        finally:mm._mmap.close()
    return {'metadata':meta,'points':points,'note':'Contiguous power bins, not individual packets. Full-resolution data remains on disk.'}
