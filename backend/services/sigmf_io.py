"""Bounded SigMF core decoding and derived-window export.

Implements the sample layouts and index semantics in https://sigmf.org/ (1.2.6).
Optional extensions remain opaque provenance; required extensions are refused.
This is deliberately not an unrestricted archive reader or full schema validator.
"""
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import io
import json
import re
import zipfile

import numpy as np


CORE_VERSION = '1.2.6'
MAX_INDEX = 2**63 - 1
MAX_METADATA_DEPTH = 64
COMPONENT_TYPES = {'i8':'i1', 'u8':'u1',
                   **{f'{kind}{bits}_{end}':prefix+kind+str(bits//8)
                      for kind,bits in [('i',16),('i',32),('u',16),('u',32),('f',32),('f',64)]
                      for end,prefix in [('le','<'),('be','>')]}}
DATATYPES = {prefix+name:(np.dtype(dtype),2 if prefix=='c' else 1)
             for prefix in ['c','r'] for name,dtype in COMPONENT_TYPES.items()}


def integer(value, name, low=0, high=MAX_INDEX):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f'{name} must be an integer between {low} and {high}.')
    return value


def number(value, name, low=-1e12, high=1e12):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
        raise ValueError(f'{name} must be a finite number between {low} and {high}.')
    return value


def utc(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',value):
        raise ValueError('SigMF timestamps must be RFC 3339 UTC strings with a Z suffix.')
    try:return datetime.fromisoformat(value[:-1]+'+00:00')
    except ValueError as exc:
        raise ValueError('Invalid SigMF UTC timestamp; leap-second conversion is unsupported.') from exc


def geolocation(value):
    if not isinstance(value,dict) or value.get('type')!='Point' or not isinstance(value.get('coordinates'),list) or len(value['coordinates']) not in [2,3]:
        raise ValueError('SigMF geolocation must be a GeoJSON Point with longitude, latitude and optional altitude.')
    number(value['coordinates'][0],'Geolocation longitude',-180,180)
    number(value['coordinates'][1],'Geolocation latitude',-90,90)
    if len(value['coordinates'])==3:number(value['coordinates'][2],'Geolocation altitude')


def validate_metadata_depth(metadata):
    """Bound opaque JSON before persistence or recursive response encoding."""
    pending=[(metadata,0)]
    while pending:
        value,depth=pending.pop()
        if depth>MAX_METADATA_DEPTH:
            raise ValueError(f'SigMF metadata nesting exceeds the supported depth of {MAX_METADATA_DEPTH}.')
        if isinstance(value,dict):pending.extend((child,depth+1) for child in value.values())
        elif isinstance(value,list):pending.extend((child,depth+1) for child in value)


@dataclass(frozen=True)
class Layout:
    datatype: str
    component: np.dtype
    components: int
    channels: int
    channel: int
    zero: float
    scale: float
    rate: float
    sample_offset: int
    samples: int
    headers: tuple
    trailing_bytes: int
    ignored_extensions: tuple

    @property
    def frame_bytes(self):return self.component.itemsize*self.components*self.channels

    def blocks(self, path, start, end):
        """Read selected channel only in bounded sample blocks, skipping NCD headers."""
        positions=[p for p,_ in self.headers]
        cumulative=[0]
        for _,size in self.headers:cumulative.append(cumulative[-1]+size)
        block_samples=min(65536,max(1,1024**2//self.frame_bytes))
        with path.open('rb') as stream:
            offset=start
            while offset<end:
                segment=bisect_right(positions,offset)
                stop=min(end,offset+block_samples,positions[segment] if segment<len(positions) else end)
                count=stop-offset
                stream.seek(offset*self.frame_bytes+cumulative[segment])
                raw=stream.read(count*self.frame_bytes)
                if len(raw)!=count*self.frame_bytes:raise ValueError('SigMF sample data is truncated.')
                block=np.frombuffer(raw,dtype=self.component).reshape(count,self.channels,self.components)[:,self.channel,:]
                converted=(block.astype(np.float64)-self.zero)*self.scale
                if self.components==1:converted=np.column_stack((converted[:,0],np.zeros(count)))
                yield converted
                offset=stop


def parse_metadata(metadata, options, size, filename, source_sha512):
    validate_metadata_depth(metadata)
    if not isinstance(metadata,dict) or not isinstance(metadata.get('global'),dict):
        raise ValueError('A SigMF metadata object with global properties is required.')
    if len(json.dumps(metadata,allow_nan=False).encode())>65536:raise ValueError('SigMF metadata exceeds 64 KiB.')
    g=metadata['global'];datatype=g.get('core:datatype')
    if not isinstance(datatype,str) or datatype not in DATATYPES:
        raise ValueError('Unsupported SigMF datatype. Supported core types: '+', '.join(DATATYPES))
    component,components=DATATYPES[datatype]
    for field in ['core:author','core:collection','core:description','core:hw','core:license','core:recorder','core:data_doi','core:meta_doi']:
        if field in g and not isinstance(g[field],str):raise ValueError(field+' must be a string.')
    if 'core:version' in g and (not isinstance(g['core:version'],str) or not re.fullmatch(r'[01]\.\d+\.\d+',g['core:version'])):
        raise ValueError('Unsupported SigMF core:version; this reader implements core 1.2.6 layouts.')
    rate=g.get('core:sample_rate',options.sampleRate)
    number(rate,'SigMF sample rate',1,1e9)
    if options.sampleRate is not None and 'core:sample_rate' in g and options.sampleRate!=rate:
        raise ValueError('Supplied sample rate conflicts with SigMF metadata; clear the override or correct the metadata.')
    channels=integer(g.get('core:num_channels',1),'SigMF channel count',1,64)
    if channels>1 and options.channelIndex is None:
        raise ValueError('Multichannel SigMF requires an explicit channelIndex; no channel is selected automatically.')
    channel=0 if options.channelIndex is None else options.channelIndex
    integer(channel,'Selected SigMF channel',0,channels-1)
    if component.kind=='u' and options.sampleZero is None:
        raise ValueError('Unsigned SigMF requires explicit sampleZero (ADC zero/reference level); no midpoint is guessed.')
    if components==1 and options.realMode!='i-only':
        raise ValueError("Real SigMF requires explicit realMode='i-only': Q=0 is recorded, not inferred or recovered.")
    if components==2 and options.realMode is not None:
        raise ValueError('realMode applies only to real-valued SigMF datatypes; complex recordings retain both I and Q.')
    zero=0. if options.sampleZero is None else float(options.sampleZero)
    offset=integer(g.get('core:offset',0),'SigMF core:offset')
    trailing=integer(g.get('core:trailing_bytes',0),'SigMF trailing bytes',0,size)
    if g.get('core:sha512') and (not isinstance(g['core:sha512'],str) or g['core:sha512'].lower()!=source_sha512):
        raise ValueError('SigMF data SHA-512 does not match metadata.')
    if 'core:dataset' in g and g['core:dataset']!=filename:
        raise ValueError('SigMF dataset must match the uploaded filename; external paths are never opened.')
    if 'core:metadata_only' in g and not isinstance(g['core:metadata_only'],bool):raise ValueError('SigMF metadata_only must be boolean.')
    extensions=g.get('core:extensions',[])
    if not isinstance(extensions,list) or len(extensions)>128:raise ValueError('Invalid SigMF extension declarations.')
    ignored=[]
    for e in extensions:
        if not isinstance(e,dict) or set(e)!={'name','version','optional'} or not isinstance(e['name'],str) or not e['name'] or not isinstance(e['version'],str) or not e['version'] or type(e['optional']) is not bool:
            raise ValueError('SigMF extensions require exactly name, version and boolean optional fields.')
        if not e['optional']:raise ValueError('Required SigMF extensions are not supported: '+e['name'])
        ignored.append(e['name'])
    if 'core:geolocation' in g:geolocation(g['core:geolocation'])
    captures=metadata.get('captures',[])
    if not isinstance(captures,list) or len(captures)>1024:raise ValueError('SigMF captures must be a bounded array of metadata objects.')
    previous=offset-1;headers=[]
    for capture in captures:
        if not isinstance(capture,dict):raise ValueError('SigMF captures must contain metadata objects.')
        start=integer(capture.get('core:sample_start'),'SigMF capture sample_start',offset)
        if start<=previous:raise ValueError('SigMF capture starts must be strictly increasing absolute sample indices at or above core:offset.')
        previous=start
        header=integer(capture.get('core:header_bytes',0),'SigMF per-capture header bytes',0,size)
        if header:headers.append((start-offset,header))
        if 'core:frequency' in capture:number(capture['core:frequency'],'SigMF center frequency')
        if 'core:datetime' in capture:utc(capture['core:datetime'])
        if 'core:global_index' in capture:integer(capture['core:global_index'],'SigMF global_index')
        if 'core:geolocation' in capture:geolocation(capture['core:geolocation'])
    if (headers or trailing) and ('core:dataset' not in g or filename.lower().endswith('.sigmf-data')):
        raise ValueError('SigMF headers/trailers require an explicitly named non-conforming dataset with core:dataset.')
    data_bytes=size-trailing-sum(h for _,h in headers)
    frame_bytes=component.itemsize*components*channels
    if data_bytes<=0 or data_bytes%frame_bytes:
        raise ValueError('SigMF byte length is not a whole number of declared channel/sample frames after headers and trailers.')
    samples=data_bytes//frame_bytes
    if offset+samples>MAX_INDEX:raise ValueError('SigMF sample index exceeds the supported signed 64-bit range.')
    if any(c['core:sample_start']>=offset+samples for c in captures):raise ValueError('SigMF capture starts lie outside the recorded samples.')
    annotations=metadata.get('annotations',[])
    if not isinstance(annotations,list) or len(annotations)>4096:raise ValueError('SigMF annotations must be a bounded array.')
    previous=offset
    for annotation in annotations:
        if not isinstance(annotation,dict):raise ValueError('SigMF annotations must contain metadata objects.')
        start=integer(annotation.get('core:sample_start'),'SigMF annotation sample_start',offset,offset+samples-1)
        if start<previous:raise ValueError('SigMF annotations must be sorted by absolute sample_start.')
        previous=start
        count=annotation_count(annotation,captures,offset+samples)
        if start+count>offset+samples:raise ValueError('SigMF annotation extends beyond the recorded samples.')
        low,high=annotation.get('core:freq_lower_edge'),annotation.get('core:freq_upper_edge')
        if (low is None)!=(high is None):raise ValueError('SigMF annotation frequency edges must be supplied together.')
        if low is not None:
            number(low,'SigMF lower frequency edge');number(high,'SigMF upper frequency edge')
            if low>high:raise ValueError('SigMF annotation frequency edges are reversed.')
        for field in ['core:label','core:comment','core:generator','core:uuid']:
            if field in annotation and not isinstance(annotation[field],str):raise ValueError(field+' must be a string.')
    return Layout(datatype,component,components,channels,channel,zero,float(options.scale),float(rate),offset,samples,tuple(headers),trailing,tuple(ignored))


def annotation_count(annotation, captures, end):
    if 'core:sample_count' in annotation:return integer(annotation['core:sample_count'],'SigMF annotation sample_count')
    start=annotation['core:sample_start']
    return next((c['core:sample_start']-start for c in captures if c['core:sample_start']>start),end-start)


def export_capture(record_id, start=0, length=128):
    """ZIP of a bounded canonical window, valid derived metadata and original metadata.

    The export is an exact copy of managed float32 samples, not the original ADC
    bytes. Header removal/channel selection/zero/scale conversion are explicit.
    """
    from backend.services import captures
    iq,meta=captures.window(record_id,start,length)
    original=meta.get('sigmf')
    raw=iq.astype('<f4').tobytes();first=meta.get('sigmfSampleOffset',0)+meta['sourceOffset']+start;end=first+length
    g={'core:datatype':'cf32_le','core:version':CORE_VERSION,'core:num_channels':1,'core:offset':first,
       'core:sha512':hashlib.sha512(raw).hexdigest(),
       'core:description':'SatModClass derived analysis window; canonical float32 I/Q, not original ADC bytes.',
       'core:extensions':[{'name':'satmod','version':'1.0.0','optional':True}],
       'satmod:provenance':{'captureId':meta['id'],'captureHash':meta['hash'],'sourceHash':meta['sourceHash'],
                            'sourceDatatype':meta['datatype'],'selectedChannel':meta.get('channelIndex',0),
                            'sourceChannels':meta.get('sourceChannels',1),'sampleZero':meta.get('sampleZero') or 0,
                            'scale':meta['scale'],'realMode':meta.get('realMode'),'fileSampleStart':meta['sourceOffset']+start,
                            'conversion':'Stored little-endian float32 I/Q after explicit source zero/scale and channel selection; no resampling.',
                            'originalMetadataFile':'source.sigmf-meta' if original is not None else None}}
    if meta.get('sampleRateKnown'):g['core:sample_rate']=meta['sampleRate']
    source_global=original.get('global',{}) if original else {}
    for field in ['core:author','core:license','core:hw']:
        if field in source_global:g[field]=deepcopy(source_global[field])
    segments=original.get('captures',[]) if original else []
    active=next((c for c in reversed(segments) if c['core:sample_start']<=first),None)
    selected=[(first,active)]+[(c['core:sample_start'],c) for c in segments if first<c['core:sample_start']<end]
    derived=[]
    for position,source in selected:
        entry={'core:sample_start':position}
        if source is not None:
            delta=position-source['core:sample_start']
            if 'core:frequency' in source:entry['core:frequency']=source['core:frequency']
            if 'core:datetime' in source:
                try:entry['core:datetime']=(utc(source['core:datetime'])+timedelta(seconds=delta/meta['sampleRate'])).isoformat().replace('+00:00','Z')
                except OverflowError as exc:raise ValueError('Derived SigMF timestamp exceeds the supported datetime range.') from exc
            if 'core:global_index' in source:entry['core:global_index']=integer(source['core:global_index']+delta,'Derived SigMF global_index')
            if delta==0 and 'core:geolocation' in source:entry['core:geolocation']=deepcopy(source['core:geolocation'])
        elif meta.get('centerFrequency') is not None:entry['core:frequency']=meta['centerFrequency']
        derived.append(entry)
    annotations=[]
    for source in original.get('annotations',[]) if original else []:
        lo=source['core:sample_start'];hi=lo+annotation_count(source,segments,meta.get('sigmfSampleOffset',0)+meta['sourceSampleCount'])
        left,right=max(first,lo),min(end,hi)
        if right<=left:continue
        entry={key:deepcopy(value) for key,value in source.items() if key.startswith('core:')}
        entry.update({'core:sample_start':left,'core:sample_count':right-left})
        if left!=lo or right!=hi:entry['satmod:original_span']={'start':lo,'count':hi-lo}
        annotations.append(entry)
    result={'global':g,'captures':derived,'annotations':annotations}
    encoded=json.dumps(result,indent=2,allow_nan=False).encode()
    if len(encoded)>65536:raise ValueError('Derived SigMF metadata exceeds 64 KiB; choose a smaller window.')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('capture.sigmf-data',raw)
        archive.writestr('capture.sigmf-meta',encoded)
        if original is not None:archive.writestr('source.sigmf-meta',json.dumps(original,indent=2,allow_nan=False).encode())
    return output.getvalue()
