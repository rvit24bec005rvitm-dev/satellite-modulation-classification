"""SigMF core layouts and metadata, exercised with explicitly generated fixtures."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import captures, operations, sigmf_io, workspace_store as store


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SATMOD_DATA_DIR',str(tmp_path/'sigmf-workspace'))
    monkeypatch.delenv('SATMOD_PUBLIC',raising=False)
    monkeypatch.delenv('SATMOD_ACCESS_TOKEN',raising=False)
    with TestClient(app) as session:yield session
    runner=operations.RUNNERS.pop(str(store.root()),None)
    if runner:runner[0].shutdown(wait=True)
    store.close_connections()


def metadata(datatype, **global_fields):
    return {'global':{'core:datatype':datatype,'core:version':'1.2.6','core:sample_rate':8000,**global_fields},
            'captures':[{'core:sample_start':0,'core:frequency':137e6,'core:datetime':'2026-11-12T00:00:00Z'}],
            'annotations':[]}


def upload(client, data, meta, options=None, filename='fixture.sigmf-data'):
    return client.post('/api/workspace/captures',data={'options':json.dumps({
        'format':'sigmf','sessionId':'generated-sigmf-fixture','source':'Generated SigMF fixture, not RF',**(options or {})})},
        files={'file':(filename,data),'metadata':('fixture.sigmf-meta',json.dumps(meta).encode())})


def contents(client, record_id):
    value=client.get(f'/api/workspace/captures/{record_id}/window?length=256').json()
    return np.asarray([[p['i'],p['q']] for p in value['samples']],dtype=np.float32)


@pytest.mark.parametrize('datatype',list(sigmf_io.DATATYPES))
def test_every_core_datatype_decodes_and_roundtrips_derived_export(client,datatype):
    dtype,components=sigmf_io.DATATYPES[datatype]
    original=(np.arange(256*components).reshape(256,components)%13)-6
    zero=128 if dtype.kind=='u' else 0
    encoded=(original+zero).astype(dtype)
    options={'scale':.001}
    if dtype.kind=='u':options['sampleZero']=zero
    if components==1:options['realMode']='i-only'
    meta=metadata(datatype)
    meta['global']['core:sha512']=hashlib.sha512(encoded.tobytes()).hexdigest()
    before=deepcopy(meta)
    response=upload(client,encoded.tobytes(),meta,options)
    assert response.status_code==200,response.text
    record=response.json()
    expected=(original.astype(float)*.001).astype(np.float32)
    if components==1:expected=np.column_stack((expected[:,0],np.zeros(256,dtype=np.float32)))
    np.testing.assert_array_equal(contents(client,record['id']),expected)
    assert record['datatype']==datatype and record['sourceComponents']==components
    assert record['sampleZero']==zero and meta==before
    exported=sigmf_io.export_capture(record['id'],0,256)
    with zipfile.ZipFile(io.BytesIO(exported)) as archive:
        raw=archive.read('capture.sigmf-data')
        derived=json.loads(archive.read('capture.sigmf-meta'))
        source=json.loads(archive.read('source.sigmf-meta'))
    assert source==meta
    assert derived['global']['core:datatype']=='cf32_le'
    assert derived['global']['core:sha512']==hashlib.sha512(raw).hexdigest()
    assert derived['global']['satmod:provenance']['sourceDatatype']==datatype
    assert derived['global']['satmod:provenance']['realMode']==options.get('realMode')
    roundtrip=upload(client,raw,derived,filename='capture.sigmf-data')
    assert roundtrip.status_code==200,roundtrip.text
    np.testing.assert_array_equal(contents(client,roundtrip.json()['id']),expected)


@pytest.mark.parametrize('datatype',['cf32_le','cf64_be','ci16_be','cu16_le','rf64_le','ru32_be'])
@pytest.mark.parametrize('channel',[0,1])
def test_multichannel_selection_is_explicit_and_uses_channel_frame_order(client,datatype,channel):
    dtype,components=sigmf_io.DATATYPES[datatype]
    value=np.arange(256*2*components).reshape(256,2,components)%29
    options={'channelIndex':channel,'scale':.0001}
    if dtype.kind=='u':options['sampleZero']=10
    if components==1:options['realMode']='i-only'
    response=upload(client,value.astype(dtype).tobytes(),metadata(datatype,**{'core:num_channels':2}),options)
    assert response.status_code==200,response.text
    selected=(value[:,channel,:].astype(float)-(10 if dtype.kind=='u' else 0))*.0001
    if components==1:selected=np.column_stack((selected[:,0],np.zeros(256)))
    np.testing.assert_array_equal(contents(client,response.json()['id']),selected.astype(np.float32))
    assert response.json()['sourceChannels']==2 and response.json()['channelIndex']==channel


def test_ncd_headers_trailers_and_absolute_sample_offsets(client):
    iq=(np.arange(256*2).reshape(256,2)%17).astype('<i2')
    raw=b'HDR'+iq[:128].tobytes()+b'HEAD2'+iq[128:].tobytes()+b'TRAILER'
    meta=metadata('ci16_le',**{'core:dataset':'recording.dat','core:offset':1000,'core:trailing_bytes':7,
                              'core:sha512':hashlib.sha512(raw).hexdigest()})
    meta['captures']=[{'core:sample_start':1000,'core:global_index':7000,'core:header_bytes':3,
                       'core:datetime':'2026-11-12T00:00:00Z','core:frequency':137e6},
                      {'core:sample_start':1128,'core:global_index':8000,'core:header_bytes':5,
                       'core:datetime':'2026-11-12T00:00:01Z','core:frequency':138e6}]
    response=upload(client,raw,meta,{'sampleOffset':64,'sampleCount':128,'scale':.001},filename='recording.dat')
    assert response.status_code==200,response.text
    record=response.json();data,stored=captures.window(record['id'])
    np.testing.assert_array_equal(data,(iq[64:192].astype(float)*.001).astype(np.float32))
    assert stored['sourceOffset']==64 and stored['sigmfSampleOffset']==1000
    assert captures.timing(stored,0)['windowCrossesCaptureSegment']
    assert captures.timing(stored,0)['captureBoundariesWithinWindow']==[1128]
    position=captures.timing(stored,64)
    assert position['sourceSample']==1128 and position['fileSample']==128
    assert position['globalSampleIndex']==8000 and position['timestamp']=='2026-11-12T00:00:01+00:00'
    assert position['centerFrequency']==138e6
    with zipfile.ZipFile(io.BytesIO(sigmf_io.export_capture(record['id'],0,128))) as archive:
        derived=json.loads(archive.read('capture.sigmf-meta'));data=archive.read('capture.sigmf-data')
    assert derived['global']['core:offset']==1064
    assert 'core:trailing_bytes' not in derived['global'] and 'core:dataset' not in derived['global']
    assert [c['core:sample_start'] for c in derived['captures']]==[1064,1128]
    assert derived['captures'][0]['core:global_index']==7064
    assert derived['captures'][0]['core:datetime']=='2026-11-12T00:00:00.008000Z'
    assert all('core:header_bytes' not in c for c in derived['captures'])
    again=upload(client,data,derived,filename='capture.sigmf-data')
    assert again.status_code==200,again.text
    np.testing.assert_array_equal(captures.window(again.json()['id'])[0],captures.window(record['id'])[0])


def test_scoped_segment_metadata_does_not_carry_over_and_annotations_are_clipped(client):
    raw=np.zeros((256,2),dtype='<f4').tobytes()
    meta=metadata('cf32_le',**{'core:offset':500})
    meta['captures']=[{'core:sample_start':500,'core:datetime':'2026-11-12T00:00:00Z','core:frequency':-125},
                      {'core:sample_start':628}]
    meta['annotations']=[{'core:sample_start':600,'core:label':'known region'},
                         {'core:sample_start':620,'core:sample_count':80,'core:comment':'crosses selection'}]
    response=upload(client,raw,meta,{'sampleOffset':100,'sampleCount':128})
    assert response.status_code==200,response.text
    record=response.json()
    first=captures.timing(record,0);later=captures.timing(record,28)
    assert first['centerFrequency']==-125 and first['sourceSample']==600
    assert later['timestamp'] is None and later['centerFrequency'] is None
    with zipfile.ZipFile(io.BytesIO(sigmf_io.export_capture(record['id'],16,112))) as archive:
        derived=json.loads(archive.read('capture.sigmf-meta'))
    assert derived['annotations'][0]['core:sample_start']==616
    assert derived['annotations'][0]['core:sample_count']==12
    assert derived['annotations'][0]['satmod:original_span']=={'start':600,'count':28}
    assert derived['captures'][1]=={'core:sample_start':628}


def test_optional_extensions_preserved_without_applying_unknown_calibration(client):
    raw=np.ones((256,2),dtype='<f4').tobytes()
    meta=metadata('cf32_le',**{'core:extensions':[{'name':'antenna','version':'1.0.0','optional':True}],
                              'antenna:model':'Fixture antenna description','antenna:gain':99})
    before=deepcopy(meta)
    response=upload(client,raw,meta)
    assert response.status_code==200,response.text
    record=response.json()
    assert record['sigmf']==before and record['ignoredOptionalExtensions']==['antenna']
    np.testing.assert_array_equal(contents(client,record['id']),np.ones((256,2)))
    with zipfile.ZipFile(io.BytesIO(sigmf_io.export_capture(record['id']))) as archive:
        assert json.loads(archive.read('source.sigmf-meta'))==before
        derived=json.loads(archive.read('capture.sigmf-meta'))
        assert 'antenna:gain' not in derived['global']


@pytest.mark.parametrize('datatype,options,message',[
    ('cu8',{},'sampleZero'),('ru16_be',{'sampleZero':32768},'realMode'),
    ('ri16_le',{},'realMode'),('cf32_le',{'channelIndex':1},'channel'),
    ('cf32_le',{'realMode':'i-only'},'realMode'),
])
def test_explicit_conversion_and_channel_configuration_required(client,datatype,options,message):
    dtype,components=sigmf_io.DATATYPES[datatype]
    response=upload(client,np.zeros((256,components),dtype=dtype).tobytes(),metadata(datatype),options)
    assert response.status_code==422 and message in response.text
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))


@pytest.mark.parametrize('mutate',[
    lambda m:m['global'].update({'core:num_channels':2}),
    lambda m:m['global'].update({'core:datatype':'cf16_le'}),
    lambda m:m['global'].update({'core:version':'2.0.0'}),
    lambda m:m['global'].update({'core:offset':True}),
    lambda m:m['global'].update({'core:offset':1000}),
    lambda m:m['global'].update({'core:dataset':'../../elsewhere.dat'}),
    lambda m:m['global'].update({'core:trailing_bytes':1}),
    lambda m:m['global'].update({'core:extensions':[{'name':'unknown','version':'1.0','optional':False}]}),
    lambda m:m['global'].update({'core:extensions':[{'name':'invalid','optional':True}]}),
    lambda m:m['captures'][0].update({'core:sample_start':256}),
    lambda m:m['captures'][0].update({'core:header_bytes':4}),
    lambda m:m['captures'][0].update({'core:frequency':True}),
    lambda m:m['captures'][0].update({'core:datetime':'2026-01-01T00:00:00+00:00'}),
    lambda m:m['captures'].append({'core:sample_start':0}),
    lambda m:m['annotations'].append({'core:sample_start':255,'core:sample_count':2}),
    lambda m:m['annotations'].append({'core:sample_start':0,'core:freq_lower_edge':1}),
    lambda m:m['annotations'].append({'core:sample_start':0,'core:freq_lower_edge':2,'core:freq_upper_edge':1}),
])
def test_malformed_metadata_rejected_with_managed_file_cleanup(client,mutate):
    meta=metadata('cf32_le');mutate(meta)
    response=upload(client,np.zeros((256,2),dtype='<f4').tobytes(),meta)
    assert response.status_code==422,response.text
    assert not list(store.root().glob('*.upload')) and not list(store.root().glob('*.iq'))


def test_missing_clock_can_only_be_filled_by_explicit_operator_metadata(client):
    meta=metadata('cf32_le');meta['global'].pop('core:sample_rate')
    raw=np.zeros((256,2),dtype='<f4').tobytes()
    assert upload(client,raw,meta).status_code==422
    assert upload(client,raw,meta,{'sampleRate':16000}).status_code==200
    meta['global']['core:sample_rate']=8000
    response=upload(client,raw,meta,{'sampleRate':16000})
    assert response.status_code==422 and 'conflicts' in response.text


def test_bounded_multichannel_reads_and_truncated_frames(client):
    dtype,components=sigmf_io.DATATYPES['cf64_be']
    options=captures.ImportOptions(format='sigmf',sessionId='fixture',channelIndex=63)
    meta=metadata('cf64_be',**{'core:num_channels':64})
    layout=sigmf_io.parse_metadata(meta,options,4096*64*components*dtype.itemsize,'fixture.sigmf-data','unused')
    assert layout.frame_bytes==1024
    values=np.arange(4096*64*components).reshape(4096,64,components)%19
    raw=values.astype(dtype).tobytes();requests=[]
    class RecordingStream(io.BytesIO):
        def read(self,size=-1):
            requests.append(size)
            return super().read(size)
    class RecordingPath:
        def open(self,mode):return RecordingStream(raw)
    blocks=list(layout.blocks(RecordingPath(),0,4096))
    assert len(blocks)>1 and all(0<size<=1024**2 for size in requests)
    np.testing.assert_array_equal(np.concatenate(blocks),values[:,63,:])
    response=upload(client,b'\x00'*2049,metadata('cf32_le'))
    assert response.status_code==422 and 'whole number' in response.text


def test_export_unknown_clock_never_claims_one_hz(client):
    stream=io.BytesIO();np.save(stream,np.zeros((256,2),dtype=np.float32),allow_pickle=False)
    response=client.post('/api/workspace/captures',data={'options':json.dumps({'sessionId':'fixture'})},
                         files={'file':('fixture.npy',stream.getvalue())})
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(sigmf_io.export_capture(response.json()['id']))) as archive:
        derived=json.loads(archive.read('capture.sigmf-meta'))
        assert 'core:sample_rate' not in derived['global']
        assert 'source.sigmf-meta' not in archive.namelist()


@pytest.mark.parametrize('field,value,message',[
    ('core:datetime','9999-12-31T23:59:59Z','timestamp exceeds'),
    ('core:global_index',sigmf_io.MAX_INDEX,'Derived SigMF global_index'),
])
def test_derived_position_overflow_is_a_descriptive_validation_error(client,field,value,message):
    meta=metadata('cf32_le',**{'core:sample_rate':1})
    meta['captures'][0]={ 'core:sample_start':0,field:value }
    response=upload(client,np.zeros((256,2),dtype='<f4').tobytes(),meta)
    assert response.status_code==200,response.text
    record=response.json()
    with pytest.raises(ValueError,match=message):captures.timing(record,1)
    with pytest.raises(ValueError,match=message):sigmf_io.export_capture(record['id'],1,128)
    inspected=client.get(f"/api/workspace/captures/{record['id']}/window?start=1")
    assert inspected.status_code==422 and message in inspected.text
    exported=client.get(f"/api/workspace/captures/{record['id']}/sigmf?start=1")
    assert exported.status_code==422 and message in exported.text


@pytest.mark.parametrize('background',[False,True])
def test_deep_opaque_metadata_is_rejected_before_capture_or_job_persistence(client,background):
    raw=b'{"global":{"core:datatype":"cf32_le","core:sample_rate":8000,"fixture:nested":'+b'['*20000+b'0'+b']'*20000+b'}}'
    response=client.post('/api/workspace/captures',params={'background':str(background).lower()},
                         data={'options':json.dumps({'format':'sigmf','sessionId':'generated-depth-fixture'})},
                         files={'file':('fixture.sigmf-data',np.zeros((256,2),dtype='<f4').tobytes()),
                                'metadata':('fixture.sigmf-meta',raw)})
    assert response.status_code==422 and 'nesting' in response.text
    assert client.get('/api/workspace/captures').json()['total']==0
    assert client.get('/api/workspace/jobs').json()['total']==0
    assert not list(store.root().glob('*.iq')) and not list(store.root().glob('*.upload'))


def test_depth_guard_applies_to_service_callers_and_preserves_safe_opaque_metadata(client):
    nested=0
    for _ in range(32):nested=[nested]
    meta=metadata('cf32_le',**{'fixture:nested':nested})
    response=upload(client,np.zeros((256,2),dtype='<f4').tobytes(),meta)
    assert response.status_code==200,response.text
    assert response.json()['sigmf']==meta
    nested=0
    for _ in range(65):nested=[nested]
    meta['global']['fixture:nested']=nested
    with pytest.raises(ValueError,match='nesting'):
        sigmf_io.parse_metadata(meta,captures.ImportOptions(format='sigmf',sessionId='generated-depth-fixture'),2048,'fixture.sigmf-data','unused')
