"""SatModClass REST API. Launch from the repository with uvicorn backend.main:app."""
import logging
import json
import struct
import zlib
import zipfile
from time import perf_counter
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np
from backend.core.config import VERSION, CORS_ORIGINS, MAX_UPLOAD_BYTES
from backend.api.schemas import ClassifyRequest, ClassificationResult, SignalRequest, GenerateRequest, CompareRequest, BenchmarkRequest, HealthStatus, ExplainRequest, ExplanationResult
from backend.services import classifier as svc
from backend.services.signals import analyze, parse_capture, from_rows, rows
from backend.services.evaluation import evaluation, snr_results
from backend.services.model_registry import REGISTRY, describe_models, benchmark_model
from backend.services.explain import explain as explain_signal
from utils.signal_processing import generate_iq_signal, CONSTELLATION_REFS, simulate_orbit_pass, generate_audio_waveform

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('satmod')
app = FastAPI(title='SatModClass — Satellite Signal Intelligence',version=VERSION)
import threading
IMPORT_SLOTS=threading.BoundedSemaphore(2)

class WorkspaceAccess:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        import os,hmac
        if scope['type']=='http' and scope['path'].startswith('/api/') and scope['path']!='/api/health' and scope['method']!='OPTIONS':
            expected=os.getenv('SATMOD_ACCESS_TOKEN','')
            supplied=dict(scope.get('headers',[])).get(b'x-satmod-token',b'').decode('latin1')
            if os.getenv('SATMOD_PUBLIC')=='1' and not expected:
                return await JSONResponse(status_code=503,content={'detail':'Public workspace access is not configured.'})(scope,receive,send)
            if expected and not hmac.compare_digest(supplied,expected):
                return await JSONResponse(status_code=401,content={'detail':'Workspace access token required. Configure it in Settings.'})(scope,receive,send)
        await self.app(scope,receive,send)

class BodyLimit:
    """Bound both Content-Length and streamed/chunked bodies before parsing."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        import os
        limit = (int(os.getenv('SATMOD_CAPTURE_UPLOAD_BYTES',str(128*1024**2)))+131072 if scope['path']=='/api/workspace/captures'
                 else MAX_UPLOAD_BYTES + 65536 if scope['path'].endswith('/upload') else 16*1024*1024)
        content_length = dict(scope.get('headers', [])).get(b'content-length')
        if content_length is not None:
            # HTTP Content-Length is decimal digits only, never a signed value.
            if not content_length or not content_length.isdigit():
                return await JSONResponse(status_code=400,content={'detail':'Invalid Content-Length.'})(scope,receive,send)
            try: declared = int(content_length)
            except ValueError:
                return await JSONResponse(status_code=400,content={'detail':'Invalid Content-Length.'})(scope,receive,send)
            if declared > limit:
                return await JSONResponse(status_code=413,content={'detail':'Request body exceeds the configured size limit.'})(scope,receive,send)
        # Managed imports are read/spooled by Starlette; bound their stream
        # without accumulating a second large in-memory body.
        if scope['path']=='/api/workspace/captures':
            from tempfile import SpooledTemporaryFile
            if not IMPORT_SLOTS.acquire(blocking=False):
                return await JSONResponse(status_code=429,content={'detail':'Two capture imports are already active.'})(scope,receive,send)
            try:
                with SpooledTemporaryFile(max_size=1024**2) as spool:
                    size=0
                    while True:
                        message=await receive()
                        if message['type']=='http.disconnect':return
                        chunk=message.get('body',b'');size+=len(chunk)
                        if size>limit:return await JSONResponse(status_code=413,content={'detail':'External capture request exceeds its size limit.'})(scope,receive,send)
                        spool.write(chunk)
                        if not message.get('more_body',False):break
                    spool.seek(0);remaining=size
                    async def limited_receive():
                        nonlocal remaining
                        if remaining:
                            data=spool.read(min(65536,remaining));remaining-=len(data)
                            return {'type':'http.request','body':data,'more_body':remaining>0}
                        return await receive()
                    await self.app(scope,limited_receive,send)
            finally:IMPORT_SLOTS.release()
            return
        size, payload = 0, bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect': return
            size += len(message.get('body',b''))
            if size > limit:
                return await JSONResponse(status_code=413,content={'detail':'Request body exceeds the configured size limit.'})(scope,receive,send)
            payload.extend(message.get('body',b''))
            if not message.get('more_body',False): break
        buffered = bytes(payload)
        del payload
        consumed = False
        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {'type':'http.request','body':buffered,'more_body':False}
            return await receive()
        await self.app(scope,bounded_receive,send)
app.add_middleware(BodyLimit)
app.add_middleware(WorkspaceAccess)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,allow_credentials=False,allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type','X-Satmod-Token'])

@app.exception_handler(ValueError)
async def invalid_input(request: Request, exc: ValueError):
    return JSONResponse(status_code=422,content={'detail':str(exc)})

@app.exception_handler(KeyError)
async def record_missing(request: Request,exc: KeyError):
    return JSONResponse(status_code=404,content={'detail':'Managed record not found.'})

@app.exception_handler(RequestValidationError)
async def invalid_schema(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,content={'detail':'; '.join(f"{'.'.join(map(str,e['loc']))}: {e['msg']}" for e in exc.errors())})

@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    logger.exception(json.dumps({'event':'request_failed','path':request.url.path,'error':type(exc).__name__}))
    response = JSONResponse(status_code=500,content={'detail':'The signal service could not complete this request. Check the backend log.'})
    if getattr(request.state,'request_id',None):response.headers['X-Request-ID']=request.state.request_id
    # Starlette's outer ServerErrorMiddleware creates this response outside the
    # normal CORS middleware. Apply the same configured origin policy here so
    # a separate-origin frontend can read the error without exposing details.
    origin = request.headers.get('origin')
    if origin and '*' in CORS_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin in CORS_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers.add_vary_header('Origin')
    return response

@app.middleware('http')
async def log_request(request: Request, call_next):
    from uuid import uuid4
    request_id=str(uuid4());request.state.request_id=request_id
    t0 = perf_counter()
    response = await call_next(request)
    response.headers['X-Request-ID']=request_id
    logger.info(json.dumps({'event':'request','requestId':request_id,'method':request.method,'path':request.url.path,'status':response.status_code,'elapsedMs':round((perf_counter()-t0)*1000,2)}))
    return response

@app.get('/api/health',response_model=HealthStatus)
def health():
    try:
        backend = svc.model()
        svc.stats()
        return {'connected':True,'modelId':'1d-cnn','backend':backend.backend_name,'version':VERSION}
    except Exception as exc:
        logger.error(json.dumps({'event':'model_unavailable','error':type(exc).__name__}))
        return JSONResponse(status_code=503,content={'connected':False,'modelId':'1d-cnn','error':'Model or training statistics unavailable; inspect backend logs.','version':VERSION})

@app.get('/api/system/info')
def system_info():
    mean,std,classes = svc.stats()
    return {'name':'SatModClass','version':VERSION,'classes':classes,'inputShape':[128,2],
            'architecture':'Conv1D(32,5,same,ReLU) → BN → MaxPool → Conv1D(64,5,same,ReLU) → BN → MaxPool → Conv1D(96,3,same,ReLU) → BN → GlobalAveragePool → Dense(64,ReLU) → Dense(3,Softmax)',
            'normalization':{'mean':mean,'std':std,'policy':'Preserve amplitude; training z-score only'},
            'defaultAmplitudeRms':float(std*np.sqrt(2)), 'maxUploadBytes':MAX_UPLOAD_BYTES,'maxSamples':65536,
            'generatorModulations':['BPSK','QPSK','16-QAM','32-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK'],
            'historyPolicy':'SQLite-backed single shared workspace; managed reports persist across restarts. Clear/delete through history controls.'}

@app.get('/api/models')
def models():
    return describe_models()

@app.post('/api/classify',response_model=ClassificationResult)
def classify(body: ClassifyRequest):
    try: svc.model()
    except Exception as exc: raise HTTPException(503,str(exc)) from exc
    return svc.classify(from_rows(body.samples),body.source,body.sampleRate,body.windowStart)

@app.post('/api/classify/upload',response_model=ClassificationResult)
def upload(file: UploadFile = File(...), windowStart: int = Query(0,ge=0,le=65408)):
    try:
        payload = file.file.read(MAX_UPLOAD_BYTES+1)
        if len(payload)>MAX_UPLOAD_BYTES: raise HTTPException(413,'Capture exceeds upload size limit.')
        try: iq,fs = parse_capture(file.filename or 'capture',payload)
        except (ValueError, OSError, KeyError, IndexError, EOFError, TypeError, struct.error, zlib.error, zipfile.BadZipFile) as exc:
            raise HTTPException(422,f'Invalid capture: {exc}') from exc
        try: svc.model()
        except Exception as exc: raise HTTPException(503,str(exc)) from exc
        return svc.classify(iq,(file.filename or 'upload')[:200],fs,windowStart)
    finally: file.file.close()

@app.post('/api/signal/upload')
def upload_signal(file: UploadFile = File(...)):
    try:
        payload=file.file.read(MAX_UPLOAD_BYTES+1)
        if len(payload)>MAX_UPLOAD_BYTES: raise HTTPException(413,'Capture exceeds upload size limit.')
        try: iq,fs=parse_capture(file.filename or 'capture',payload)
        except (ValueError,OSError,KeyError,IndexError,EOFError,TypeError,struct.error,zlib.error,zipfile.BadZipFile) as exc:
            raise HTTPException(422,f'Invalid capture: {exc}') from exc
        return {'samples':rows(iq),'sampleRate':fs,'source':(file.filename or 'Uploaded capture')[:200],'analysis':analyze(iq,fs)}
    finally: file.file.close()

@app.get('/api/samples')
def samples():
    d = svc.dataset()
    return [{'id':i,'modulation':str(label).replace('QAM16','16-QAM'),'snrDb':float(d['snrs'][i]),'sampleCount':128} for i,label in enumerate(d['labels'])]

@app.get('/api/samples/{sample_id}')
def sample(sample_id: int):
    d = svc.dataset()
    if not 0<=sample_id<len(d['signals']): raise HTTPException(404,'Sample does not exist.')
    iq = d['signals'][sample_id]
    return {'samples':rows(iq),'source':f'Demo sample {sample_id}','modulation':str(d['labels'][sample_id]).replace('QAM16','16-QAM'),'snrDb':float(d['snrs'][sample_id]),'sampleRate':1,'analysis':analyze(iq)}

@app.get('/api/constellations')
def constellation_tables():
    from utils.modulation import constellation
    names=['BPSK','QPSK','16-QAM','32-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK']
    return {'mapping':'Natural-binary symbol index, MSB first; educational APSK and rectangular 8×4 32-QAM, no DVB-S2 claim.',
            'energyPolicy':'Unit average ideal symbol energy; received pulse/channel samples need not land on these points.',
            'modulations':{name:rows(constellation(name)) for name in names}}

@app.post('/api/signal/generate')
def generate(body: GenerateRequest):
    if abs(body.dopplerHz+body.cfoHz) > body.sampleRate/2:
        raise ValueError('Combined carrier offset must be within the Nyquist interval ±sampleRate/2.')
    mod = {'32-QAM':'QAM32','16-QAM':'QAM16','8-PSK':'8PSK','16-APSK':'16APSK','32-APSK':'32APSK'}.get(body.modulation,body.modulation)
    iq, generation = generate_iq_signal(mod,num_samples=body.numSamples,sps=body.sps,snr_db=body.snrDb,rrc_alpha=body.rrcAlpha,
            doppler_hz=body.dopplerHz,cfo_hz=body.cfoHz,sample_rate=body.sampleRate,phase_noise_std=body.phaseNoiseStd,
            rician_k=body.ricianK if body.ricianEnabled else 0,rician_enabled=body.ricianEnabled,
            twta_alpha_a=2.1587 if body.twtaDrive>0 else 0,twta_drive=body.twtaDrive,rng_seed=body.seed,rrc_enabled=body.rrcEnabled, gaussian_bt=body.gaussianBt,
            bfsk_deviation_hz=body.bfskDeviationHz, continuous_phase=body.continuousPhase,
            apsk_radii=body.apskRadii, apsk_offsets=body.apskOffsets, phase_noise_mode=body.phaseNoiseMode,
            fading_block_samples=body.fadingBlockSamples, normalize_output=body.normalizeOutput, return_metadata=True)
    iq *= body.amplitudeRms
    generation['outputAmplitudeScale'] = body.amplitudeRms
    return {'samples':rows(iq),'modulation':body.modulation,'snrDb':body.snrDb,'sampleRate':body.sampleRate,
            'source':f'Synthetic {body.modulation} at {body.snrDb} dB','parameters':body.model_dump(),
            'generation':generation, 'reference':rows(np.asarray(generation['reference']).reshape(-1,2)*body.amplitudeRms),
            'classifierSupported':body.modulation in ['BPSK','QPSK','16-QAM'], 'analysis':analyze(iq,body.sampleRate,body.sps if mod not in ['GMSK','BFSK'] else None)}

@app.post('/api/signal/analyze')
def analysis(body: SignalRequest):
    return analyze(from_rows(body.samples),body.sampleRate)

@app.post('/api/signal/compare')
def compare(body: CompareRequest):
    begin = perf_counter()
    a,b = from_rows(body.first.samples),from_rows(body.second.samples)
    aa,bb = analyze(a,body.first.sampleRate),analyze(b,body.second.sampleRate)
    normalized = body.first.sampleRate == 1 or body.second.sampleRate == 1
    def prediction(iq):
        if len(iq) < 128: return {'status':'unavailable', 'reason':'A complete 128-sample classification window is required.'}
        try: return {'status':'available', **svc.predict_window(iq)}
        except RuntimeError as exc: return {'status':'unavailable', 'reason':str(exc)}
    def plot_data(value):
        fs = value['sampleRate']
        return {'waveform':[{'x':p['sample'] if normalized else p['time'], 'i':p['i'], 'q':p['q']} for p in value['waveform']],
                'spectrum':[{'frequency':p['frequency']/fs if normalized else p['frequency'],
                             'psd':p['psd']+10*np.log10(fs) if normalized else p['psd']} for p in value['spectrum']]}
    n = min(len(a),len(b))
    return {'first':aa, 'second':bb, 'predictions':{'first':prediction(a), 'second':prediction(b)},
            'processingMs':(perf_counter()-begin)*1000,
            'powerDifferenceDb':aa['powerDb']-bb['powerDb'] if aa['powerDb'] is not None and bb['powerDb'] is not None else None,
            'sampleMse':float(np.mean((a[:n].astype(float)-b[:n])**2)) if body.first.sampleRate==body.second.sampleRate else None,
            'overlapSamples':n, 'lengthsEqual':len(a)==len(b), 'ratesEqual':body.first.sampleRate==body.second.sampleRate,
            'plots':{'first':plot_data(aa),'second':plot_data(bb), 'frequencyUnit':'cycles/sample' if normalized else 'Hz',
                     'timeUnit':'Sample index' if normalized else 'Time (s)'},
            'note':'MSE uses overlapping unaligned samples only at equal sample rates; it is not a modulation similarity score. '
                   + ('At least one sample rate is unknown (Fs=1): both plots use normalized sample/frequency axes. No resampling is inferred.' if normalized else 'Physical frequency/time axes use each capture’s supplied sample clock. No alignment or resampling is inferred.')}

@app.post('/api/explain',response_model=ExplanationResult)
def explain(body: ExplainRequest):
    try: svc.model()
    except RuntimeError as exc: raise HTTPException(503, str(exc)) from exc
    try: return explain_signal(from_rows(body.samples), body.windowStart, body.targetClass, body.method, body.steps)
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc

@app.get('/api/experiments')
def experiments():
    from backend.services.experiments import experiment_reports
    return experiment_reports()

@app.get('/api/evaluation')
def evaluation_results(): return evaluation()

@app.get('/api/evaluation/diagnostics')
def diagnostics():
    from backend.services.diagnostics import demo_diagnostics
    return demo_diagnostics()

@app.get('/api/evaluation/snr')
def snr(): return snr_results()

@app.post('/api/benchmark')
def benchmark(body: BenchmarkRequest):
    if body.modelId not in REGISTRY: raise HTTPException(404, 'Model is not registered.')
    if REGISTRY[body.modelId].loader is None: raise HTTPException(409, 'This model has no independently trained artifact.')
    try: return benchmark_model(body.modelId,body.sampleId,body.iterations)
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc

@app.get('/api/history')
def history():
    from backend.services.workspace_store import listing
    return listing('analysis',100)['items']

@app.get('/api/orbit')
def orbit(altitudeKm: float = Query(550,ge=200,le=2000),carrierGhz: float = Query(8.4,gt=0,le=100),elevationDeg: float = Query(78,ge=5,le=90)):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in simulate_orbit_pass(altitudeKm,carrierGhz,elevationDeg).items()}

@app.post('/api/signal/audio')
def audio(body: SignalRequest):
    return Response(generate_audio_waveform(from_rows(body.samples)),media_type='audio/wav')

from backend.api.workspace import router as workspace_router
app.include_router(workspace_router)

# Optional single-origin production hosting after `npm run build`.
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.core.config import ROOT
if os.getenv('SATMOD_SERVE_FRONTEND') == '1':
    dist = ROOT/'frontend/dist'
    if not (dist/'index.html').exists():
        raise RuntimeError('Frontend production build is missing. Run npm run build in frontend/.')
    app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/{frontend_path:path}',include_in_schema=False)
    def frontend(frontend_path: str):
        if frontend_path=='api' or frontend_path.startswith('api/'):
            raise HTTPException(404,'API endpoint does not exist.')
        requested=(dist/frontend_path).resolve()
        if requested.is_relative_to(dist.resolve()) and requested.is_file():
            return FileResponse(requested)
        if frontend_path.startswith(('textures/','maps/')) or frontend_path in ['favicon.ico','robots.txt']:
            raise HTTPException(404,'Static asset does not exist.')
        return FileResponse(dist/'index.html')
