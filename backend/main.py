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
from backend.api.schemas import ClassifyRequest, ClassificationResult, SignalRequest, GenerateRequest, CompareRequest, BenchmarkRequest, HealthStatus
from backend.services import classifier as svc
from backend.services.signals import analyze, parse_capture, from_rows, rows
from backend.services.evaluation import evaluation, snr_results
from backend.services.model_registry import REGISTRY, describe_models
from utils.signal_processing import generate_iq_signal, CONSTELLATION_REFS, simulate_orbit_pass, generate_audio_waveform

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('satmod')
app = FastAPI(title='SatModClass — Satellite Signal Intelligence',version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Content-Type'])


class BodyLimit:
    """Bound both Content-Length and streamed/chunked bodies before parsing."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 65536 if scope['path'].endswith('/upload') else 16*1024*1024
        size, chunks = 0, []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect': return
            size += len(message.get('body',b''))
            if size > limit:
                return await JSONResponse(status_code=413,content={'detail':'Request body exceeds the configured size limit.'})(scope,receive,send)
            chunks.append(message)
            if not message.get('more_body',False): break
        async def bounded_receive():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope,bounded_receive,send)
app.add_middleware(BodyLimit)

@app.exception_handler(ValueError)
async def invalid_input(request: Request, exc: ValueError):
    return JSONResponse(status_code=422,content={'detail':str(exc)})

@app.exception_handler(RequestValidationError)
async def invalid_schema(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,content={'detail':'; '.join(f"{'.'.join(map(str,e['loc']))}: {e['msg']}" for e in exc.errors())})

@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    logger.exception(json.dumps({'event':'request_failed','path':request.url.path,'error':type(exc).__name__}))
    return JSONResponse(status_code=500,content={'detail':'The signal service could not complete this request. Check the backend log.'})

@app.middleware('http')
async def log_request(request: Request, call_next):
    t0 = perf_counter()
    response = await call_next(request)
    logger.info(json.dumps({'event':'request','method':request.method,'path':request.url.path,'status':response.status_code,'elapsedMs':round((perf_counter()-t0)*1000,2)}))
    return response

@app.get('/api/health',response_model=HealthStatus)
def health():
    try:
        backend = svc.model()
        svc.stats()
        return {'connected':True,'modelId':'1d-cnn','backend':backend.backend_name,'version':VERSION}
    except Exception as exc:
        return JSONResponse(status_code=503,content={'connected':False,'modelId':'1d-cnn','error':str(exc),'version':VERSION})

@app.get('/api/system/info')
def system_info():
    mean,std,classes = svc.stats()
    return {'name':'SatModClass','version':VERSION,'classes':classes,'inputShape':[128,2],
            'architecture':'Conv1D(32,5,same,ReLU) → BN → MaxPool → Conv1D(64,5,same,ReLU) → BN → MaxPool → Conv1D(96,3,same,ReLU) → BN → GlobalAveragePool → Dense(64,ReLU) → Dense(3,Softmax)',
            'normalization':{'mean':mean,'std':std,'policy':'Preserve amplitude; training z-score only'},
            'defaultAmplitudeRms':float(std*np.sqrt(2)), 'maxUploadBytes':MAX_UPLOAD_BYTES,'maxSamples':65536,
            'generatorModulations':['BPSK','QPSK','16-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK'],
            'historyPolicy':'In-memory, most recent 100 classifications; clears on restart'}

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

@app.post('/api/signal/generate')
def generate(body: GenerateRequest):
    if abs(body.dopplerHz+body.cfoHz) > body.sampleRate/2:
        raise ValueError('Combined carrier offset must be within the Nyquist interval ±sampleRate/2.')
    mod = {'16-QAM':'QAM16','8-PSK':'8PSK','16-APSK':'16APSK','32-APSK':'32APSK'}.get(body.modulation,body.modulation)
    iq = generate_iq_signal(mod,num_samples=body.numSamples,sps=body.sps,snr_db=body.snrDb,rrc_alpha=body.rrcAlpha,
            doppler_hz=body.dopplerHz/body.sampleRate,cfo_hz=body.cfoHz/body.sampleRate,phase_noise_std=body.phaseNoiseStd,
            rician_k=body.ricianK if body.ricianEnabled else 0,rician_enabled=body.ricianEnabled,
            twta_alpha_a=2.1587 if body.twtaDrive>0 else 0,twta_drive=body.twtaDrive,rng_seed=body.seed,rrc_enabled=body.rrcEnabled)*body.amplitudeRms
    return {'samples':rows(iq),'modulation':body.modulation,'snrDb':body.snrDb,'sampleRate':body.sampleRate,
            'source':f'Synthetic {body.modulation} at {body.snrDb} dB','parameters':body.model_dump(),
            'reference':rows(CONSTELLATION_REFS.get(mod,np.empty((0,2)))*body.amplitudeRms),
            'classifierSupported':body.modulation in svc.stats()[2], 'analysis':analyze(iq,body.sampleRate,body.sps)}

@app.post('/api/signal/analyze')
def analysis(body: SignalRequest):
    return analyze(from_rows(body.samples),body.sampleRate)

@app.post('/api/signal/compare')
def compare(body: CompareRequest):
    a,b = from_rows(body.first.samples),from_rows(body.second.samples)
    aa,bb = analyze(a,body.first.sampleRate),analyze(b,body.second.sampleRate)
    n = min(len(a),len(b))
    return {'first':aa,'second':bb,'powerDifferenceDb':aa['powerDb']-bb['powerDb'],
            'sampleMse':float(np.mean((a[:n].astype(float)-b[:n])**2)) if body.first.sampleRate==body.second.sampleRate else None,
            'note':'MSE compares overlapping, unaligned samples only; it is not a modulation similarity score.'}

@app.get('/api/evaluation')
def evaluation_results(): return evaluation()

@app.get('/api/evaluation/snr')
def snr(): return snr_results()

@app.post('/api/benchmark')
def benchmark(body: BenchmarkRequest):
    spec = REGISTRY.get(body.modelId)
    if spec is None:
        raise HTTPException(404, 'Model is not registered.')
    if spec.loader is None or spec.prepare is None:
        raise HTTPException(409, 'This model has no independently trained artifact.')
    backend = spec.loader()
    x = spec.prepare(svc.dataset()['signals'][body.sampleId])
    with svc.LOCK:
        for _ in range(3): backend.predict(x)
        times = []
        for _ in range(body.iterations):
            t = perf_counter(); backend.predict(x); times.append((perf_counter()-t)*1000)
    return {'modelId':body.modelId,'iterations':body.iterations,'sampleId':body.sampleId,'warmupIterations':3,
            'meanMs':float(np.mean(times)),'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),
            'minMs':min(times),'maxMs':max(times),'throughputFps':float(1000/np.mean(times)),'latenciesMs':times,
            'scope':'Single-frame CPU inference only; excludes preprocessing, upload, and analysis. Host-dependent.'}

@app.get('/api/history')
def history():
    with svc.LOCK: return list(svc.HISTORY)

@app.get('/api/orbit')
def orbit(altitudeKm: float = Query(550,ge=200,le=2000),carrierGhz: float = Query(8.4,gt=0,le=100),elevationDeg: float = Query(78,ge=5,le=90)):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in simulate_orbit_pass(altitudeKm,carrierGhz,elevationDeg).items()}

@app.post('/api/signal/audio')
def audio(body: SignalRequest):
    return Response(generate_audio_waveform(from_rows(body.samples)),media_type='audio/wav')

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
        if frontend_path.startswith('api/'):
            raise HTTPException(404,'API endpoint does not exist.')
        requested=(dist/frontend_path).resolve()
        if requested.is_relative_to(dist.resolve()) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(dist/'index.html')
