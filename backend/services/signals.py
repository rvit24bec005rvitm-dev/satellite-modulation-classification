"""Safe capture ingestion and scientific diagnostics; never execute uploaded code."""
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat, wavfile, whosmat
from utils.signal_processing import compute_psd, extract_features, compute_spectrogram, compute_eye_diagram
from backend.core.config import MAX_SAMPLES, MAX_UPLOAD_BYTES


def validate_iq(value: np.ndarray) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype.kind not in 'fciu':
        raise ValueError('I/Q data must contain real or complex numbers; objects and strings are forbidden.')
    if a.size > MAX_SAMPLES * 2:
        raise ValueError(f'Capture exceeds {MAX_SAMPLES:,} samples.')
    if np.iscomplexobj(a):
        if a.ndim == 2 and 1 in a.shape:
            a = a.reshape(-1)
        if a.ndim != 1:
            raise ValueError('Complex captures must be a single vector.')
        a = np.column_stack((a.real, a.imag))
    elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if a.ndim != 2 or a.shape[1] != 2 or not 16 <= len(a) <= MAX_SAMPLES:
        raise ValueError('Expected (N,2), (2,N), or a complex vector, with 16–65,536 samples.')
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1e6:
        raise ValueError('I/Q values must be finite and between -1,000,000 and 1,000,000.')
    return a.astype(np.float32)


def from_rows(samples) -> np.ndarray:
    return validate_iq(np.array([[p.i, p.q] for p in samples], dtype=np.float64))


def rows(iq: np.ndarray) -> list[dict[str, float]]:
    return [{'i': float(i), 'q': float(q)} for i, q in iq]


def read_npy(stream) -> np.ndarray:
    # Inspect the header before allocation, including members inside ZIP archives.
    from numpy.lib import format
    version = format.read_magic(stream)
    if version not in [(1, 0), (2, 0), (3, 0)]:
        raise ValueError("Unsupported NPY format version")
    reader = format.read_array_header_1_0 if version == (1, 0) else format.read_array_header_2_0
    shape, _, dtype = reader(stream)
    if dtype.hasobject or dtype.kind not in 'fciu' or np.prod(shape, dtype=object) > MAX_SAMPLES*2 or len(shape) > 2:
        raise ValueError('Unsafe NumPy dtype or oversized array dimensions.')
    stream.seek(0)
    return np.load(stream, allow_pickle=False)


def parse_capture(filename: str, payload: bytes) -> tuple[np.ndarray, float]:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError('Empty or oversized capture.')
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(payload)
    fs = 1.0
    if ext == '.npy':
        a = read_npy(buf)
    elif ext == '.npz':
        with zipfile.ZipFile(buf) as archive:
            members = archive.infolist()
            if len(members) > 16 or sum(m.file_size for m in members) > MAX_UPLOAD_BYTES:
                raise ValueError('NumPy archive is too large when decompressed.')
            names = archive.namelist()
            key = next((k for k in ['iq.npy','samples.npy','signal.npy','arr_0.npy'] if k in names), None)
            if key is None:
                if len(names) != 1:
                    raise ValueError('NPZ must contain one capture or a named iq/samples/signal array.')
                key = names[0]
            a = read_npy(io.BytesIO(archive.read(key)))
    elif ext == '.csv':
        text = payload.decode('utf-8-sig')
        lines = text.strip().splitlines()
        if len(lines) > MAX_SAMPLES+1:
            raise ValueError('Too many CSV rows.')
        header = lines[0].strip().lower().replace(' ', '')
        skip = header in ('i,q','in-phase,quadrature')
        a = np.loadtxt(io.StringIO(text), delimiter=',', skiprows=int(skip))
    elif ext == '.mat':
        # Compressed MAT blocks can allocate far more than the uploaded byte count.
        # Bound decompression before SciPy parses array headers.
        import struct, zlib
        if payload[:4] == b'\x89HDF':
            raise ValueError('MAT v7.3/HDF5 is unsupported. Export MAT v5.')
        pos = 128
        endian = '<' if payload[126:128] == b'IM' else '>'
        budget = 0
        while pos + 8 <= len(payload):
            tag, length = struct.unpack(endian+'II', payload[pos:pos+8])
            if length > MAX_UPLOAD_BYTES or pos+8+length > len(payload):
                raise ValueError('Invalid MAT data block.')
            if tag == 15:
                dec = zlib.decompressobj()
                out = dec.decompress(payload[pos+8:pos+8+length], MAX_UPLOAD_BYTES+1-budget)
                budget += len(out)
                if budget > MAX_UPLOAD_BYTES or dec.unconsumed_tail:
                    raise ValueError('Compressed MAT data exceeds the decompression limit.')
            pos += 8+length
        meta = whosmat(io.BytesIO(payload))
        if any(np.prod(shape, dtype=object) > MAX_SAMPLES*2 for _,shape,_ in meta):
            raise ValueError('MAT array exceeds the sample limit.')
        contents = loadmat(io.BytesIO(payload))
        keys = [k for k in contents if not k.startswith('_')]
        key = next((k for k in ['iq','samples','signal'] if k in keys), None)
        if key is None and len(keys) != 1:
            raise ValueError('MAT must contain one capture or a named iq/samples/signal array.')
        a = contents[key or keys[0]]
    elif ext == '.wav':
        fs, a = wavfile.read(buf)
        if a.dtype.kind in 'iu':
            if a.dtype.kind == 'u':
                a = (a.astype(float) - 128) / 128
            else:
                a = a.astype(float) / (2**(8*a.dtype.itemsize-1))
        if a.ndim == 1:
            # Append zero Q only after unsigned PCM's midpoint is removed.
            a = np.column_stack((a, np.zeros_like(a)))
    elif ext == '.json':
        value = json.loads(payload)
        a = value.get('samples') if isinstance(value, dict) else value
        if a and isinstance(a[0], dict):
            a = [[p['i'],p['q']] for p in a]
        a = np.asarray(a)
    else:
        raise ValueError('Use .npy, .npz, .csv, .mat, .wav or .json captures.')
    return validate_iq(a), float(fs)


def analyze(iq: np.ndarray, fs: float = 1.0, sps: int | None = None) -> dict:
    z = iq[:,0].astype(float) + 1j*iq[:,1]
    power = float(np.mean(np.abs(z)**2))
    freq, psd = compute_psd(iq.astype(float), fs)
    mag = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(z)))/len(z)+1e-30)
    density = 10**(psd/10)
    cdf = np.cumsum(density) / max(float(np.sum(density)), 1e-30)
    lo, hi = np.searchsorted(cdf, [.005,.995]).clip(0,len(freq)-1)
    idx = np.unique(np.linspace(0,len(iq)-1,min(len(iq),1024)).astype(int))
    # Spectral bins use contiguous block averages in linear power, avoiding aliasing of peaks.
    bins = np.array_split(np.arange(len(freq)),min(1024,len(freq)))
    spectrum = [{'frequency':float(np.mean(freq[b])), 'psd':float(10*np.log10(np.mean(density[b])+1e-30)),
                 'magnitude':float(np.max(mag[b]))} for b in bins]
    def histogram(values):
        counts, edges = np.histogram(values, bins=32)
        return [{'value':float((edges[i]+edges[i+1])/2), 'count':int(n)} for i,n in enumerate(counts)]
    f,t,stft = compute_spectrogram(iq,fs,nperseg=min(64,len(iq)))
    ti = np.unique(np.linspace(0,len(t)-1,min(64,len(t))).astype(int))
    result = {
        'sampleCount':len(iq), 'sampleRate':fs, 'signalPower':power, 'powerDb':float(10*np.log10(power+1e-30)),
        'rmsAmplitude':float(np.sqrt(power)), 'bandwidthHz':float(freq[hi]-freq[lo]),
        'noisePower':None, 'estimatedSnrDb':None,
        'qualityNote':'Noise and SNR cannot be uniquely inferred from an arbitrary capture without a reference or noise-only region. Power is in relative sample units (not calibrated watts).',
        'features':extract_features(iq.astype(float)),
        'waveform':[{'sample':int(i),'time':float(i/fs),'i':float(iq[i,0]),'q':float(iq[i,1])} for i in idx],
        'constellation':rows(iq[idx]), 'spectrum':spectrum,
        'amplitudeDistribution':histogram(np.abs(z)), 'phaseDistribution':histogram(np.angle(z)),
        'spectrogram':{'frequencies':f.tolist(),'times':t[ti].tolist(),'db':stft[:,ti].tolist()},
        'eye':None,
    }
    if sps:
        time, it, qt = compute_eye_diagram(iq,sps)
        result['eye'] = {'time':time.tolist(),'i':[v.tolist() for v in it],'q':[v.tolist() for v in qt]}
    return result
