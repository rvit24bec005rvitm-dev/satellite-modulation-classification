"""Bounded complex STFT density views with explicit sample/window coordinates."""
import hashlib
import numpy as np
from scipy.signal import get_window
from backend.services.signals import validate_iq

MAX_TIME_COLUMNS=128
MAX_FREQUENCY_BINS=256
DISPLAY_FLOOR_DB=-300.
DISPLAY_FLOOR_LINEAR=10**(DISPLAY_FLOOR_DB/10)


def analyze(iq, sample_rate=1, fft_size=64, overlap=.5, window='hann', sample_rate_known=None):
    x=validate_iq(iq)
    if type(fft_size) is not int or not 16<=fft_size<=1024 or fft_size>len(x):
        raise ValueError('FFT size must be an integer from 16 to 1024 and cannot exceed the selected sample count.')
    if isinstance(overlap,bool) or not isinstance(overlap,(int,float)) or not np.isfinite(overlap) or not 0<=overlap<1:
        raise ValueError('Overlap must be a finite fraction from 0 up to, but not including, 1.')
    if window not in ['hann','boxcar']:raise ValueError('Use a periodic Hann or boxcar window.')
    if isinstance(sample_rate,bool) or not isinstance(sample_rate,(int,float)) or not np.isfinite(sample_rate) or not 1<=sample_rate<=1e9:
        raise ValueError('Sample rate must be finite and between 1 and 1e9 Hz.')
    if sample_rate_known is not None and type(sample_rate_known) is not bool:raise ValueError('sampleRateKnown must be boolean when supplied.')
    known=sample_rate!=1 if sample_rate_known is None else sample_rate_known
    if not known and sample_rate!=1:
        raise ValueError('An unknown sample clock must use the normalized Fs=1 convention; a supplied physical clock cannot be silently discarded.')
    fs=float(sample_rate)
    overlap_samples=int(np.floor(float(overlap)*fft_size));hop=fft_size-overlap_samples
    available=1+(len(x)-fft_size)//hop
    selected=np.unique(np.linspace(0,available-1,min(MAX_TIME_COLUMNS,available),dtype=int))
    starts=selected*hop
    # Gather only selected complete frames. A nearly-unit overlap never creates
    # an N-by-Nfft matrix or an unbounded full-resolution time-frequency array.
    indices=starts[:,None]+np.arange(fft_size)[None,:]
    z=x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64)
    weights=get_window(window,fft_size,fftbins=True)
    transformed=np.fft.fft(z[indices]*weights[None,:],axis=1)
    density=np.fft.fftshift(np.abs(transformed)**2/(fs*np.sum(weights**2)),axes=1).T
    frequencies=np.fft.fftshift(np.fft.fftfreq(fft_size,d=1/fs))
    mean_density=density.mean(axis=1)
    candidates=np.flatnonzero((mean_density>=np.roll(mean_density,1)) &
                             (mean_density>=np.roll(mean_density,-1)) &
                             ((mean_density>np.roll(mean_density,1)) | (mean_density>np.roll(mean_density,-1))) &
                             (mean_density>DISPLAY_FLOOR_LINEAR))
    strongest=sorted(candidates,key=lambda i:(-mean_density[i],frequencies[i]))[:8]
    peaks=[{'frequency':float(frequencies[i]),'densityDb':float(10*np.log10(max(mean_density[i],DISPLAY_FLOOR_LINEAR)))} for i in strongest]
    # Contiguous frequency bins are averaged in linear density, preserving
    # integrated power when combined with the exported per-bin widths.
    count=min(MAX_FREQUENCY_BINS,fft_size)
    sizes=np.full(count,fft_size//count,dtype=int);sizes[:fft_size%count]+=1
    groups=np.r_[0,np.cumsum(sizes[:-1])]
    plotted=np.add.reduceat(density,groups,axis=0)/sizes[:,None]
    plotted_frequencies=np.add.reduceat(frequencies,groups)/sizes
    display_db=10*np.log10(np.maximum(plotted,DISPLAY_FLOOR_LINEAR))
    unit='Hz' if known else 'cycles/sample'
    return {'reportVersion':'satmod-time-frequency-v1',
            'inputHash':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
            'sampleCount':len(x),'sampleRate':fs,'sampleRateKnown':known,
            'frequencies':plotted_frequencies.tolist(),'times':((starts+fft_size/2)/fs).tolist(),
            'db':display_db.tolist(),'frequencyUnit':unit,'timeUnit':'s' if known else 'samples',
            'densityUnit':'dB re 1 sample-unit²/Hz' if known else 'dB re 1 sample-unit²/(cycles/sample)',
            'parameters':{'fftSize':fft_size,'overlap':float(overlap),'window':window,
                          'overlapSamples':overlap_samples,'effectiveOverlap':overlap_samples/fft_size,
                          'hopSamples':hop,'availableTimeWindows':available,'sampledTimeWindows':len(starts),
                          'sampledWindowStarts':starts.tolist(),'frequencyBinWidths':(sizes*fs/fft_size).tolist(),
                          'outputFrequencyBins':count,'detrend':False,'padding':False,
                          'displayFloorDb':DISPLAY_FLOOR_DB,'timeReference':'Complete-window center: start + FFT size / 2'},
            'note':f'Complex two-sided PSD density: |FFT(window × samples)|² / (Fs × sum(window²)); periodic Hann or boxcar, no mean removal, padding or resampling. '
                   f'{len(starts)} of {available} complete hop-grid windows are shown, evenly selected before computation; unshown times are not analyzed by this view. '
                   f'Plots average contiguous frequency bins in linear power (at most {MAX_FREQUENCY_BINS} bins). Peak diagnostics use full FFT bins of these sampled windows and are not validated RF-carrier detections. '
                   'Time coordinates denote window centers relative to the selected input. Overlapping windows are dependent. '
                   'The -300 dB floor is a display convention, not a measured noise floor; relative sample units are not calibrated watts.',
            'peaks':peaks}
