"""Computed diagnostics, with assumptions kept separate from ground truth."""
import numpy as np
from scipy import signal


def diagnostics(iq, fs, modulation=None, full_scale=None):
    x=np.asarray(iq,dtype=np.float64);z=x[:,0]+1j*x[:,1]
    power=np.abs(z)**2;std=x.std(axis=0);dc=x.mean(axis=0)
    frequencies,psd=signal.welch(z,fs=fs,nperseg=min(256,len(z)),return_onesided=False,scaling='density')
    order=np.argsort(frequencies)
    cfo=None
    if modulation in ['BPSK','QPSK'] and np.all(np.abs(z)>0):
        m=2 if modulation=='BPSK' else 4
        phase=(z/np.abs(z))**m
        coherence=np.mean(phase[1:]*phase[:-1].conj())
        if abs(coherence)>1e-8:cfo={'estimate':float(np.angle(coherence)*fs/(2*np.pi*m)), 'estimateHz':float(np.angle(coherence)*fs/(2*np.pi*m)) if fs!=1 else None,'unit':'Hz' if fs!=1 else 'cycles/sample', 'coherence':float(abs(coherence)),
            'ambiguityIntervalHz':[-fs/(2*m),fs/(2*m)],'method':f'{m}th-power adjacent phase increment',
            'assumptions':'Known M-PSK, adequate SNR, negligible symbol/pulse/fading bias. Diagnostic estimate; not carrier recovery or measured CFO.'}
    width=max(1,int(np.ceil(len(x)/256)))
    trajectory=[{'sample':i,'seconds':i/fs,'power':float(power[i:i+width].mean()),'envelope':float(np.sqrt(power[i:i+width]).mean())} for i in range(0,len(x),width)]
    return {'sampleRate':fs,'dcOffset':{'i':float(dc[0]),'q':float(dc[1])},
        'iqGainImbalanceDb':float(20*np.log10(std[0]/std[1])) if np.all(std>0) else None,
        'iqCorrelation':float(np.mean((x[:,0]-dc[0])*(x[:,1]-dc[1]))/np.prod(std)) if np.all(std>0) else None,
        'imbalanceNote':'Distribution indicators; modulation and channel statistics can also produce asymmetry.',
        'clippedFraction':float(np.mean(np.any(np.abs(x)>=full_scale,axis=1))) if full_scale else None,
        'clippingNote':'Requires a user-supplied calibrated ADC full-scale threshold; otherwise unavailable.',
        'powerCoefficientOfVariation':float(power.std()/power.mean()) if power.mean()>0 else None,
        'frequencyOffset':cfo,'welch':[{'frequency':float(frequencies[i]),'density':float(psd[i])} for i in order],
        'frequencyUnit':'Hz' if fs!=1 else 'cycles/sample','welchUnits':'sample units squared / Hz' if fs!=1 else 'sample units squared / (cycles/sample)','powerOverTime':trajectory}


def activities(points, threshold, minimum_bins=1):
    segments=[];start=None
    for i,p in enumerate(points+[{'power':-1}]):
        if p['power']>threshold and start is None:start=i
        if p['power']<=threshold and start is not None:
            if i-start>=minimum_bins:segments.append({'startSample':points[start]['sample'],'endSample':points[i]['sample'] if i<len(points) else None,'bins':i-start})
            start=None
    return segments
