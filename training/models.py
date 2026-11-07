"""Real optional Keras architectures and a reproducible expert-feature RF baseline."""
import numpy as np
from utils.signal_processing import extract_features


def expert_features(frames):
    result=[]
    for frame in frames:
        z=frame[:,0].astype(float)+1j*frame[:,1]
        features=extract_features(frame.astype(float))
        values=[0. if value is None else float(value) for _,value in sorted(features.items())]
        power=np.mean(abs(z)**2)
        # Scale-free complex moments capture phase/symbol structure.
        values += [float(abs(np.mean(z**order))/(power**(order/2)+1e-30)) for order in [2,4,6]]
        result.append(values)
    return np.asarray(result,dtype=float)


def build_keras_model(kind: str, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers=tf.keras.layers
    inputs=layers.Input((128,2));x=inputs
    if kind=='resnet':
        for width in [32,64,96]:
            residual=layers.Conv1D(width,1,padding='same')(x)
            for kernel in [7,5,3]:
                x=layers.Conv1D(width,kernel,padding='same',use_bias=False)(x)
                x=layers.BatchNormalization()(x)
                if kernel!=3:x=layers.Activation('relu')(x)
            x=layers.Activation('relu')(layers.Add()([x,residual]))
            x=layers.MaxPooling1D(2)(x)
        x=layers.GlobalAveragePooling1D()(x)
    elif kind=='cldnn':
        for width in [32,64]:
            x=layers.Conv1D(width,5,padding='same',activation='relu')(x)
            x=layers.BatchNormalization()(x);x=layers.MaxPooling1D(2)(x)
        x=layers.TimeDistributed(layers.Dense(32,activation='relu'))(x)
        x=layers.LSTM(32)(x)
    elif kind=='cnn':
        for width,kernel in [(32,5),(64,5),(96,3)]:
            x=layers.Conv1D(width,kernel,padding='same',activation='relu')(x)
            x=layers.BatchNormalization()(x)
            if width!=96:x=layers.MaxPooling1D(2)(x);x=layers.Dropout(.25)(x)
        x=layers.GlobalAveragePooling1D()(x)
    else:raise ValueError('Choose cnn, resnet or cldnn.')
    x=layers.Dense(64,activation='relu')(x);x=layers.Dropout(.3)(x)
    logits=layers.Dense(3,name='class_logits')(x)
    return tf.keras.Model(inputs,logits,name=f'satmod_{kind}')


def augment_raw(frames, rng, awgn_snr=None, phase_rotation=0., cfo_max=0.):
    z=frames[:,:,0].astype(float)+1j*frames[:,:,1]
    if awgn_snr is not None:
        lo,hi=awgn_snr
        snr=rng.uniform(lo,hi,len(z))
        noise_power=np.mean(abs(z)**2,axis=1)*10**(-snr/10)
        z += np.sqrt(noise_power[:,None]/2)*(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape))
    if phase_rotation:
        z *= np.exp(1j*rng.uniform(-phase_rotation,phase_rotation,len(z)))[:,None]
    if cfo_max:
        z *= np.exp(2j*np.pi*rng.uniform(-cfo_max,cfo_max,len(z))[:,None]*np.arange(z.shape[1]))
    return np.stack((z.real,z.imag),axis=-1).astype(np.float32)
