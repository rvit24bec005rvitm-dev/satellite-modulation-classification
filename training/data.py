"""External labeled data contract and group/duplicate-safe train/validation/test split."""
from pathlib import Path
import hashlib
from functools import lru_cache
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from utils.preprocessing import CLASS_LABELS

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def forbidden_demo_frame_hashes():
    with np.load(ROOT/'classifier_samples.npz',allow_pickle=False) as archive:
        return {hashlib.sha256(frame.astype(np.float32).tobytes()).digest() for frame in archive['signals']}


def load_dataset(path: str | Path, allow_small=False) -> dict:
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest==hashlib.sha256((ROOT/'classifier_samples.npz').read_bytes()).hexdigest():
        raise ValueError('The supplied demo subset has unknown split provenance and cannot be used for training/calibration.')
    with np.load(path,allow_pickle=False) as archive:
        required={'signals','labels','snrs','groups'}
        if not required.issubset(archive.files): raise ValueError('Dataset requires signals, labels, snrs and original source-capture/session groups.')
        values={key:archive[key].copy() for key in required}
        if 'splits' in archive:values['splits']=archive['splits'].astype(str)
        if 'snrKnown' in archive:values['snrKnown']=archive['snrKnown'].copy()
        if 'registryManifest' in archive:
            import json
            encoded=archive['registryManifest']
            if encoded.dtype!=np.uint8 or encoded.ndim!=1 or len(encoded)>8*1024**2:raise ValueError('Invalid bounded registry provenance manifest.')
            values['registryManifest']=json.loads(encoded.tobytes().decode('utf-8'))
    x,y,snr,groups=values['signals'],values['labels'],values['snrs'],values['groups']
    if x.ndim!=3 or x.shape[1:]!=(128,2) or x.dtype.kind not in 'fiu' or not np.isfinite(x).all() or np.max(abs(x))>1e6:
        raise ValueError('Signals must be finite raw (N,128,2) I-first numeric frames.')
    if groups.dtype.kind not in 'iuUS' or y.dtype.kind not in 'US': raise ValueError('Group IDs must be strings or integers; labels must use the canonical class strings.')
    if any(value.shape!=(len(x),) for value in [y,snr,groups]): raise ValueError('Labels, SNRs and groups must match the frame count.')
    if 'snrKnown' in values:
        known=values['snrKnown']
        if known.dtype.kind!='b' or known.shape!=(len(x),) or not np.isfinite(snr[known]).all() or not np.isnan(snr[~known]).all():
            raise ValueError('snrKnown must mark finite known SNRs and NaN-only unavailable SNRs.')
    elif not np.isfinite(snr).all():raise ValueError('Labels, SNRs and groups must match the frame count; missing SNR requires an explicit snrKnown mask.')
    if 'registryManifest' in values:
        manifest=values['registryManifest']
        if not isinstance(manifest,dict) or manifest.get('schemaVersion')!='satmod-training-content-v1':raise ValueError('Unsupported registry training manifest.')
        for key,expected in manifest.get('arrayHashes',{}).items():
            if key not in values or not isinstance(values[key],np.ndarray) or hashlib.sha256(values[key].tobytes()).hexdigest()!=expected:
                raise ValueError('Registry training arrays no longer match their exported provenance hashes.')
    if len(x)<1000 and not allow_small: raise ValueError('Fewer than 1000 frames: use --allow-small-dataset only for explicitly experimental smoke tests.')
    if any(hashlib.sha256(frame.astype(np.float32).tobytes()).digest() in forbidden_demo_frame_hashes() for frame in x):
        raise ValueError('Dataset contains supplied demo frames with unknown split provenance; repackaging does not make them training data.')
    labels=np.char.replace(y.astype(str),'QAM16','16-QAM')
    if set(labels)!=set(CLASS_LABELS):raise ValueError('All three canonical class labels must be present; labels are not inferred or relabeled.')
    if len(np.unique(groups))<9:raise ValueError('At least nine independent source groups are required.')
    values.update(signals=x.astype(np.float32),labels=np.array([CLASS_LABELS.index(v) for v in labels]),groups=groups.astype(str),
                  datasetSha256=digest,datasetId=path.stem)
    return values


def split_dataset(data: dict, seed=42) -> dict[str,np.ndarray]:
    x,y,groups=data['signals'],data['labels'],data['groups']
    if 'splits' in data:
        if data['splits'].shape != (len(x),): raise ValueError('One split assignment per frame is required.')
        if not set(data['splits']).issubset({'train','validation','test'}):raise ValueError('Unknown split name.')
        result={name:np.flatnonzero(data['splits']==name) for name in ['train','validation','test']}
    else:
        train,rest=next(GroupShuffleSplit(n_splits=1,train_size=.6,random_state=seed).split(x,y,groups))
        val,test=next(GroupShuffleSplit(n_splits=1,train_size=.5,random_state=seed+1).split(x[rest],y[rest],groups[rest]))
        result={'train':train,'validation':rest[val],'test':rest[test]}
    observed_groups,observed_frames=set(),set()
    for name,indices in result.items():
        if not len(indices) or set(y[indices])!={0,1,2}:raise ValueError(f'{name} must contain every class and at least one frame.')
        group_set=set(groups[indices]);frame_set={hashlib.sha256(x[i].tobytes()).digest() for i in indices}
        if group_set & observed_groups:raise ValueError('Source group leakage across splits.')
        if frame_set & observed_frames:raise ValueError('Identical frame leakage across splits, even though group IDs differ.')
        observed_groups.update(group_set);observed_frames.update(frame_set)
    return result
