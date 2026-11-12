"""Bounded, reproducible registry exports for an operator-controlled offline trainer."""
import hashlib
import io
import json
import threading
import zipfile

import numpy as np

from backend.services import captures, workspace_store as store
from backend.services.operations import SplitGuard
from training.data import forbidden_demo_frame_hashes, split_dataset
from training.runner import validate_config
from utils.preprocessing import CLASS_LABELS

EXPORT_SLOTS=threading.BoundedSemaphore(2)
MAX_RETAINED_REPORT_BYTES=16*1024**2


class ExportCapacityReached(ValueError):
    """Concurrent bounded export capacity is exhausted, not a data error."""


def check_retained_holdouts(guard):
    """Retained completed test evidence survives registry deletion/reimport."""
    for job in store.all_records('job'):
        if job.get('kind')!='evaluation' or job.get('state')!='completed':continue
        path=store.file_path(job['id'],'json')
        if not path.is_file():raise ValueError('A retained completed evaluation report is missing; restore its holdout provenance before exporting training data.')
        if path.stat().st_size>MAX_RETAINED_REPORT_BYTES:
            raise ValueError('A retained evaluation exceeds the bounded holdout-review size; review that provenance offline before training export.')
        report=json.loads(path.read_text())
        entries=report.get('dataset',{}).get('entries',[])
        if entries:
            holdouts=[entry for entry in entries if entry.get('split')=='test']
        else:
            # Earlier Phase 2 reports retained capture provenance and actual
            # test predictions before they included the labeled manifest.
            holdouts=[]
            for prediction in report.get('predictions',[]):
                record_id=store.identifier(prediction['captureId'])
                meta=report.get('captures',{}).get(record_id) or report.get('captures',{}).get(prediction['captureId'])
                if meta is None:raise ValueError('A retained legacy evaluation lacks capture holdout provenance; restore it before training export.')
                holdouts.append({'split':'test','sessionId':meta.get('sessionId'),'frameHash':prediction.get('frameHash'),
                                 'sourceHash':meta.get('sourceHash'),'sourceStart':meta.get('sourceOffset',0)+prediction['start'],
                                 'canonicalSourceHash':meta.get('hash'),'start':prediction['start']})
        if not holdouts:raise ValueError('A retained completed evaluation has no identifiable held-out windows; restore its provenance before training export.')
        for entry in holdouts:
            try:guard.check({**entry,'split':'test'})
            except ValueError as exc:
                raise ValueError('Final held-out evaluation data cannot enter a training/validation split: '+str(exc)) from exc


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def deterministic_npz(arrays):
    """NPY members in stable order/timestamps; never pickle or execute metadata."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def export_training_dataset(dataset_id, model='random-forest', seed=42, allow_small=False):
    if not EXPORT_SLOTS.acquire(blocking=False):
        raise ExportCapacityReached('Two training exports are already active; retry when they finish.')
    try:return _export_training_dataset(dataset_id,model,seed,allow_small)
    finally:EXPORT_SLOTS.release()


def _export_training_dataset(dataset_id, model='random-forest', seed=42, allow_small=False):
    if model not in ['random-forest', 'cnn', 'resnet', 'cldnn']:
        raise ValueError('Choose a supported offline training architecture.')
    config = {'model': model, 'seed': seed, 'temperatureScaling': True,
              **({'trees': 100} if model == 'random-forest' else {'epochs': 50, 'batchSize': 64, 'learningRate': .001})}
    validate_config(config)
    with store.LOCK:
        dataset = store.get('dataset', dataset_id)
        entries = dataset.get('entries', [])
        if not 1 <= len(entries) <= 5000:
            raise ValueError('Training exports require 1–5000 explicitly registered windows.')
        if len(entries) < 1000 and not allow_small:
            raise ValueError('Fewer than 1000 frames: explicitly enable the small experimental dataset option to export.')
        guard = SplitGuard()
        for registered in store.all_records('dataset'):
            for entry in registered.get('entries', []):
                guard.add(entry)
        selected_guard=SplitGuard()
        for entry in entries:selected_guard.add(entry)
        check_retained_holdouts(selected_guard)
        signals = np.empty((len(entries), 128, 2), dtype='<f4')
        provenance = [];capture_configurations = {}
        forbidden = forbidden_demo_frame_hashes()
        for index, entry in enumerate(entries):
            if entry.get('label') not in CLASS_LABELS or entry.get('split') not in ['train', 'validation', 'test']:
                raise ValueError('Training export requires explicit canonical labels and preserved train/validation/test assignments.')
            iq, meta = captures.window(entry['captureId'], entry['start'], 128)
            frame_hash = hashlib.sha256(iq.astype('<f4').tobytes()).hexdigest()
            if frame_hash != entry.get('frameHash') or meta['sessionId'] != entry.get('sessionId'):
                raise ValueError('Capture samples/session no longer match the registered dataset provenance.')
            if bytes.fromhex(frame_hash) in forbidden:
                raise ValueError('Supplied demo frames have unknown historical membership and cannot enter an offline training export.')
            if meta['split'] != 'unassigned' and meta['split'] != entry['split']:
                raise ValueError('Capture and window split assignments conflict.')
            signals[index] = iq
            if meta['id'] not in capture_configurations:
                capture_configurations[meta['id']] = {key: meta.get(key) for key in [
                    'id', 'sourceHash', 'hash', 'sourceOffset', 'sampleRate', 'sampleRateKnown',
                    'channelIndex', 'sourceChannels', 'sourceComponents', 'sampleZero', 'scale',
                    'realMode', 'datatype', 'layout', 'iqOrder', 'conversionNote', 'sigmfSampleOffset',
                ]}
                if 'sigmf' in meta:
                    capture_configurations[meta['id']]['sigmfMetadataHash'] = hashlib.sha256(canonical_json(meta['sigmf'])).hexdigest()
            provenance.append({
                **entry, 'captureHash': meta['hash'], 'sampleRate': meta['sampleRate'] if meta['sampleRateKnown'] else None,
                'source': meta['source'], 'license': meta['license'], 'sourceOffset': meta['sourceOffset'],
                'centerFrequency': meta.get('centerFrequency'), 'position': captures.timing(meta, entry['start']),
            })
    labels = np.array([entry['label'] for entry in entries], dtype='<U8')
    groups = np.array([entry['sessionId'] for entry in entries], dtype='<U200')
    splits = np.array([entry['split'] for entry in entries], dtype='<U10')
    known = np.array([entry.get('snrDb') is not None for entry in entries], dtype=bool)
    snrs = np.array([entry.get('snrDb') if entry.get('snrDb') is not None else np.nan for entry in entries], dtype='<f8')
    if not np.isfinite(snrs[known]).all():
        raise ValueError('Known SNR metadata must be finite; unavailable SNR is explicitly masked.')
    if set(labels) != set(CLASS_LABELS):
        raise ValueError('The existing offline trainer requires all three canonical classes.')
    if len(np.unique(groups)) < 9:
        raise ValueError('At least nine independent declared source/session groups are required for offline training.')
    # Invoke the actual trainer's split validation; never repartition a registered
    # final test window or fit normalization while producing an export.
    partitions = split_dataset({'signals': signals,
                                'labels': np.array([CLASS_LABELS.index(label) for label in labels]),
                                'groups': groups, 'splits': splits}, seed)
    arrays = {'signals': signals, 'labels': labels, 'groups': groups, 'splits': splits, 'snrs': snrs, 'snrKnown': known}
    content = {
        'schemaVersion': 'satmod-training-content-v1',
        'datasetId': dataset['id'], 'datasetHash': dataset['hash'], 'registeredAt': dataset['createdAt'],
        'source': dataset['source'], 'license': dataset['license'], 'acquisitionType': dataset['acquisitionType'],
        'classLabels': list(CLASS_LABELS), 'sampleCount': len(entries),
        'splitStrategy': 'Preserved registered train/validation/test assignments; no automatic repartition',
        'holdoutPolicy':'Registered assignments and retained completed evaluation evidence are checked; deleting a registry entry does not make its observed test data fitting data.',
        'splits': {name: {'samples': len(indices), 'groups': len(np.unique(groups[indices])),
                          'indices': indices.tolist()} for name, indices in partitions.items()},
        'preprocessing': {'export': 'Raw I-first float32 windows; explicit capture scaling already applied',
                          'fitNormalizationOnlyOn': 'train', 'fitCalibrationOnlyOn': 'validation',
                          'finalTestFor': 'Frozen evaluation only; never fitting or model selection'},
        'snr': {'knownSamples': int(known.sum()), 'unavailableSamples': int((~known).sum()),
                'representation': 'snrKnown=false plus NaN; unavailable SNR is never replaced with a measured value'},
        'experimentConfig': config, 'windowProvenance': provenance, 'captureConfigurations': capture_configurations,
        'arrayHashes': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()},
        'limitations': ['Source independence and acquisition labels are importer-declared, not automatically proven.',
                        'Export is not a training run, validation of real satellite accuracy, or model promotion.'],
    }
    encoded_content=canonical_json(content)
    if len(encoded_content)>8*1024**2:raise ValueError('Training provenance exceeds its bounded 8 MiB manifest limit.')
    arrays['registryManifest'] = np.frombuffer(encoded_content, dtype=np.uint8)
    artifact = deterministic_npz(arrays)
    filename = dataset['id'] + '-training.npz'
    manifest = {
        **content, 'schemaVersion': 'satmod-training-export-v1',
        'contentManifestHash': hashlib.sha256(encoded_content).hexdigest(),
        'artifact': {'filename': filename, 'bytes': len(artifact), 'sha256': hashlib.sha256(artifact).hexdigest()},
        'allowSmallDataset': allow_small,
        'offlineCommand': f'python scripts/train.py --dataset {filename} --config experiment-config.json --output experiments/runs/NEW_EXPERIMENT'
                          + (' --allow-small-dataset' if allow_small else ''),
    }
    return manifest, artifact
