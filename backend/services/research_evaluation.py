"""Real registered-model evaluation, with frozen identity and compatible comparisons."""
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from backend.services import classifier, workspace_store as store
from backend.services.model_registry import REGISTRY, ModelSpec, hardware_info
from utils.signal_processing import shannon_entropy


class ModelUnavailable(RuntimeError):
    """An artifact/runtime is absent; no alternative predictor may be substituted."""


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass
class EvaluationModel:
    spec: ModelSpec
    backend: object
    metadata: dict

    def predict(self, raw):
        if self.spec.id == '1d-cnn':
            return classifier.predict_window(raw)
        begin = perf_counter()
        prepared = self.spec.prepare(raw)
        preprocessing_ms = (perf_counter() - begin) * 1000
        with classifier.LOCK:
            begin = perf_counter()
            probabilities = np.asarray(self.backend.predict(prepared), dtype=float)
            inference_ms = (perf_counter() - begin) * 1000
        labels = self.spec.class_labels
        if (probabilities.shape != (len(labels),) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            raise ValueError('Registered model returned an invalid class probability vector.')
        return {
            'predictedModulation': labels[int(probabilities.argmax())],
            'confidence': float(probabilities.max()), 'probabilities': dict(zip(labels, probabilities.tolist())),
            'entropyBits': shannon_entropy(probabilities), 'processingMs': inference_ms,
            'preprocessingMs': preprocessing_ms, 'modelId': self.spec.id,
            'preprocessing': self.metadata['preprocessingConfiguration'],
            'warnings': ['Experimental checkpoint: results do not establish independent satellite-domain performance.'],
            'trust': {
                'calibration': {'status': 'not-calibrated-for-this-dataset', 'temperature': 1.0},
                'ood': {'status': 'not-validated', 'detected': None},
                'interpretation': 'Raw model scores; no calibration or threshold was fitted on these test labels.',
            },
        }


def resolve_model(model_id='1d-cnn', version=None, labels=None, expected_hash=None):
    """Load only trusted registry entries, freezing their real artifact identity."""
    spec = REGISTRY.get(model_id)
    if spec is None:
        raise ValueError('Requested model is not registered.')
    if spec.lifecycle == 'Deprecated':
        raise ValueError('Deprecated models cannot start a new external evaluation.')
    if version is not None and version != spec.version:
        raise ValueError('Requested model version does not match the registered version.')
    if spec.loader is None or spec.prepare is None or spec.artifact_path is None or not spec.version:
        raise ModelUnavailable('No trained artifact and compatible preprocessing are available for this model.')
    if tuple(spec.input_shape) != (128, 2) or len(set(spec.class_labels)) != len(spec.class_labels):
        raise ValueError('Registered model has an incompatible raw-window or class-mapping contract.')
    if labels is not None and not set(labels).issubset(spec.class_labels):
        raise ValueError('Dataset labels are incompatible with the selected model class mapping.')
    try:
        actual_hash = hashlib.sha256(spec.artifact_path.read_bytes()).hexdigest()
        if spec.expected_artifact_sha256 and actual_hash != spec.expected_artifact_sha256:
            raise ModelUnavailable('Model artifact no longer matches its registered experiment identity.')
        if expected_hash and actual_hash != expected_hash:
            raise ModelUnavailable('Model artifact changed after this evaluation was queued.')
        try:
            backend = spec.loader()
        except Exception as exc:
            raise ModelUnavailable('Artifact or compatible inference runtime unavailable; inspect backend logs.') from exc
        if not callable(getattr(backend,'predict',None)) or not isinstance(getattr(backend,'backend_name',None),str):
            raise ModelUnavailable('The registered inference runtime has an incompatible predictor contract.')
        loaded_hash = getattr(backend, 'artifact_sha256', actual_hash)
        if loaded_hash != actual_hash:
            raise ModelUnavailable('Cached model identity differs from the artifact; restart before evaluating.')
        if model_id == '1d-cnn':
            mean, std, classes = classifier.stats()
            if tuple(classes) != spec.class_labels:
                raise ValueError('Loaded production class ordering differs from its registry metadata.')
            preprocessing = {'mean': mean, 'std': std, 'dtype': 'float32', 'channelOrder': ['I', 'Q'],
                             'windowLength': 128, 'unitPowerApplied': False,
                             'statisticsSource': 'Supplied production training statistics; original corpus unavailable'}
        else:
            preprocessing = {**(spec.preprocessing_metadata or {}), 'windowLength': 128,
                             'dtype': 'float32', 'channelOrder': ['I', 'Q'], 'unitPowerApplied': False}
            if (not isinstance(preprocessing.get('mean'), (int, float))
                    or not isinstance(preprocessing.get('std'), (int, float))
                    or not np.isfinite([preprocessing['mean'], preprocessing['std']]).all()
                    or preprocessing['std'] <= 0):
                raise ValueError('Registered model lacks finite, compatible training normalization.')
    except (ModelUnavailable, ValueError):
        raise
    except Exception as exc:
        import logging
        logging.getLogger('satmod').exception('Registered inference runtime unavailable: %s', model_id)
        raise ModelUnavailable('Artifact or compatible inference runtime unavailable; inspect backend logs.') from exc
    return EvaluationModel(spec, backend, {
        'modelId': spec.id, 'modelName': spec.name, 'modelVersion': spec.version, 'modelHash': actual_hash,
        'modelStatus': spec.lifecycle, 'classOrder': list(spec.class_labels),
        'backend': backend.backend_name, 'preprocessingConfiguration': preprocessing,
        'preprocessingHash': digest_json(preprocessing), 'trainingProvenance': spec.training_metadata,
        'hardware': hardware_info(),
    })


def model_catalog():
    expected = [
        ('1d-cnn', 'Production CNN', 'supplied-cnn-v1'),
        ('rf-experimental', 'Experimental Random Forest', 'controlled-synthetic-rf-v1'),
        ('resnet-synthetic-v1', 'Experimental ResNet', 'resnet-synthetic-v1'),
        ('cldnn-synthetic-v1', 'Experimental CLDNN', 'cldnn-synthetic-v1'),
    ]
    included = {item[0] for item in expected}
    expected.extend((spec.id, spec.name, spec.version) for spec in REGISTRY.values()
                    if spec.id not in included and spec.loader is not None)
    result = []
    for model_id, name, version in expected:
        spec = REGISTRY.get(model_id)
        item = {'id': model_id, 'name': name, 'version': version,
                'classLabels': list(spec.class_labels) if spec else ['BPSK', 'QPSK', '16-QAM'],
                'lifecycle': spec.lifecycle if spec else 'Experimental', 'inputShape': [128, 2]}
        try:
            context = resolve_model(model_id, version)
        except (ValueError, ModelUnavailable) as exc:
            item.update(available=False, status='Unavailable', reason=str(exc))
        else:
            item.update(available=True, status=context.spec.lifecycle,
                        artifactSha256=context.metadata['modelHash'], backend=context.metadata['backend'],
                        preprocessingConfiguration=context.metadata['preprocessingConfiguration'],
                        trainingProvenance=context.metadata['trainingProvenance'])
        result.append(item)
    return result


def evaluation_protocol(dataset, entries, classes):
    windows = [{k: entry.get(k) for k in ['captureId', 'start', 'frameHash', 'label', 'sessionId', 'snrDb']}
               for entry in entries]
    protocol = {'datasetId': dataset['id'], 'datasetHash': dataset['hash'], 'split': 'test',
                'testWindowsHash': digest_json(windows), 'classLabels': sorted(classes)}
    return {**protocol, 'protocolId': digest_json(protocol)}


def summarize_evaluation(report):
    dataset = report.get('dataset', {})
    classes = report.get('classOrder', list(classifier.stats()[2]))
    entries = [entry for entry in dataset.get('entries', []) if entry['split'] == 'test']
    protocol = report.get('evaluationProtocol')
    if protocol is None and dataset.get('id') and dataset.get('hash') and entries:
        protocol = evaluation_protocol(dataset, entries, classes)
    if protocol is None:
        return None  # No compatibility claims for older incomplete evidence.
    timings = [p['processingMs'] for p in report.get('predictions', [])]
    return {
        'jobId': report['jobId'], 'datasetId': protocol['datasetId'], 'datasetHash': protocol['datasetHash'],
        'datasetName': dataset.get('name', 'Registered dataset'), 'datasetSource': dataset.get('source'),
        'acquisitionType': dataset.get('acquisitionType', 'Unspecified; not independently verified'),
        'modelId': report['modelId'], 'modelVersion': report['modelVersion'], 'modelHash': report['modelHash'],
        'modelName': report.get('modelName',report['modelId']),
        'modelStatus': report.get('modelStatus', 'Production' if report['modelId'] == '1d-cnn' else 'Experimental'),
        'createdAt': report['createdAt'], 'completedAt': report.get('completedAt'),
        'classOrder': classes, 'evaluationProtocol': protocol, 'metrics': report['metrics'],
        'preprocessingConfiguration': report.get('preprocessingConfiguration'),
        'latency': report.get('latency', {'inferenceMedianMs': float(np.median(timings)) if timings else None}),
        'hardware': report.get('hardware'),
    }


def completed_evaluations(dataset_id=None, offset=0, limit=50):
    if dataset_id is not None:
        dataset_id = store.identifier(dataset_id)
    summaries = []
    with store.LOCK:
        for job in store.all_records('job'):
            if job.get('kind') != 'evaluation' or job.get('state') != 'completed':
                continue
            summary = job.get('evaluationSummary')
            if summary is None:
                path = store.file_path(job['id'], 'json')
                if path.is_file():
                    summary = summarize_evaluation(json.loads(path.read_text()))
            if summary and (dataset_id is None or summary['datasetId'] == dataset_id):
                summaries.append(summary)
    page = summaries[offset:offset + limit]
    groups = {}
    for summary in page:
        protocol = summary['evaluationProtocol']
        group = groups.setdefault(protocol['protocolId'], {**protocol, 'evaluations': []})
        group['evaluations'].append(summary)
    return {'items': page, 'groups': list(groups.values()), 'total': len(summaries), 'offset': offset, 'limit': limit,
            'note': 'Compare only one exact dataset/hash/test-window protocol. Acquisition independence is importer-declared.'}


def compare_evaluations(job_ids):
    items = []
    for job_id in job_ids:
        with store.LOCK:
            job = store.get('job', job_id)
            if job.get('kind') != 'evaluation' or job.get('state') != 'completed':
                raise ValueError('Comparison requires completed external evaluation jobs.')
            # Legacy reports have no cached summary. Read all retained windows,
            # not the first result page, when computing their latency summary.
            summary = job.get('evaluationSummary') or summarize_evaluation(json.loads(store.file_path(job_id,'json').read_text()))
        if summary is None:
            raise ValueError('Evaluation lacks a reproducible test-window protocol.')
        items.append(summary)
    if len({item['evaluationProtocol']['protocolId'] for item in items}) != 1:
        raise ValueError('Models can be compared only on the identical dataset ID, hash, test windows and compatible classes.')
    return {'evaluationProtocol': items[0]['evaluationProtocol'], 'evaluations': items,
            'note': 'Matched observed results only; no production promotion or independent-domain accuracy claim.'}
