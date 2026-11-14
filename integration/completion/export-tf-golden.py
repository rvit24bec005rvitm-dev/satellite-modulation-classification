"""Independent TensorFlow reference vectors for fixed, unchanged Keras artifacts.

This deliberately never imports the production/experimental inference loaders.
It reads the recorded raw test windows, each checkpoint's own normalization,
and tf.keras.models.load_model directly. No fitting or checkpoint edits occur.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

for key in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def deterministic_npz(arrays):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(arrays[name]), allow_pickle=False)
            entry = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, stream.getvalue())
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'integration/completion/neural-tf-golden.npz')
    args = parser.parse_args()
    target = args.output.resolve()
    metadata_path = target.with_suffix('.json')
    if not target.is_relative_to(ROOT / 'integration/completion'):
        raise ValueError('Golden evidence must remain under integration/completion.')
    if target.exists() or metadata_path.exists():
        raise ValueError('Refusing to overwrite existing golden evidence.')
    import tensorflow as tf
    import keras
    from training.data import load_dataset
    from utils.preprocessing import CLASS_LABELS, prepare_model_input
    source = ROOT / 'experiments/data/controlled-synthetic-v1.npz'
    data = load_dataset(source)
    source_hash = sha(source.read_bytes())
    arrays = {'classLabels': np.array(CLASS_LABELS)}
    metadata = {'schemaVersion': 'satmod-independent-tensorflow-golden-v1',
                'python': platform.python_version(), 'platform': platform.platform(),
                'tensorflow': tf.__version__, 'keras': keras.__version__, 'numpy': np.__version__,
                'datasetSha256': source_hash, 'dataset': 'experiments/data/controlled-synthetic-v1.npz',
                'referencePath': 'Direct tf.keras.models.load_model and inference; never registry/NumPy candidate loaders.',
                'purpose': 'Numerical implementation parity only; no independent RF performance, fitting, selection or promotion claim.',
                'classLabels': list(CLASS_LABELS), 'models': {}}
    first_indices = None
    for name in ['resnet-synthetic-v1', 'cldnn-synthetic-v1']:
        directory = ROOT / 'experiments/runs' / name
        report = json.loads((directory / 'report.json').read_text())
        artifact = directory / report['artifact']['filename']
        model_hash = sha(artifact.read_bytes())
        assert model_hash == report['artifact']['sha256'] and source_hash == report['dataset']['sha256']
        indices = np.array(report['splits']['test']['indices'], dtype='<i8')
        if first_indices is None:
            first_indices = indices
            arrays.update(signals=data['signals'][indices].astype('<f4'), labels=data['labels'][indices].astype('<i8'),
                          sourceIndices=indices, groups=data['groups'][indices], snrs=data['snrs'][indices])
        else:
            np.testing.assert_array_equal(first_indices, indices)
        norm = report['normalization']
        frames = np.stack([prepare_model_input(frame, norm['mean'], norm['std']) for frame in arrays['signals']])
        model = tf.keras.models.load_model(artifact, compile=False, safe_mode=True)
        assert model.input_shape == (None, 128, 2) and model.output_shape == (None, 3)
        logits = model(frames, training=False).numpy().astype('<f4')
        probabilities = softmax(logits, axis=1).astype('<f4')
        with np.load(directory / 'test-predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_allclose(probabilities, saved['probabilities'], atol=1e-6, rtol=1e-7)
            mismatch = int(np.sum(probabilities.argmax(1) != saved['probabilities'].argmax(1)))
            error = float(np.max(abs(probabilities - saved['probabilities'])))
        assert mismatch == 0
        prefix = name.split('-')[0]
        arrays[prefix + '_logits'] = logits
        arrays[prefix + '_probabilities'] = probabilities
        metadata['models'][name] = {'modelId': name, 'modelVersion': name, 'modelHash': model_hash,
                                   'normalization': norm, 'normalizationFileSha256': sha((directory / 'normalization.npz').read_bytes()),
                                   'samples': len(indices), 'logitsArray': prefix + '_logits',
                                   'probabilitiesArray': prefix + '_probabilities',
                                   'originalRecordedMaxProbabilityError': error, 'classMismatchCount': mismatch}
        print(name, 'direct TensorFlow vectors:', len(indices), 'original stored probability error:', error, flush=True)
    metadata['arrays'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': sha(value.tobytes())}
                          for key, value in arrays.items()}
    artifact = deterministic_npz(arrays)
    metadata['artifact'] = {'filename': target.name, 'sha256': sha(artifact), 'bytes': len(artifact)}
    target.write_bytes(artifact)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), 'sha256': metadata['artifact']['sha256']}), flush=True)


if __name__ == '__main__':
    main()
