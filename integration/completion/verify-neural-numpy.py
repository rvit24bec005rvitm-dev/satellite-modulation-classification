"""Verify genuine online NumPy backends against independent frozen TF vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'integration/completion/neural-native-parity.json')
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT / 'integration/completion') or target.exists():
        raise ValueError('Use a new evidence path inside integration/completion.')
    from backend.services.research_evaluation import resolve_model
    golden_path = ROOT / 'integration/completion/neural-tf-golden.npz'
    metadata = json.loads(golden_path.with_suffix('.json').read_text())
    assert sha(golden_path.read_bytes()) == metadata['artifact']['sha256']
    report = {'schemaVersion': 'satmod-native-independent-tf-parity-v1',
              'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
              'reference': str(golden_path.relative_to(ROOT)), 'referenceSha256': metadata['artifact']['sha256'],
              'referenceTensorflow': metadata['tensorflow'], 'classLabels': metadata['classLabels'],
              'datasetSha256': metadata['datasetSha256'], 'probabilityTolerance': {'atol': 2e-6, 'rtol': 2e-5},
              'scope': 'All 360 original frozen synthetic held-out windows; numeric implementation parity, no fitting or independent RF performance claim.',
              'models': {}}
    with np.load(golden_path, allow_pickle=False) as golden:
        for name in ['resnet-synthetic-v1', 'cldnn-synthetic-v1']:
            reference = metadata['models'][name]
            context = resolve_model(name, reference['modelVersion'])
            assert context.metadata['modelHash'] == reference['modelHash']
            assert 'NumPy' in context.metadata['backend']
            predictions = [context.predict(frame)['probabilities'] for frame in golden['signals']]
            actual = np.array([[value[label] for label in metadata['classLabels']] for value in predictions], dtype='<f4')
            expected = golden[reference['probabilitiesArray']]
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
            mismatches = int(np.sum(actual.argmax(1) != expected.argmax(1)))
            assert mismatches == 0
            report['models'][name] = {'modelId': name, 'modelVersion': context.metadata['modelVersion'],
                                     'modelHash': context.metadata['modelHash'], 'backend': context.metadata['backend'],
                                     'preprocessing': context.metadata['preprocessingConfiguration'],
                                     'samples': len(actual), 'classMismatchCount': mismatches,
                                     'maximumAbsoluteProbabilityError': float(np.max(abs(actual - expected))),
                                     'actualProbabilityArraySha256': sha(actual.tobytes()),
                                     'referenceProbabilityArraySha256': sha(expected.tobytes()),
                                     'actualClassIndexSha256': sha(actual.argmax(1).astype('<i8').tobytes())}
    assert 'tensorflow' not in sys.modules
    report.update(passed=True, tensorflowImported=False)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
