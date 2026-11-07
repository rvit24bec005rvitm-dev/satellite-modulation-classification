"""Optional independent TensorFlow comparison; never modifies model artifacts."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import sys
import json
import zipfile
import io
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.model_backend import NumPyCNNEngine
from backend.services.classifier import preprocess, dataset


def main():
    import tensorflow as tf
    root=Path(__file__).resolve().parents[1]
    artifact=root/'satellite_modulation_cnn_all_snr.keras'
    # Older BN configs include renorm fields removed in Keras 3. Remove only
    # disabled legacy flags from an in-memory clone for compatibility.
    with zipfile.ZipFile(artifact) as z:
        config=json.loads(z.read('config.json'))
        for layer in config['config']['layers']:
            if layer['class_name']=='BatchNormalization':
                assert not layer['config'].get('renorm',False)
                for key in ['renorm','renorm_clipping','renorm_momentum']:
                    layer['config'].pop(key,None)
            for name in ['kernel_initializer','bias_initializer']:
                if name in layer['config']:
                    for key in ['input_axes','output_axes']:
                        layer['config'][name]['config'].pop(key,None)
            layer['config'].pop('quantization_config',None)
        config.pop('compile_config',None)
        model=tf.keras.models.model_from_json(json.dumps(config))
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.weights.h5'
            path.write_bytes(z.read('model.weights.h5'))
            model.load_weights(path)
    engine=NumPyCNNEngine.from_npz(root/'models/cnn_weights.npz')
    extracted=NumPyCNNEngine.from_keras_zip(artifact)
    for key in engine.w: np.testing.assert_array_equal(engine.w[key],extracted.w[key])
    d=dataset()
    indices=list(range(0,len(d['signals']),7))
    frames=np.array([preprocess(d['signals'][i]) for i in indices])
    expected=model(frames,training=False).numpy()
    actual=np.array([engine.predict(frame) for frame in frames])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    gradient_errors=[]
    feature_model=tf.keras.Model(model.inputs,model.layers[-1].input)
    for target in range(3):
        value=tf.Variable(frames[:3])
        with tf.GradientTape() as tape:
            features=feature_model(value,training=False)
            logits=tf.linalg.matmul(features,model.layers[-1].kernel)+model.layers[-1].bias
            score=tf.reduce_sum(logits[:,target])
        expected_gradient=tape.gradient(score,value).numpy()
        actual_gradient=np.stack([engine.logit_gradient(frame,target) for frame in frames[:3]])
        np.testing.assert_allclose(actual_gradient,expected_gradient,atol=2e-4,rtol=2e-4)
        gradient_errors.append(float(np.max(np.abs(actual_gradient-expected_gradient))))
    report={'frames':len(indices),'maxAbsoluteError':float(np.max(np.abs(actual-expected))),
            'classAgreement':float(np.mean(actual.argmax(1)==expected.argmax(1))),
            'logitGradientMaxAbsoluteError':max(gradient_errors),'gradientClassTargets':3,'tensorflowVersion':tf.__version__,'npzWeightsMatchKerasArchive':True}
    print(json.dumps(report,indent=2))
    (root/'integration/keras-parity.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
