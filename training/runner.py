"""Offline experiments; preprocessing/calibration fit only on train/validation respectively."""
from pathlib import Path
import hashlib,json,platform,sys,time
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
import joblib
import sklearn
from training.data import load_dataset,split_dataset
from training.models import expert_features,augment_raw,build_keras_model
from training.metrics import evaluate_probabilities
from backend.services.uncertainty import fit_temperature
from utils.preprocessing import CLASS_LABELS,prepare_model_input


def validate_config(config: dict) -> None:
    limits={'seed':(0,4294967295),'trees':(1,1000),'epochs':(1,500),'batchSize':(1,1024),'learningRate':(1e-6,1),'phaseRotation':(0,np.pi),'cfoMax':(0,.49)}
    for key,(low,high) in limits.items():
        if key in config:
            value=config[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{key} must lie between {low} and {high}.')
            if key in ['seed','trees','epochs','batchSize'] and not isinstance(value,int):raise ValueError(f'{key} must be an integer.')
    if 'awgnSnr' in config:
        interval=config['awgnSnr']
        if not isinstance(interval,list) or len(interval)!=2 or not np.isfinite(interval).all() or not -30<=interval[0]<=interval[1]<=60:
            raise ValueError('AWGN augmentation requires an ordered [-30,60] dB interval.')
    if 'temperatureScaling' in config and not isinstance(config['temperatureScaling'],bool):raise ValueError('temperatureScaling must be boolean.')


def run_experiment(dataset_path, output, config, allow_small=False):
    allowed={'model','seed','trees','epochs','batchSize','learningRate','awgnSnr','phaseRotation','cfoMax','temperatureScaling'}
    if set(config)-allowed:raise ValueError(f'Unknown configuration keys: {set(config)-allowed}')
    validate_config(config)
    kind=config.get('model','random-forest');seed=int(config.get('seed',42))
    if kind not in ['random-forest','cnn','resnet','cldnn']:raise ValueError('Unsupported training architecture.')
    out=Path(output)
    if out.exists():raise ValueError('Choose a new output directory; experiment checkpoints are never overwritten.')
    data=load_dataset(dataset_path,allow_small);splits=split_dataset(data,seed)
    train=data['signals'][splits['train']]
    mean,std=float(train.astype(float).mean()),float(train.astype(float).std())
    if std<=0:raise ValueError('Training variance is zero.')
    prepare=lambda raw:np.stack([prepare_model_input(frame,mean,std) for frame in raw])
    normalized={name:prepare(data['signals'][indices]) for name,indices in splits.items()}
    labels={name:data['labels'][indices] for name,indices in splits.items()}
    out.mkdir(parents=True)
    np.savez(out/'normalization.npz',mean=mean,std=std,classes=np.array(CLASS_LABELS))
    np.savez(out/'splits.npz',**splits)
    begin=time.perf_counter();history={}
    augmentation={k:config.get(k) for k in ['awgnSnr','phaseRotation','cfoMax']}
    if kind=='random-forest':
        x=normalized['train'];y=labels['train']
        if any(augmentation.values()):
            augmented=augment_raw(train,np.random.default_rng(seed),config.get('awgnSnr'),config.get('phaseRotation',0),config.get('cfoMax',0))
            x=np.concatenate([x,prepare(augmented)]);y=np.r_[y,y]
        model=RandomForestClassifier(n_estimators=int(config.get('trees',100)),class_weight='balanced',random_state=seed,n_jobs=1,min_samples_leaf=2)
        model.fit(expert_features(x),y)
        joblib.dump(model,out/'random-forest.joblib')
        predict=lambda values:np.log(np.clip(model.predict_proba(expert_features(values)),1e-15,1))
        artifact=out/'random-forest.joblib'
    else:
        try:import tensorflow as tf
        except ImportError as exc:raise RuntimeError('Deep training needs optional TensorFlow in a supported Python environment; see requirements-training.txt.') from exc
        tf.keras.utils.set_random_seed(seed)
        try:tf.config.experimental.enable_op_determinism()
        except AttributeError:pass
        model=build_keras_model(kind,seed)
        model.compile(optimizer=tf.keras.optimizers.Adam(float(config.get('learningRate',.001))),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
        batch=int(config.get('batchSize',64))
        class TrainingSequence(tf.keras.utils.Sequence):
            def __init__(self):
                super().__init__();self.epoch=0;self.on_epoch_end()
            def __len__(self):return int(np.ceil(len(train)/batch))
            def __getitem__(self,index):
                ids=self.order[index*batch:(index+1)*batch]
                rng=np.random.default_rng(np.random.SeedSequence([seed,self.epoch,index]))
                augmented=augment_raw(train[ids],rng,config.get('awgnSnr'),config.get('phaseRotation',0),config.get('cfoMax',0))
                return prepare(augmented),labels['train'][ids]
            def on_epoch_end(self):
                self.epoch+=1;self.order=np.random.default_rng(seed+self.epoch).permutation(len(train))
        counts=np.bincount(labels['train'],minlength=3)
        weights={i:len(train)/(3*int(counts[i])) for i in range(3)}
        result=model.fit(TrainingSequence(),validation_data=(normalized['validation'],labels['validation']),epochs=int(config.get('epochs',50)),class_weight=weights,
                         callbacks=[tf.keras.callbacks.ModelCheckpoint(out/'best.keras',save_best_only=True,monitor='val_loss'),tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=3,factor=.5)],verbose=0)
        history={key:[float(v) for v in values] for key,values in result.history.items()}
        model=tf.keras.models.load_model(out/'best.keras',compile=False)
        predict=lambda values:model(values,training=False).numpy()
        artifact=out/'best.keras'
    train_seconds=time.perf_counter()-begin
    validation_logits=predict(normalized['validation'])
    temperature=fit_temperature(validation_logits,labels['validation']) if config.get('temperatureScaling',True) else 1.
    test_logits=predict(normalized['test'])
    raw_probabilities=softmax(test_logits,axis=1);calibrated=softmax(test_logits/temperature,axis=1)
    snrs=data['snrs'][splits['test']]
    report={'status':'experimental', 'productionPromoted':False,'model':kind,'config':config,'dataset':{'id':data['datasetId'],'sha256':data['datasetSha256'],'samples':len(data['signals'])},
            'protocolId':hashlib.sha256((data['datasetSha256']+json.dumps({k:v.tolist() for k,v in splits.items()},sort_keys=True)).encode()).hexdigest(),
            'splitStrategy':'Provided group-disjoint splits' if 'splits' in data else 'Seeded GroupShuffleSplit 60/20/20 by original capture/session',
            'splits':{name:{'samples':len(indices),'groups':len(np.unique(data['groups'][indices])),'indices':indices.tolist()} for name,indices in splits.items()},
            'normalization':{'fitSplit':'train','mean':mean,'std':std,'channelOrder':['I','Q'],'unitPowerApplied':False},
            'calibration':{'fitSplit':'validation','temperature':temperature,'testRaw':evaluate_probabilities(raw_probabilities,labels['test'],snrs,CLASS_LABELS),
                           'testCalibrated':evaluate_probabilities(calibrated,labels['test'],snrs,CLASS_LABELS)},
            'test':evaluate_probabilities(raw_probabilities,labels['test'],snrs,CLASS_LABELS),'trainSeconds':train_seconds,'history':history,
            'featureSchema': {'version':'expert-statistics-v1','description':'Sorted I/Q statistical properties plus scale-free complex moments of order 2,4,6'} if kind=='random-forest' else {'version':'raw-iq-v1','description':'128×2 I-first scalar training z-score'},
            'classLabels':list(CLASS_LABELS),
            'artifact':{'filename':artifact.name,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'bytes':artifact.stat().st_size},
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scikitLearn':sklearn.__version__,'machine':platform.machine(),'os':platform.system()},
            'limitations':['Experimental results apply only to this dataset and protocol. They do not reproduce the supplied 900-observation historical report.','The dataset provider is responsible for meaningful source groups and independent provenance; unknown original training overlap cannot be inferred.']}
    # Same untouched external test frames, each model's own preprocessing.
    from backend.services.classifier import model as production_model,preprocess as production_prepare
    prod=production_model().predict_batch(np.stack([production_prepare(frame) for frame in data['signals'][splits['test']]]))
    report['productionOnSameTest']=evaluate_probabilities(prod,labels['test'],snrs,CLASS_LABELS)
    np.savez_compressed(out/'test-predictions.npz',probabilities=raw_probabilities,calibratedProbabilities=calibrated,productionProbabilities=prod,labels=labels['test'],snrs=snrs,indices=splits['test'])
    # Latencies are measured warm, with feature extraction separately reported.
    example=normalized['test'][:1]
    for _ in range(3):predict(example)
    timings=[]
    for _ in range(30):
        t=time.perf_counter();predict(example);timings.append((time.perf_counter()-t)*1000)
    report['benchmark']={'batchSize':1,'warmup':3,'repetitions':30,'medianMs':float(np.median(timings)),'p95Ms':float(np.percentile(timings,95)),
                         'scope':'RF includes expert feature extraction; Keras uses model forward only. Do not compare across incompatible scopes.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
