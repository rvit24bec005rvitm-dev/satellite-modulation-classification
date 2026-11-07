"""Controlled experimental data, independent source sessions grouped across SNRs.

This is a newly generated educational domain, not the original training/evaluation
set and not a substitute for real captures. No production model is retrained here.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.signal_processing import generate_iq_signal
from utils.preprocessing import CLASS_LABELS

def create_dataset(output,sessions=100,seed=2026):
    if not isinstance(sessions,int) or not 3<=sessions<=10000 or not isinstance(seed,int) or not 0<=seed<=4294967295:
        raise ValueError('Use 3–10000 sessions per class and a uint32 seed.')
    out=Path(output)
    if out.exists():raise ValueError('Dataset already exists; use a new versioned path.')
    signals,labels,snrs,groups=[],[],[],[]
    for class_id,mod in enumerate(CLASS_LABELS):
        for session in range(sessions):
            # Related bit sequences and channel realization stay in one group.
            session_seed=int(np.random.SeedSequence([seed,class_id,session]).generate_state(1)[0])
            for snr in [-12,-6,0,6,12,18]:
                iq=generate_iq_signal(mod,num_samples=128,sps=4,snr_db=snr,rng_seed=session_seed,rrc_enabled=True)*.008584335
                signals.append(iq);labels.append(mod);snrs.append(snr);groups.append(f'{class_id}-{session}')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,signals=np.array(signals),labels=np.array(labels),snrs=np.array(snrs),groups=np.array(groups))
    metadata={'domain':'controlled-synthetic','seed':seed,'sourceGroups':sessions*3,'frames':len(signals),'sps':4,
              'snrs':[-12,-6,0,6,12,18],'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'limitation':'RRC/AWGN educational distribution; does not represent original RadioML capture provenance or real satellite generalization.'}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--sessions-per-class',type=int,default=100);parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args();print(json.dumps(create_dataset(args.output,args.sessions_per_class,args.seed),indent=2))
