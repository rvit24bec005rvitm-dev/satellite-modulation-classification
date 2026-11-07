"""Run a versioned, group-safe offline experiment. Production artifacts are immutable."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,help='NPZ: raw signals (N,128,2), canonical labels, snrs, original groups; optional splits')
    parser.add_argument('--config',required=True,help='Experiment JSON configuration')
    parser.add_argument('--output',required=True,help='New versioned experiment directory')
    parser.add_argument('--allow-small-dataset',action='store_true',help='Explicitly experimental smoke only; supplied demo data remains prohibited')
    args=parser.parse_args()
    from training.runner import run_experiment
    report=run_experiment(args.dataset,args.output,json.loads(Path(args.config).read_text()),args.allow_small_dataset)
    print(json.dumps({'output':args.output,'model':report['model'],'testAccuracy':report['test']['accuracy'],'macroF1':report['test']['macroF1'],'productionOnSameTest':report['productionOnSameTest']['accuracy'],'productionPromoted':False},indent=2))
if __name__=='__main__':main()
