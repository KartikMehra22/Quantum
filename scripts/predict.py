"""Predict from one recorded model; no model selection uses the new input."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from measurement_extension.inference import load_readout


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/measurement_extension")
    p.add_argument("--method",choices=["OLS","Ridge","DiagonalCov","FullCov"],default="FullCov")
    p.add_argument("--task",choices=["sine","mixed_harmonics"],default="sine")
    p.add_argument("--shots",type=int,choices=[0,32,128,512],default=32,help="0 selects the exact-feature simulator reference")
    p.add_argument("--noise",type=float,choices=[0.,0.1],default=0.)
    p.add_argument("--data-seed",type=int);p.add_argument("--measurement-seed",type=int)
    p.add_argument("--sampling-seed",type=int,default=9001)
    p.add_argument("--x",type=float,nargs="+",required=True)
    a=p.parse_args()
    model=load_readout(a.results,a.method,a.task,a.shots,a.noise,a.data_seed,a.measurement_seed)
    output=model.predict_inputs(a.x,a.sampling_seed)
    print(json.dumps(dict(method=a.method,task=a.task,shots_per_setting=a.shots,training_data_seed=model.record["data_seed"],
                          selected_lambda=model.record["lambda_"],selected_gamma=model.record["gamma"],
                          inputs=output["inputs"].tolist(),predictions=output["predictions"].tolist(),
                          sampling_seed=a.sampling_seed,simulated_shots=output["simulated_shots"],
                          note="New-input demonstration from a recorded fixed readout, not an aggregate performance evaluation."),indent=2))
