"""Demonstrate all four recorded methods; optionally refit the selected case."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from measurement_extension.inference import load_readout
from measurement_extension.models import fit_methods


def demo(directory,task,shots,refit=False):
    directory=Path(directory);config=json.loads((directory/"config.json").read_text())
    methods=config["methods"]
    readouts={m:load_readout(directory,m,task,shots) for m in methods}
    chosen=readouts["FullCov"].record;ds=chosen["data_seed"]
    with np.load(directory/"raw"/f"inputs_d{ds}.npz") as a:inputs=a["x"].copy();ideal=a["ideal_features"].copy()
    if shots:
        with np.load(directory/"raw"/chosen["records"]) as a:features=a["features"].copy();counts=a["counts"].copy()
    else:features,counts=ideal,None
    with np.load(directory/"raw"/chosen["labels"]) as a:latent,observed=a["latent"].copy(),a["observed"].copy()
    ntr,nv=config["n_train"],config["n_validation"];boundary=ntr+nv
    print("Quantum feature regression: four-method demonstration")
    print(f"Task={task}; dataset={ds}; acquisition={chosen['measurement_seed']}; shots/setting={shots}")
    print(f"Training={ntr}; validation={nv}; test={config['n_test']}; circuit fixed, classical readouts trained.")
    if refit:
        scaler,_,models=fit_methods(features[:ntr],observed[:ntr],features[ntr:boundary],observed[ntr:boundary],None if counts is None else counts[:ntr],config)
        for method in methods:
            np.testing.assert_allclose(models[method].predict(scaler.transform(features[boundary:])),readouts[method].predict_features(features[boundary:]),atol=1e-10,rtol=1e-9)
        print("REFIT CHECK PASSED: all four methods reproduce the selected saved case.")
    print(f"{'Method':14s} {'Test MSE':>12s} {'lambda':>12s} {'gamma':>8s}")
    for method,model in readouts.items():
        loss=float(np.mean((model.predict_features(features[boundary:])-latent[boundary:])**2))
        np.testing.assert_allclose(loss,model.record["test_mse"],atol=1e-12)
        print(f"{method:14s} {loss:12.8f} {model.record['lambda_']:12.5g} {model.record['gamma']:8.2g}")
    print("\nFirst five test predictions (fixed order, not selected by prediction quality):")
    print("input       target      OLS         Ridge       DiagonalCov FullCov")
    predictions={m:readouts[m].predict_features(features[boundary:]) for m in methods}
    for i in range(min(5,config["n_test"])):
        print(f"{inputs[boundary+i]:+10.6f} {latent[boundary+i]:+10.6f} "+" ".join(f"{predictions[m][i]:+10.6f}" for m in methods))
    print("\nThis is one recorded case. Overall performance must use the 20-dataset summaries.")
    print("Main finding: FullCov does not consistently improve on Ridge; do not claim universal superiority.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/measurement_extension")
    p.add_argument("--task",choices=["sine","mixed_harmonics"],default="sine")
    p.add_argument("--shots",type=int,choices=[0,32,128,512],default=32)
    p.add_argument("--refit",action="store_true")
    a=p.parse_args();demo(a.results,a.task,a.shots,a.refit)
