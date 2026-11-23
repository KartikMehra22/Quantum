"""Independently verify the measurement-aware experiment and tuning choices."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from qml_project.experiment import sha256
from qml_project.quantum import pennylane_probabilities
from measurement_extension.analysis import read_rows
from deep_audit import check_float_values
from validate_extension import validate_extension


def expanded_covariance(counts):
    """Independent estimator using explicitly expanded outcome records."""
    signs=np.array([[(-1)**((k>>1)&1),(-1)**(k&1),(-1)**(((k>>1)&1)+(k&1))] for k in range(4)],float)
    matrices=[]
    for record in counts:
        covariance=np.zeros((9,9))
        for g in range(3):
            expanded=np.repeat(signs,record[g],axis=0)
            covariance[3*g:3*g+3,3*g:3*g+3]=np.cov(expanded,rowvar=False,ddof=1)/len(expanded)
        matrices.append(covariance)
    return np.mean(matrices,axis=0)


def independent_fit(z,y,penalty):
    """Augmented least squares, distinct from the production normal solve."""
    eigenvalues,eigenvectors=np.linalg.eigh((penalty+penalty.T)/2)
    if eigenvalues.min() < -1e-11:raise ValueError("Indefinite penalty")
    root=np.sqrt(np.maximum(eigenvalues,0))[:,None]*eigenvectors.T
    augmented=np.vstack([np.column_stack([np.ones(len(z)),z])/np.sqrt(len(z)),np.column_stack([np.zeros(z.shape[1]),root])])
    labels=np.r_[y/np.sqrt(len(z)),np.zeros(z.shape[1])]
    solution=np.linalg.lstsq(augmented,labels,rcond=None)[0]
    return solution[1:],solution[0]


def audit(directory,deep_physics=False):
    directory=Path(directory)
    config=json.loads((directory/"config.json").read_text())
    validate_extension(config,json.loads((ROOT/"configs/semester.json").read_text()))
    metadata=json.loads((directory/"metadata.json").read_text())
    for relative,digest in metadata["source_sha256"].items():
        if sha256(ROOT/relative)!=digest or sha256(directory/"source_snapshot"/relative)!=digest:
            raise ValueError(f"Experimental source or snapshot changed: {relative}")
    checksums=json.loads((directory/"checksums.json").read_text())
    for relative,digest in checksums.items():
        if sha256(directory/relative)!=digest:raise ValueError(f"Changed original artifact: {relative}")
    rows=read_rows(directory)
    key=lambda r:(r["task"],r["label_noise_std"],r["data_seed"],r["measurement_seed"],r["shots"],r["method"])
    index={key(r):r for r in rows}
    expected=set()
    for task,noise,ds,method in product(config["tasks"],config["label_noise_stds"],config["data_seeds"],config["methods"]):
        expected.add((task,noise,ds,-1,0,method))
        for ms,s in product(config["measurement_seeds"],config["shots_per_setting"]):expected.add((task,noise,ds,ms,s,method))
    if len(index)!=len(rows) or set(index)!=expected or [r["row_id"] for r in rows]!=list(range(len(rows))):
        raise ValueError("Duplicate, missing or invalid evaluated readout keys")
    with np.load(directory/"fitted_models.npz") as saved:models={k:saved[k].copy() for k in saved.files}
    ntr,nv,nt=[config[k] for k in ["n_train","n_validation","n_test"]];boundary=ntr+nv;n=boundary+nt
    for name,shape in [("predictions",(len(rows),nt)),("weights",(len(rows),9)),("intercepts",(len(rows),)),("means",(len(rows),9)),("scales",(len(rows),9))]:
        if models[name].shape!=shape or not np.isfinite(models[name]).all():raise ValueError(f"Invalid fitted-model array: {name}")
    signs=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    inputs={};features={};covariances={};expected_raw=set();physics_inputs=0
    for ds in config["data_seeds"]:
        name=f"inputs_d{ds}.npz";expected_raw.add(name)
        with np.load(directory/"raw"/name) as a:x,p,ideal=[a[k].copy() for k in ["x","probabilities","ideal_features"]]
        expected_x=np.random.default_rng(np.random.SeedSequence([ds,0])).uniform(*config["input_domain"],size=n)
        check_float_values(x,expected_x)
        if p.shape!=(n,3,4) or np.any(p<0) or not np.isfinite(p).all():raise ValueError("Invalid probabilities")
        np.testing.assert_allclose(p.sum(axis=-1),1,atol=1e-14)
        np.testing.assert_allclose(ideal,(p@signs).reshape(n,9),atol=1e-14)
        if deep_physics:
            np.testing.assert_allclose(p,pennylane_probabilities(x),atol=1e-13,rtol=1e-13);physics_inputs+=n
        inputs[ds]=x;features[ds,-1,0]=ideal;covariances[ds,-1,0]=np.zeros((9,9))
        for ms,s in product(config["measurement_seeds"],config["shots_per_setting"]):
            name=f"records_d{ds}_m{ms}_s{s}.npz";expected_raw.add(name)
            with np.load(directory/"raw"/name) as a:counts,f=a["counts"].copy(),a["features"].copy()
            if counts.shape!=(n,3,4) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):raise ValueError("Invalid counts")
            np.testing.assert_array_equal(counts.sum(axis=-1),s)
            np.testing.assert_allclose(f,(counts@signs/s).reshape(n,9))
            rng=np.random.default_rng(np.random.SeedSequence([ds,ms,s,1]))
            np.testing.assert_array_equal(counts,np.array([[rng.multinomial(s,g) for g in item] for item in p]))
            features[ds,ms,s]=f;covariances[ds,ms,s]=expanded_covariance(counts[:ntr])
    labels={}
    for task_i,task in enumerate(config["tasks"]):
        for noise_i,noise in enumerate(config["label_noise_stds"]):
            for ds in config["data_seeds"]:
                name=f"labels_{task}_noise{noise:g}_d{ds}.npz";expected_raw.add(name)
                latent=np.sin(inputs[ds]) if task=="sine" else np.sin(2*inputs[ds])+0.3*np.cos(3*inputs[ds])
                observed=latent.copy();rng=np.random.default_rng(np.random.SeedSequence([ds,task_i,noise_i,2]))
                observed[:boundary]+=rng.normal(0,noise,size=boundary)
                with np.load(directory/"raw"/name) as a:
                    check_float_values(a["latent"],latent);check_float_values(a["observed"],observed)
                    labels[task,noise,ds]=(a["latent"].copy(),a["observed"].copy())
    if {p.name for p in (directory/"raw").glob("*.npz")}!=expected_raw:raise ValueError("Raw file coverage differs")
    search=config["ridge_search"]
    ridge_grid=np.unique(np.r_[np.logspace(search["log10_min"],search["log10_max"],search["logspace_points"]),config["ridge_lambdas"]])
    conditions=sorted({key(r)[:-1] for r in rows})
    for task,noise,ds,ms,s in conditions:
        f=features[ds,ms,s];latent,y=labels[task,noise,ds]
        mean=f[:ntr].mean(axis=0);scale=f[:ntr].std(axis=0);scale=np.where(scale==0,1,scale)
        z=(f-mean)/scale;cov=covariances[ds,ms,s]/scale[:,None]/scale[None,:]
        for method in config["methods"]:
            row=index[task,noise,ds,ms,s,method];i=row["row_id"]
            if row["feature_mode"]!=("exact" if s==0 else "finite"):raise ValueError("Incorrect mode")
            candidates=[]
            if method=="OLS":options=[(0.,0.)]
            elif method=="Ridge":options=[(float(lam),0.) for lam in ridge_grid]
            else:options=list(product(config["ridge_lambdas"],config["covariance_gammas"]))
            c=np.diag(np.diag(cov)) if method=="DiagonalCov" else cov
            for lam,gamma in options:
                penalty=lam*np.eye(9)+(gamma*c if method in ["DiagonalCov","FullCov"] else 0)
                w,b=independent_fit(z[:ntr],y[:ntr],penalty)
                loss=float(np.mean((b+z[ntr:boundary]@w-y[ntr:boundary])**2))
                candidates.append((loss,gamma,lam,w,b))
            loss,gamma,lam,w,b=min(candidates,key=lambda r:(r[0],r[1],r[2]))
            # Numerically equal gamma penalties at zero covariance can differ
            # at machine precision; with the same computed penalty they tie.
            if row["lambda_"]!=lam or row["gamma"]!=gamma:
                stored=next(r for r in candidates if r[1]==row["gamma"] and r[2]==row["lambda_"])
                if abs(stored[0]-loss)>1e-11:raise ValueError("Validation choice is not an independent minimum")
                loss,gamma,lam,w,b=stored
            if row["tuning_candidates"]!=len(options):raise ValueError("Incorrect tuning count")
            np.testing.assert_allclose(models["means"][i],mean,atol=1e-13)
            np.testing.assert_allclose(models["scales"][i],scale,atol=1e-13)
            np.testing.assert_allclose(models["weights"][i],w,atol=1e-9,rtol=1e-8)
            np.testing.assert_allclose(models["intercepts"][i],b,atol=1e-9,rtol=1e-8)
            predicted=b+z@w
            np.testing.assert_allclose(models["predictions"][i],predicted[boundary:],atol=1e-9,rtol=1e-8)
            checks=dict(test_mse=np.mean((predicted[boundary:]-latent[boundary:])**2),train_mse=np.mean((predicted[:ntr]-y[:ntr])**2),
                        validation_mse=loss,coefficient_norm=np.linalg.norm(w),covariance_trace=np.trace(cov),
                        covariance_offdiagonal_norm=np.linalg.norm(cov-np.diag(np.diag(cov))),feature_mse=np.mean((f-features[ds,-1,0])**2))
            for field,value in checks.items():np.testing.assert_allclose(row[field],value,atol=1e-10,rtol=1e-8,err_msg=field)
            for split,count in [("train",ntr),("validation",nv),("test",nt)]:
                if row[split+"_shots"]!=3*s*count:raise ValueError("Invalid shot ledger")
    summary=json.loads((directory/"summary.json").read_text())
    group_keys={(r["task"],r["label_noise_std"],r["shots"],r["method"]) for r in rows}
    actual_keys=[(r["task"],r["label_noise_std"],r["shots"],r["method"]) for r in summary["summary"]]
    if len(actual_keys)!=len(set(actual_keys)) or set(actual_keys)!=group_keys:raise ValueError("Invalid summary keys")
    for group,entry in zip(actual_keys,summary["summary"]):
        subset=[r for r in rows if (r["task"],r["label_noise_std"],r["shots"],r["method"])==group]
        for metric in ["test_mse","coefficient_norm","gamma","lambda_"]:
            values=[np.mean([r[metric] for r in subset if r["data_seed"]==ds]) for ds in config["data_seeds"]]
            np.testing.assert_allclose([entry[metric+"_mean"],entry[metric+"_sd"]],[np.mean(values),np.std(values,ddof=1)],atol=1e-13)
        if entry["n_data_seeds"]!=len(config["data_seeds"]):raise ValueError("Incorrect independent dataset count")
        np.testing.assert_allclose(entry["gamma_zero_fraction"],np.mean([r["gamma"]==0 for r in subset]),atol=1e-14)
    contrast_keys=[(r["task"],r["label_noise_std"],r["shots"],r["reference"],r["candidate"]) for r in summary["contrasts"]]
    expected_contrasts={(t,n,s,a,b) for t,n,s in {(r["task"],r["label_noise_std"],r["shots"]) for r in rows}
                        for a,b in [("Ridge","FullCov"),("Ridge","DiagonalCov"),("DiagonalCov","FullCov"),("OLS","Ridge")]}
    if len(contrast_keys)!=len(set(contrast_keys)) or set(contrast_keys)!=expected_contrasts:raise ValueError("Invalid paired contrast coverage")
    for entry in summary["contrasts"]:
        task,noise,s=entry["task"],entry["label_noise_std"],entry["shots"]
        diffs=[];refs=[]
        for ds in config["data_seeds"]:
            values=[];base_values=[]
            for ms in config["measurement_seeds"] if s else [-1]:
                base=index[task,noise,ds,ms,s,entry["reference"]]["test_mse"]
                new=index[task,noise,ds,ms,s,entry["candidate"]]["test_mse"]
                values.append(base-new);base_values.append(base)
            diffs.append(np.mean(values));refs.append(np.mean(base_values))
        for field,value in [("reference_minus_candidate_mse",np.mean(diffs)),("paired_difference_sd",np.std(diffs,ddof=1)),
                            ("relative_mse_reduction_percent",100*np.mean(diffs)/np.mean(refs)),("dataset_means_favouring_candidate",sum(v>0 for v in diffs))]:
            np.testing.assert_allclose(entry[field],value,atol=1e-12,rtol=1e-9)
    budget=len(config["data_seeds"])*len(config["measurement_seeds"])*n*3*sum(config["shots_per_setting"])
    if metadata["unique_simulated_shots"]!=budget:raise ValueError("Global budget mismatch")
    outcome=dict(status="passed",independent_model_evaluations=len(rows),validation_candidates_reconstructed=sum(r["tuning_candidates"] for r in rows),
                 expanded_training_covariances_checked=len(covariances)-len(config["data_seeds"]),
                 physics_inputs_checked=physics_inputs,summary_groups_checked=len(actual_keys),paired_contrasts_checked=len(summary["contrasts"]),
                 unique_simulated_shots=budget)
    (directory/"verification.json").write_text(json.dumps(outcome,indent=2)+"\n");print(json.dumps(outcome,indent=2))
    return outcome


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/measurement_extension")
    parser.add_argument("--deep-physics",action="store_true")
    args=parser.parse_args();audit(args.results,args.deep_physics)
