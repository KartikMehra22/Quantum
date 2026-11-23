"""Fresh-seed experiment; no change to the preserved original study."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime,timezone
import numpy as np
from qml_project.quantum import measurement_probabilities,exact_features,sample_records
from qml_project.experiment import write_csv,sha256,target
from .models import fit_methods,METHODS


def run_extension(config_path,output,root):
    start=time.perf_counter()
    config_path,output,root=Path(config_path),Path(output),Path(root)
    config=json.loads(config_path.read_text())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use an empty destination; completed and partial runs are preserved")
    raw=output/"raw";raw.mkdir(parents=True)
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    ntr,nv,nt=[config[key] for key in ["n_train","n_validation","n_test"]]
    n=ntr+nv+nt;boundary=ntr+nv
    rows=[];predictions=[];weights=[];intercepts=[];means=[];scales=[]
    unique_shots=0
    for ds in config["data_seeds"]:
        rng=np.random.default_rng(np.random.SeedSequence([ds,0]))
        x=rng.uniform(*config["input_domain"],size=n)
        probabilities=measurement_probabilities(x);ideal=exact_features(probabilities)
        np.savez_compressed(raw/f"inputs_d{ds}.npz",x=x,probabilities=probabilities,ideal_features=ideal)
        conditions=[(0,-1,ideal,None,"exact")]
        for ms in config["measurement_seeds"]:
            for shots in config["shots_per_setting"]:
                rng=np.random.default_rng(np.random.SeedSequence([ds,ms,shots,1]))
                counts,features=sample_records(probabilities,shots,rng)
                name=f"records_d{ds}_m{ms}_s{shots}.npz"
                np.savez_compressed(raw/name,counts=counts,features=features)
                conditions.append((shots,ms,features,counts,name))
                unique_shots+=n*3*shots
        for task_i,task in enumerate(config["tasks"]):
            latent=target(task,x)
            for noise_i,noise in enumerate(config["label_noise_stds"]):
                observed=latent.copy()
                rng=np.random.default_rng(np.random.SeedSequence([ds,task_i,noise_i,2]))
                observed[:boundary]+=rng.normal(0,noise,size=boundary)
                label_file=f"labels_{task}_noise{noise:g}_d{ds}.npz"
                np.savez_compressed(raw/label_file,latent=latent,observed=observed)
                for shots,ms,features,counts,record_file in conditions:
                    scaler,covariance,models=fit_methods(features[:ntr],observed[:ntr],features[ntr:boundary],observed[ntr:boundary],None if counts is None else counts[:ntr],config)
                    z=scaler.transform(features)
                    for method in METHODS:
                        model=models[method];prediction=model.predict(z[boundary:])
                        rows.append(dict(row_id=len(rows),task=task,label_noise_std=noise,data_seed=ds,measurement_seed=ms,
                                         shots=shots,feature_mode="exact" if shots==0 else "finite",method=method,
                                         lambda_=model.lambda_,gamma=model.gamma,tuning_candidates=model.candidates,
                                         test_mse=float(np.mean((prediction-latent[boundary:])**2)),
                                         validation_mse=model.validation_mse,
                                         train_mse=float(np.mean((model.predict(z[:ntr])-observed[:ntr])**2)),
                                         coefficient_norm=float(np.linalg.norm(model.weights)),
                                         covariance_trace=float(np.trace(covariance)),
                                         covariance_offdiagonal_norm=float(np.linalg.norm(covariance-np.diag(np.diag(covariance)))),
                                         feature_mse=float(np.mean((features-ideal)**2)),
                                         train_shots=ntr*3*shots,validation_shots=nv*3*shots,test_shots=nt*3*shots,
                                         records=record_file,labels=label_file))
                        predictions.append(prediction);weights.append(model.weights);intercepts.append(model.intercept)
                        means.append(scaler.mean_);scales.append(scaler.scale_)
        print(f"Completed dataset {ds}: {len(rows)} evaluated readouts",flush=True)
    write_csv(output/"metrics.csv",rows)
    np.savez_compressed(output/"fitted_models.npz",predictions=np.asarray(predictions),weights=np.asarray(weights),
                        intercepts=np.asarray(intercepts),means=np.asarray(means),scales=np.asarray(scales))
    sources=list((root/"src/measurement_extension").glob("*.py"))+[
        root/"src/qml_project/quantum.py",root/"src/qml_project/experiment.py",
        root/"scripts/run_extension.py",root/"scripts/validate_extension.py",root/"docs/EXTENSION_PROTOCOL.md"]
    source_hashes={str(p.relative_to(root)):sha256(p) for p in sorted(sources)}
    for relative in source_hashes:
        destination=output/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,destination)
    versions={name:importlib.metadata.version(name) for name in ["numpy","scipy","scikit-learn","matplotlib","pennylane"]}
    metadata=dict(completed_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-start,
                  initial_protocol_commit="3962994",protocol_document_sha256=sha256(root/"docs/EXTENSION_PROTOCOL.md"),
                  python=sys.version.split()[0],platform=platform.platform(),
                  versions=versions,source_sha256=source_hashes,config_file_sha256=sha256(config_path),
                  evaluated_readouts=len(rows),raw_acquisitions=len(list(raw.glob("records_*.npz"))),
                  unique_simulated_shots=unique_shots,
                  accounting="Shared acquisitions are counted once across methods, tasks, label conditions and validation candidates; the shot ledger is not QPU runtime.",
                  novelty="Adaptation of an existing covariance-regularized readout, not a new invention or full paper reproduction.")
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    hashes={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output/"checksums.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(f"Saved {len(rows)} evaluated readouts, {unique_shots:,} simulated shots; {metadata['runtime_seconds']:.2f} seconds")

