"""Dataset-level summaries and prespecified paired contrasts for four methods."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from qml_project.experiment import write_csv


def read_rows(directory):
    with (Path(directory)/"metrics.csv").open(newline="") as stream:rows=list(csv.DictReader(stream))
    integers=["row_id","data_seed","measurement_seed","shots","tuning_candidates","train_shots","validation_shots","test_shots"]
    strings={"task","feature_mode","method","records","labels"}
    for row in rows:
        for field in row:
            if field in integers:row[field]=int(row[field])
            elif field not in strings:row[field]=float(row[field])
    return rows


def summarize(directory):
    directory=Path(directory);rows=read_rows(directory)
    groups=defaultdict(list)
    for row in rows:groups[row["task"],row["label_noise_std"],row["shots"],row["method"]].append(row)
    summary=[]
    for (task,noise,shots,method),group in sorted(groups.items()):
        by_seed=defaultdict(list)
        for row in group:by_seed[row["data_seed"]].append(row)
        entry=dict(task=task,label_noise_std=noise,shots=shots,method=method,n_data_seeds=len(by_seed))
        for metric in ["test_mse","coefficient_norm","gamma","lambda_"]:
            values=[np.mean([r[metric] for r in items]) for items in by_seed.values()]
            entry[metric+"_mean"]=float(np.mean(values));entry[metric+"_sd"]=float(np.std(values,ddof=1))
        entry["gamma_zero_fraction"]=float(np.mean([r["gamma"]==0 for r in group]))
        summary.append(entry)
    index={(r["task"],r["label_noise_std"],r["shots"],r["data_seed"],r["measurement_seed"],r["method"]):r for r in rows}
    contrasts=[]
    for task,noise,shots in sorted({(r["task"],r["label_noise_std"],r["shots"]) for r in rows}):
        for reference,candidate in [("Ridge","FullCov"),("Ridge","DiagonalCov"),("DiagonalCov","FullCov"),("OLS","Ridge")]:
            ds_values=[];ds_reference=[]
            for ds in sorted({r["data_seed"] for r in rows}):
                acquired=sorted({r["measurement_seed"] for r in rows if r["data_seed"]==ds and r["shots"]==shots})
                diffs=[];refs=[]
                for ms in acquired:
                    base=index[task,noise,shots,ds,ms,reference]["test_mse"]
                    new=index[task,noise,shots,ds,ms,candidate]["test_mse"]
                    diffs.append(base-new);refs.append(base)
                ds_values.append(float(np.mean(diffs)));ds_reference.append(float(np.mean(refs)))
            contrasts.append(dict(task=task,label_noise_std=noise,shots=shots,reference=reference,candidate=candidate,
                                  reference_minus_candidate_mse=float(np.mean(ds_values)),
                                  paired_difference_sd=float(np.std(ds_values,ddof=1)),
                                  relative_mse_reduction_percent=100*float(np.mean(ds_values))/float(np.mean(ds_reference)),
                                  dataset_means_favouring_candidate=sum(v>0 for v in ds_values),n_data_seeds=len(ds_values)))
    write_csv(directory/"summary.csv",summary);write_csv(directory/"paired_comparisons.csv",contrasts)
    payload=dict(summary=summary,contrasts=contrasts,uncertainty="Descriptive sample SD over dataset means after averaging acquisitions; not confidence intervals or significance tests.")
    (directory/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    return payload

