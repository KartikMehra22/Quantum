"""Descriptive aggregation that respects repeated acquisitions within datasets."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import write_csv


def read_metrics(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = ["label_noise_std", "lambda_", "sklearn_alpha", "test_mse", "validation_mse", "train_mse",
               "coefficient_norm", "condition_number", "feature_mse"]
    integer = ["data_seed", "measurement_seed", "shots", "augmented_feature_rank", "train_shots", "validation_shots", "test_shots"]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        for key in integer:
            row[key] = int(row[key])
    return rows


def summarise(result_dir):
    result_dir = Path(result_dir)
    rows = read_metrics(result_dir / "metrics.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["label_noise_std"], row["shots"], row["method"])].append(row)
    summaries = []
    for (task, noise, shots, method), group in sorted(groups.items()):
        by_seed = defaultdict(list)
        for row in group:
            by_seed[row["data_seed"]].append(row)
        entry = dict(task=task, label_noise_std=noise, shots=shots, method=method,
                     n_data_seeds=len(by_seed), acquisition_repeats_per_seed=len(group) // len(by_seed))
        for metric in ("test_mse", "coefficient_norm", "feature_mse", "lambda_"):
            values = [np.mean([r[metric] for r in by_seed[seed]]) for seed in sorted(by_seed)]
            entry[metric + "_mean"] = float(np.mean(values))
            entry[metric + "_sd"] = float(np.std(values, ddof=1))
        summaries.append(entry)
    write_csv(result_dir / "summary.csv", summaries)
    comparisons = []
    for task in sorted({r["task"] for r in rows}):
        for noise in sorted({r["label_noise_std"] for r in rows}):
            for shots in sorted({r["shots"] for r in rows}):
                subset = [r for r in rows if (r["task"], r["label_noise_std"], r["shots"]) == (task, noise, shots)]
                index = {(r["data_seed"], r["measurement_seed"], r["method"]): r for r in subset}
                deltas = defaultdict(list)
                for (ds, ms, method), ols in index.items():
                    if method == "OLS":
                        ridge = index[ds, ms, "Ridge"]
                        deltas[ds].append(ols["test_mse"] - ridge["test_mse"])
                values = [float(np.mean(deltas[seed])) for seed in sorted(deltas)]
                ols_mean = next(r["test_mse_mean"] for r in summaries if (r["task"], r["label_noise_std"], r["shots"], r["method"]) == (task, noise, shots, "OLS"))
                comparisons.append(dict(task=task, label_noise_std=noise, shots=shots,
                                        ols_minus_ridge_mse=float(np.mean(values)),
                                        paired_delta_sd=float(np.std(values, ddof=1)),
                                        relative_reduction_percent=100 * float(np.mean(values)) / ols_mean,
                                        data_seeds_favouring_ridge=sum(v > 0 for v in values),
                                        n_data_seeds=len(values)))
    write_csv(result_dir / "paired_comparisons.csv", comparisons)
    payload = dict(summary=summaries, paired_comparisons=comparisons,
                   uncertainty="Sample SD over dataset-level means after averaging acquisition repetitions. Descriptive, not confidence intervals; acquisitions sharing a dataset are not independent datasets.")
    (result_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload

