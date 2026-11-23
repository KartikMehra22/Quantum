"""Deterministic experiment; raw acquisitions, results and provenance are saved."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .models import fit_readouts, mse
from .quantum import exact_features, measurement_probabilities, sample_records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target(task, inputs):
    if task == "sine":
        return np.sin(inputs)
    if task == "mixed_harmonics":
        return np.sin(2 * inputs) + 0.3 * np.cos(3 * inputs)
    raise ValueError(f"Unknown task: {task}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path, output):
    started = time.perf_counter()
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config["qubits"] != 2:
        raise ValueError("This study implements exactly two qubits")
    if any(output.glob("metrics.csv")):
        raise FileExistsError(f"Existing results in {output}; use a new output directory")
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    ntr, nval, ntest = [config[key] for key in ("n_train", "n_validation", "n_test")]
    n = ntr + nval + ntest
    slices = (slice(0, ntr), slice(ntr, ntr + nval), slice(ntr + nval, n))
    rows = []
    predictions = []
    acquired_shots = 0
    measurement_files = []
    for data_seed in config["data_seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([data_seed, 0]))
        inputs = rng.uniform(*config["input_domain"], size=n)
        probabilities = measurement_probabilities(inputs)
        ideal = exact_features(probabilities)
        np.savez_compressed(raw / f"inputs_d{data_seed}.npz", x=inputs,
                            probabilities=probabilities, ideal_features=ideal)
        acquisitions = []
        for shot_seed in config["measurement_seeds"]:
            for shots in config["shots_per_setting"]:
                rng = np.random.default_rng(np.random.SeedSequence([data_seed, shot_seed, shots, 1]))
                counts, features = sample_records(probabilities, shots, rng)
                filename = f"records_d{data_seed}_m{shot_seed}_s{shots}.npz"
                np.savez_compressed(raw / filename, counts=counts, features=features)
                measurement_files.append(filename)
                acquisitions.append((shots, shot_seed, features, filename))
                acquired_shots += n * 3 * shots
        for task_index, task in enumerate(config["tasks"]):
            latent = target(task, inputs)
            for noise_index, noise in enumerate(config["label_noise_stds"]):
                label_rng = np.random.default_rng(np.random.SeedSequence([data_seed, task_index, noise_index, 2]))
                observed = latent.copy()
                observed[:ntr + nval] += label_rng.normal(0, noise, size=ntr + nval)
                labels_name = f"labels_{task}_noise{noise:g}_d{data_seed}.npz"
                np.savez_compressed(raw / labels_name, latent=latent, observed=observed)
                conditions = [(0, -1, ideal, "exact")] + acquisitions
                for shots, shot_seed, features, record_name in conditions:
                    train, validation, test = [features[part] for part in slices]
                    scaler, models = fit_readouts(train, observed[slices[0]], validation,
                                                 observed[slices[1]], config["ridge_lambdas"])
                    scaled_train = scaler.transform(train)
                    augmented = np.column_stack((np.ones(ntr), scaled_train))
                    singular = np.linalg.svd(augmented, compute_uv=False)
                    rank = int(np.linalg.matrix_rank(augmented))
                    condition = float(singular[0] / singular[-1]) if singular[-1] else float("inf")
                    for method, (model, penalty) in models.items():
                        pred = model.predict(scaler.transform(test))
                        row = dict(task=task, label_noise_std=noise, data_seed=data_seed,
                                   measurement_seed=shot_seed, shots=shots,
                                   feature_mode="exact" if shots == 0 else "finite",
                                   method=method, lambda_=penalty, sklearn_alpha=ntr * penalty,
                                   test_mse=mse(latent[slices[2]], pred),
                                   validation_mse=mse(observed[slices[1]], model.predict(scaler.transform(validation))),
                                   train_mse=mse(observed[slices[0]], model.predict(scaled_train)),
                                   coefficient_norm=float(np.linalg.norm(model.coef_)),
                                   augmented_feature_rank=rank, condition_number=condition,
                                   feature_mse=float(np.mean((features - ideal) ** 2)),
                                   train_shots=ntr * 3 * shots,
                                   validation_shots=nval * 3 * shots,
                                   test_shots=ntest * 3 * shots,
                                   records=record_name, labels=labels_name)
                        rows.append(row)
                        # Store the entire test vector, not just a selected pretty curve.
                        predictions.append(dict(task=task, label_noise_std=noise,
                                                data_seed=data_seed, measurement_seed=shot_seed,
                                                shots=shots, method=method,
                                                prediction=pred.tolist()))
    write_csv(output / "metrics.csv", rows)
    (output / "predictions.json").write_text(json.dumps(predictions, separators=(",", ":")))
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pennylane"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    source_dir = Path(__file__).resolve().parent
    metadata = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.perf_counter() - started,
        python=sys.version.split()[0], platform=platform.platform(),
        versions=versions, config_sha256=sha256(config_path),
        source_sha256={p.name: sha256(p) for p in sorted(source_dir.glob("*.py"))},
        metrics_rows=len(rows), raw_measurement_files=len(measurement_files),
        unique_finite_shots=acquired_shots,
        accounting="Each raw record is reused by both methods, tasks, label-noise conditions and ridge candidates. Counts are NOT multiplied by these reuses. Independent settings each cost S shots.",
        test_target="noiseless latent function", exact_reference="diagnostic only; zero physical acquisition cost is NOT a deployable advantage",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hashes = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Saved {len(rows)} fits and {len(measurement_files)} raw acquisitions to {output}")
    print(f"Unique simulated finite-shot executions: {acquired_shots:,}; runtime: {metadata['runtime_seconds']:.2f}s")

