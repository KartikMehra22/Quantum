"""Independent scientific audit, beyond reusing the experiment's fit helper.

Uses NumPy least squares / linear solves rather than scikit-learn estimators.
The optional --deep-physics check evaluates every input using PennyLane.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from qml_project.analysis import read_metrics
from qml_project.quantum import OUTCOME_SIGNS, pennylane_probabilities
from check_results import check


KEY_FIELDS = ("task", "label_noise_std", "data_seed", "measurement_seed", "shots", "method")


def key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def unique_index(rows):
    index = {}
    for row in rows:
        identifier = key(row)
        if identifier in index:
            raise ValueError(f"Duplicate model/result key: {identifier}")
        index[identifier] = row
    return index


def expected_keys(config):
    result = set()
    for task, noise, ds, method in product(config["tasks"], config["label_noise_stds"], config["data_seeds"], ["OLS", "Ridge"]):
        result.add((task, noise, ds, -1, 0, method))
        for ms, shots in product(config["measurement_seeds"], config["shots_per_setting"]):
            result.add((task, noise, ds, ms, shots, method))
    return result


def check_float_values(actual, expected):
    """Allow tiny ARM/x86 rounding differences without relaxing data integrity.

    Checksums validate saved bytes exactly. Recomputed floating-point arithmetic
    is compared numerically because uniform scaling and trig may differ by ULPs.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Floating-point records require matching shapes and finite values")
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def fit_numpy(train, labels, validation, val_labels, penalties, method):
    """Independent standardization and fit; never calls the training helper."""
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    scale = np.where(scale == 0, 1, scale)
    z, val = (train - mean) / scale, (validation - mean) / scale
    if method == "OLS":
        # Centering preserves the unpenalized-intercept convention even when
        # there are redundant columns or fewer observations than features.
        ymean, zmean = np.mean(labels), np.mean(z, axis=0)
        weights = np.linalg.lstsq(z-zmean, labels-ymean, rcond=None)[0]
        intercept, penalty = ymean-zmean @ weights, 0.0
    elif method == "Ridge":
        ymean = np.mean(labels)
        zmean = np.mean(z, axis=0)
        centered = z - zmean
        fits = []
        for penalty in penalties:
            weights = np.linalg.solve(centered.T @ centered + len(z) * penalty * np.eye(z.shape[1]), centered.T @ (labels - ymean))
            intercept = ymean - zmean @ weights
            loss = float(np.mean((val_labels - (intercept + val @ weights)) ** 2))
            fits.append((loss, penalty, intercept, weights))
        _, penalty, intercept, weights = min(fits, key=lambda r: (r[0], r[1]))
    else:
        raise ValueError(f"Unknown method: {method}")
    return mean, scale, intercept, weights, penalty


def independent_summary(rows, metric):
    """Each dataset gets equal weight irrespective of repeated acquisitions."""
    seeds = sorted({row["data_seed"] for row in rows})
    means = np.array([np.mean([row[metric] for row in rows if row["data_seed"] == seed]) for seed in seeds])
    return float(means.mean()), float(means.std(ddof=1))


def audit(directory, deep_physics=False):
    directory = Path(directory)
    check(directory)
    config = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = read_metrics(directory / "metrics.csv")
    index = unique_index(rows)
    if set(index) != expected_keys(config):
        raise ValueError("Missing, extra, or invalid experimental conditions")
    predictions = unique_index(json.loads((directory / "predictions.json").read_text()))
    if set(predictions) != set(index):
        raise ValueError("Prediction vectors do not match the experiment keys")
    ntr, nv, nt = [config[k] for k in ["n_train", "n_validation", "n_test"]]
    n, boundary = ntr + nv + nt, ntr + nv
    datasets, records = {}, {}
    physics_inputs = 0
    for ds in config["data_seeds"]:
        with np.load(directory / "raw" / f"inputs_d{ds}.npz") as archive:
            x, probs, ideal = [archive[k].copy() for k in ["x", "probabilities", "ideal_features"]]
        expected_x = np.random.default_rng(np.random.SeedSequence([ds, 0])).uniform(*config["input_domain"], size=n)
        check_float_values(x, expected_x)
        if not np.isfinite(probs).all() or np.any(probs < 0) or probs.shape != (n, 3, 4):
            raise ValueError("Invalid Born probabilities")
        np.testing.assert_allclose(probs.sum(axis=-1), 1, atol=1e-14)
        np.testing.assert_allclose(ideal, (probs @ OUTCOME_SIGNS).reshape(n, 9), atol=1e-14)
        if deep_physics:
            np.testing.assert_allclose(probs, pennylane_probabilities(x), atol=1e-13, rtol=1e-13)
            physics_inputs += n
        datasets[ds] = (x, ideal)
        for ms, shots in product(config["measurement_seeds"], config["shots_per_setting"]):
            with np.load(directory / "raw" / f"records_d{ds}_m{ms}_s{shots}.npz") as archive:
                counts, features = archive["counts"].copy(), archive["features"].copy()
            if not np.issubdtype(counts.dtype, np.integer):
                raise ValueError("Counts must have integer dtype")
            rng = np.random.default_rng(np.random.SeedSequence([ds, ms, shots, 1]))
            regenerated = np.array([[rng.multinomial(shots, p) for p in group] for group in probs])
            np.testing.assert_array_equal(counts, regenerated)
            records[ds, ms, shots] = features
    expected_files = {f"inputs_d{ds}.npz" for ds in config["data_seeds"]}
    expected_files |= {f"records_d{ds}_m{ms}_s{s}.npz" for ds, ms, s in records}
    for task_i, task in enumerate(config["tasks"]):
        for noise_i, noise in enumerate(config["label_noise_stds"]):
            for ds in config["data_seeds"]:
                name = f"labels_{task}_noise{noise:g}_d{ds}.npz"
                expected_files.add(name)
                x, _ = datasets[ds]
                latent = np.sin(x) if task == "sine" else np.sin(2 * x) + 0.3 * np.cos(3 * x)
                observed = latent.copy()
                rng = np.random.default_rng(np.random.SeedSequence([ds, task_i, noise_i, 2]))
                observed[:boundary] += rng.normal(0, noise, size=boundary)
                with np.load(directory / "raw" / name) as archive:
                    check_float_values(archive["latent"], latent)
                    check_float_values(archive["observed"], observed)
    if {p.name for p in (directory / "raw").glob("*.npz")} != expected_files:
        raise ValueError("Raw file set has missing or extra files")
    for identifier, row in index.items():
        task, noise, ds, ms, shots, method = identifier
        if row["feature_mode"] != ("exact" if shots == 0 else "finite"):
            raise ValueError("Incorrect feature-mode label")
        features = datasets[ds][1] if shots == 0 else records[ds, ms, shots]
        expected_record = "exact" if shots == 0 else f"records_d{ds}_m{ms}_s{shots}.npz"
        if row["records"] != expected_record or row["labels"] != f"labels_{task}_noise{noise:g}_d{ds}.npz":
            raise ValueError("CSV record/label pointer does not match its condition")
        with np.load(directory / "raw" / row["labels"]) as archive:
            observed, latent = archive["observed"].copy(), archive["latent"].copy()
        mean, scale, intercept, weights, penalty = fit_numpy(features[:ntr], observed[:ntr], features[ntr:boundary], observed[ntr:boundary], config["ridge_lambdas"], method)
        if row["lambda_"] != penalty:
            raise ValueError(f"Independent validation selection differs: {identifier}")
        pred = intercept + ((features - mean) / scale) @ weights
        design=np.column_stack((np.ones(ntr),(features[:ntr]-mean)/scale))
        spectrum=np.linalg.svd(design,compute_uv=False)
        tolerance=spectrum[0]*max(design.shape)*np.finfo(float).eps
        if row["augmented_feature_rank"] != int(np.sum(spectrum>tolerance)):
            raise ValueError("Recorded rank differs from independent calculation")
        np.testing.assert_allclose(row["condition_number"],spectrum[0]/spectrum[-1],rtol=1e-10)
        recorded_pred = np.asarray(predictions[identifier]["prediction"])
        if recorded_pred.shape != (nt,) or not np.isfinite(recorded_pred).all():
            raise ValueError("Invalid saved test vector")
        np.testing.assert_allclose(recorded_pred, pred[boundary:], atol=1e-10, rtol=1e-9)
        measured = dict(test_mse=np.mean((pred[boundary:] - latent[boundary:]) ** 2),
                        train_mse=np.mean((pred[:ntr] - observed[:ntr]) ** 2),
                        validation_mse=np.mean((pred[ntr:boundary] - observed[ntr:boundary]) ** 2),
                        coefficient_norm=np.linalg.norm(weights), sklearn_alpha=ntr * penalty,
                        feature_mse=np.mean((features - datasets[ds][1]) ** 2))
        for field, value in measured.items():
            np.testing.assert_allclose(row[field], value, atol=1e-11, rtol=1e-8, err_msg=f"{field}: {identifier}")
        for split, count in [("train", ntr), ("validation", nv), ("test", nt)]:
            if row[split + "_shots"] != 3 * shots * count:
                raise ValueError("Wrong split measurement budget")
    summary = json.loads((directory / "summary.json").read_text())
    expected_groups = {(r["task"], r["label_noise_std"], r["shots"], r["method"]) for r in rows}
    groups = [(r["task"], r["label_noise_std"], r["shots"], r["method"]) for r in summary["summary"]]
    if len(groups) != len(set(groups)) or set(groups) != expected_groups:
        raise ValueError("Invalid summary condition keys")
    for group, entry in zip(groups, summary["summary"]):
        subset = [r for r in rows if (r["task"], r["label_noise_std"], r["shots"], r["method"]) == group]
        expected_repeats = len(config["measurement_seeds"]) if entry["shots"] else 1
        if entry["n_data_seeds"] != len(config["data_seeds"]) or entry["acquisition_repeats_per_seed"] != expected_repeats:
            raise ValueError("Summary repetition counts are incorrect")
        for metric in ["test_mse", "coefficient_norm", "feature_mse", "lambda_"]:
            avg, sd = independent_summary(subset, metric)
            np.testing.assert_allclose([entry[metric+"_mean"], entry[metric+"_sd"]], [avg, sd], atol=1e-14)
    paired_groups = [(r["task"],r["label_noise_std"],r["shots"]) for r in summary["paired_comparisons"]]
    expected_paired = {(r["task"],r["label_noise_std"],r["shots"]) for r in rows}
    if len(paired_groups) != len(set(paired_groups)) or set(paired_groups) != expected_paired:
        raise ValueError("Invalid paired-comparison keys")
    for group, entry in zip(paired_groups, summary["paired_comparisons"]):
        task, noise, shots = group
        means, ols_values = [], []
        for ds in config["data_seeds"]:
            differences=[]
            for ms in config["measurement_seeds"] if shots else [-1]:
                ordinary=index[task,noise,ds,ms,shots,"OLS"]["test_mse"]
                ridge=index[task,noise,ds,ms,shots,"Ridge"]["test_mse"]
                differences.append(ordinary-ridge)
                ols_values.append(ordinary)
            means.append(np.mean(differences))
        for field, expected in [("ols_minus_ridge_mse",np.mean(means)), ("paired_delta_sd",np.std(means,ddof=1)),
                                ("relative_reduction_percent",100*np.mean(means)/np.mean(ols_values)),
                                ("data_seeds_favouring_ridge",sum(v>0 for v in means)), ("n_data_seeds",len(means))]:
            np.testing.assert_allclose(entry[field],expected,atol=1e-12,rtol=1e-10)
    expected_budget = len(config["data_seeds"]) * len(config["measurement_seeds"]) * n * 3 * sum(config["shots_per_setting"])
    if metadata["unique_finite_shots"] != expected_budget:
        raise ValueError("Global budget does not equal unique acquisition cost")
    outcome = dict(status="passed", independent_numpy_fits=len(index), prediction_vectors_checked=len(predictions),
                   summary_groups_checked=len(groups), raw_seed_reconstructions=len(records),
                   paired_comparisons_checked=len(paired_groups),
                   physics_inputs_checked_with_pennylane=physics_inputs,
                   unique_simulated_shots=expected_budget,
                   warning="Numerical checks do not establish general scientific validity, significance, or paper reproduction.")
    (directory / "deep_audit.json").write_text(json.dumps(outcome, indent=2)+"\n")
    print(json.dumps(outcome, indent=2))
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/semester")
    parser.add_argument("--deep-physics", action="store_true")
    args = parser.parse_args()
    audit(args.results, args.deep_physics)
