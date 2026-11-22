"""Audit every saved acquisition and independently refit every saved result."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from qml_project.analysis import read_metrics
from qml_project.experiment import sha256
from qml_project.models import fit_readouts, mse
from qml_project.quantum import OUTCOME_SIGNS


def check(directory):
    config = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    for file, digest in metadata["source_sha256"].items():
        if sha256(ROOT / "src/qml_project" / file) != digest:
            raise AssertionError(f"Experiment source changed since the recorded run: {file}")
    checksums = json.loads((directory / "checksums.json").read_text())
    for file, digest in checksums.items():
        if sha256(directory / file) != digest:
            raise AssertionError(f"Changed result file: {file}")
    ntr, nval, ntest = [config[k] for k in ["n_train","n_validation","n_test"]]
    boundary = ntr+nval
    for file in (directory / "raw").glob("records_*.npz"):
        shots = int(file.stem.split("_s")[-1])
        record = np.load(file)
        counts, features = record["counts"], record["features"]
        assert counts.shape == (boundary+ntest,3,4)
        assert np.all(counts>=0)
        np.testing.assert_array_equal(counts.sum(axis=-1),shots)
        np.testing.assert_allclose(features,(counts @ OUTCOME_SIGNS / shots).reshape(-1,9))
    rows = read_metrics(directory / "metrics.csv")
    expected = len(config["tasks"])*len(config["label_noise_stds"])*len(config["data_seeds"])*2*(1+len(config["measurement_seeds"])*len(config["shots_per_setting"]))
    assert len(rows)==expected
    pairs=Counter()
    for row in rows:
        pair = (row["task"],row["label_noise_std"],row["data_seed"],row["measurement_seed"],row["shots"])
        pairs[pair]+=1
        base=np.load(directory / "raw" / f"inputs_d{row['data_seed']}.npz")
        features=base["ideal_features"] if row["shots"]==0 else np.load(directory / "raw" / row["records"])["features"]
        labels=np.load(directory / "raw" / row["labels"])
        scaler, models=fit_readouts(features[:ntr],labels["observed"][:ntr],features[ntr:boundary],labels["observed"][ntr:boundary],config["ridge_lambdas"])
        model, penalty=models[row["method"]]
        actual=mse(labels["latent"][boundary:],model.predict(scaler.transform(features[boundary:])))
        np.testing.assert_allclose(actual,row["test_mse"],rtol=1e-12,atol=1e-14)
        assert penalty==row["lambda_"]
    assert set(pairs.values())=={2}
    result=dict(status="passed",fits_reconstructed=len(rows),acquisitions_checked=len(list((directory/"raw").glob("records_*.npz"))),checksums_verified=len(checksums),paired_method_groups=len(pairs))
    (directory / "verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT / "results/semester")
    check(p.parse_args().results)
