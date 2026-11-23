import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from measurement_extension.inference import load_readout

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results/measurement_extension"
CONFIG=json.loads((RESULTS/"config.json").read_text())
BOUNDARY=CONFIG["n_train"]+CONFIG["n_validation"]


@pytest.mark.parametrize("method",["OLS","Ridge","DiagonalCov","FullCov"])
def test_saved_readout_reconstructs_recorded_test_predictions(method):
    model=load_readout(RESULTS,method=method)
    with np.load(RESULTS/"raw"/model.record["records"]) as a:features=a["features"][BOUNDARY:]
    with np.load(RESULTS/"fitted_models.npz") as a:expected=a["predictions"][model.record["row_id"]]
    np.testing.assert_allclose(model.predict_features(features),expected,atol=1e-13)
    assert not model.weights.flags.writeable


def test_all_readouts_use_same_fresh_measurements_and_reproduce_by_seed():
    results=[load_readout(RESULTS,method=m).predict_inputs([-.5,.2,.9],123) for m in ["OLS","Ridge","DiagonalCov","FullCov"]]
    for r in results:
        np.testing.assert_array_equal(r["counts"],results[0]["counts"])
        assert r["simulated_shots"]==3*3*32
    repeated=load_readout(RESULTS).predict_inputs([-.5,.2,.9],123)
    np.testing.assert_array_equal(repeated["predictions"],results[-1]["predictions"])


def test_exact_reference_has_no_sampled_counts():
    output=load_readout(RESULTS,shots=0).predict_inputs([0.2])
    assert output["counts"] is None and output["simulated_shots"]==0


@pytest.mark.parametrize("inputs",[[4.0],[float("nan")],[],[[0.1,0.2]]])
def test_unsupported_inference_inputs_fail(inputs):
    with pytest.raises(ValueError):load_readout(RESULTS).predict_inputs(inputs)


def test_unknown_recorded_case_is_not_silently_substituted():
    with pytest.raises(ValueError):load_readout(RESULTS,data_seed=999999)


def test_prediction_cli_and_four_method_refit_demo():
    p=subprocess.run([sys.executable,str(ROOT/"scripts/predict.py"),"--x","0.2","0.5"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    output=json.loads(p.stdout)
    assert len(output["predictions"])==2 and output["simulated_shots"]==192
    p=subprocess.run([sys.executable,str(ROOT/"scripts/defense_demo.py"),"--refit"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert "REFIT CHECK PASSED" in p.stdout and "FullCov" in p.stdout


def test_four_method_walkthrough_code_cells_execute(monkeypatch):
    notebook=json.loads((ROOT/"notebooks/02_measurement_aware_walkthrough.ipynb").read_text())
    monkeypatch.chdir(ROOT)
    namespace={}
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":exec(compile("".join(cell["source"]),"walkthrough-cell","exec"),namespace)
