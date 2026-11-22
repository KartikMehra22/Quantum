import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from deep_audit import unique_index, independent_summary, fit_numpy, expected_keys, check_float_values


def test_duplicate_model_keys_are_rejected_even_when_row_count_is_correct():
    row = dict(task="sine", label_noise_std=0, data_seed=11, measurement_seed=101, shots=32, method="OLS")
    with pytest.raises(ValueError, match="Duplicate"):
        unique_index([row, row.copy()])


def test_acquisition_repetitions_do_not_get_independent_dataset_weight():
    rows = [dict(data_seed=1, test_mse=v) for v in [0, 2, 4]] + [dict(data_seed=2, test_mse=10)]
    avg, sd = independent_summary(rows, "test_mse")
    assert avg == 6  # equal weights for the dataset means 2 and 10
    assert sd == pytest.approx(np.sqrt(32))


def test_independent_solver_recovers_known_linear_relation():
    x = np.array([[0.,1.],[1.,0.],[2.,1.],[3.,-1.],[4.,2.]])
    y = 1.2 + x @ np.array([0.7, -0.3])
    mean, scale, intercept, weights, penalty = fit_numpy(x, y, x, y, [0.01], "OLS")
    np.testing.assert_allclose(intercept + ((x-mean)/scale) @ weights, y, atol=1e-13)
    assert penalty == 0




def test_independent_ols_handles_rank_deficient_features():
    from qml_project.models import fit_readouts
    x=np.array([[0.,1.,2.,1.],[1.,2.,4.,1.],[2.,3.,6.,1.]])
    y=np.array([0.3,0.7,-0.1])
    mean,scale,intercept,weights,_=fit_numpy(x,y,x,y,[0.1],"OLS")
    scaler,models=fit_readouts(x,y,x,y,[0.1])
    np.testing.assert_allclose(weights,models["OLS"][0].coef_,atol=1e-12)
    np.testing.assert_allclose(intercept,models["OLS"][0].intercept_,atol=1e-12)


def test_recomputed_floats_allow_rounding_but_reject_material_changes():
    original=np.array([-2.333762,0.0,0.004537,2.69])
    check_float_values(original, original+4*np.finfo(float).eps)
    with pytest.raises(AssertionError):
        check_float_values(original, original+1e-6)
    with pytest.raises(ValueError):
        check_float_values(np.array([np.nan]),np.array([np.nan]))
