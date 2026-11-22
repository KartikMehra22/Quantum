import numpy as np
import pytest
from qml_project.quantum import (
    statevector, measurement_probabilities, exact_features, sample_records,
    pennylane_probabilities, OUTCOME_SIGNS,
)
from qml_project.models import fit_readouts


def test_born_probabilities_and_library_agree():
    inputs = np.array([-np.pi, -1.4, 0, 0.83, np.pi])
    for x in inputs:
        assert np.isclose(np.linalg.norm(statevector(x)), 1)
    actual = measurement_probabilities(inputs)
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(actual, pennylane_probabilities(inputs), atol=1e-13)


def test_joint_records_have_consistent_single_and_pair_expectations():
    # A correlated |00>/<11| distribution: singles fluctuate, pair is always +1.
    probs = np.tile([0.5, 0, 0, 0.5], (4000, 3, 1))
    counts, features = sample_records(probs, 32, np.random.default_rng(7))
    np.testing.assert_array_equal(counts.sum(axis=-1), 32)
    np.testing.assert_array_equal(features[:, 2::3], 1)
    np.testing.assert_array_equal(features[:, 0], features[:, 1])
    assert abs(np.var(features[:, 0]) - 1 / 32) < 0.003


def test_measurements_match_exact_expectations_within_sampling_error():
    probabilities = measurement_probabilities(np.array([-1.2, 0.3, 2.1]))
    counts, features = sample_records(probabilities, 100000, np.random.default_rng(42))
    np.testing.assert_allclose(features, exact_features(probabilities), atol=0.012)
    np.testing.assert_allclose((counts @ OUTCOME_SIGNS / 100000).reshape(-1, 9), features)


def test_training_only_scaling_and_ridge_objective():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(30, 4))
    validation = rng.normal(size=(10, 4)) + 8
    labels = rng.normal(size=30)
    val_labels = rng.normal(size=10)
    scaler, models = fit_readouts(train, labels, validation, val_labels, [0.1])
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
    z = scaler.transform(train)
    expected = np.linalg.solve(z.T @ z + 30 * 0.1 * np.eye(4), z.T @ (labels - labels.mean()))
    np.testing.assert_allclose(models["Ridge"][0].coef_, expected, atol=1e-12)
    assert models["Ridge"][0].intercept_ == pytest.approx(labels.mean())


def test_reproducible_records():
    probabilities = measurement_probabilities(np.array([0.2, 1.0]))
    a = sample_records(probabilities, 32, np.random.default_rng(42))
    b = sample_records(probabilities, 32, np.random.default_rng(42))
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(ValueError):
        sample_records(probabilities, 0, np.random.default_rng())


def test_exact_features_against_direct_pauli_expectations():
    # Independent of the measurement-basis rotation and outcome decoder.
    identity = np.eye(2)
    paulis = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1,-1])]
    inputs = np.random.default_rng(84).uniform(-np.pi, np.pi, 101)
    actual = exact_features(measurement_probabilities(inputs))
    expected = []
    for x in inputs:
        state = statevector(x)
        expected.append([np.vdot(state, op @ state).real for p in paulis
                         for op in (np.kron(p,identity),np.kron(identity,p),np.kron(p,p))])
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_ridge_selection_minimizes_only_supplied_validation_mse():
    from sklearn.linear_model import Ridge
    from qml_project.models import mse
    rng=np.random.default_rng(98)
    train=rng.normal(size=(23,5)); labels=rng.normal(size=23)
    validation=rng.normal(size=(17,5)); val_labels=rng.normal(size=17)
    penalties=[1e-6,0.01,1,10]
    scaler,models=fit_readouts(train,labels,validation,val_labels,penalties)
    losses=[]
    for penalty in penalties:
        candidate=Ridge(alpha=23*penalty,solver="svd").fit(scaler.transform(train),labels)
        losses.append((mse(val_labels,candidate.predict(scaler.transform(validation))),penalty))
    assert models["Ridge"][1]==min(losses)[1]
    # Zero labels give exact ties; smaller lambda wins regardless of list order.
    _,tied=fit_readouts(train,np.zeros(23),validation,np.zeros(17),list(reversed(penalties)))
    assert tied["Ridge"][1]==min(penalties)


def test_ols_against_augmented_least_squares_solution():
    rng=np.random.default_rng(91)
    train=rng.normal(size=(25,4)); labels=1.7+rng.normal(size=25)
    validation=rng.normal(size=(12,4)); val_labels=rng.normal(size=12)
    scaler,models=fit_readouts(train,labels,validation,val_labels,[0.01])
    design=np.column_stack([np.ones(25),scaler.transform(train)])
    expected=np.linalg.lstsq(design,labels,rcond=None)[0]
    ols=models["OLS"][0]
    np.testing.assert_allclose(np.r_[ols.intercept_,ols.coef_],expected,atol=1e-12)
