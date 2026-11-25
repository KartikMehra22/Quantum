"""Optional independent SDK checks; install requirements-qiskit.txt to run."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector
from qiskit_demo.circuit import build_circuit, decode_counts, measurement_probabilities, sample_features
from qml_project.quantum import statevector, measurement_probabilities as reference_probabilities


def test_state_and_all_basis_probabilities_match_original():
    inputs = np.r_[-np.pi, 0, np.pi, np.random.default_rng(17).uniform(-np.pi, np.pi, 30)]
    for x in inputs:
        actual = Statevector.from_instruction(build_circuit(x)).data[[0, 2, 1, 3]]
        np.testing.assert_allclose(actual, statevector(x), atol=1e-13, rtol=0)
    np.testing.assert_allclose(measurement_probabilities(inputs), reference_probabilities(inputs), atol=1e-13, rtol=0)


def test_counts_swap_qubits_without_swapping_physical_labels():
    np.testing.assert_array_equal(decode_counts({"01": 7, "10": 3, "11": 2}), [0, 3, 7, 2])


def test_sampler_reproducibility_totals_and_physical_features():
    counts, features = sample_features(0.53, shots=20000, seed=71)
    repeated, again = sample_features(0.53, shots=20000, seed=71)
    np.testing.assert_array_equal(counts.sum(axis=1), [20000]*3)
    np.testing.assert_array_equal(counts, repeated)
    np.testing.assert_array_equal(features, again)
    assert np.all(np.abs(features) <= 1)
    # Compare against the existing independently sampled-state implementation.
    signs = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    expected = (reference_probabilities([0.53]) @ signs).reshape(9)
    np.testing.assert_allclose(features, expected, atol=0.035, rtol=0)


def test_measurement_circuits_and_invalid_arguments():
    for basis in ("XX", "YY", "ZZ"):
        circuit = build_circuit(0.2, basis, measure=True)
        assert circuit.num_clbits == 2 and circuit.count_ops()["measure"] == 2
    with pytest.raises(ValueError):
        build_circuit(float("nan"))
    with pytest.raises(ValueError):
        build_circuit(0.2, "XY")
    with pytest.raises(ValueError):
        sample_features(0.2, shots=0)
