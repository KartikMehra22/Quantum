"""Explicit two-qubit statevector and joint-outcome measurement simulator.

Basis convention: |q0 q1> = |00>, |01>, |10>, |11> (q0 most significant).
Numpy supplies the main experiment. An independently expressed PennyLane
circuit checks its probabilities in tests; PennyLane is not needed to run it.
"""

import numpy as np

I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDG = np.diag([1, -1j])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
CZ = np.diag([1, 1, 1, -1])
OUTCOME_SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
FEATURE_NAMES = ("X0", "X1", "X0X1", "Y0", "Y1", "Y0Y1", "Z0", "Z1", "Z0Z1")


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def statevector(x):
    """All numeric constants are fixed, never fitted to training or test data."""
    state = np.array([1, 0, 0, 0], complex)
    for gate in (
        np.kron(ry(x), I), np.kron(rz(0.4 * x), I),
        np.kron(I, ry(2 * x + 0.3)), np.kron(I, rz(-0.7 * x)),
        CNOT, np.kron(ry(0.7), I), np.kron(I, ry(-0.4)),
        np.kron(I, rz(0.2)), CZ,
    ):
        state = gate @ state
    return state


def measurement_probabilities(inputs):
    """Shape (inputs, 3 settings, 4 joint outcomes), in XX, YY, ZZ order."""
    rotations = (H, H @ SDG, I)
    output = np.empty((len(inputs), 3, 4))
    for i, x in enumerate(inputs):
        state = statevector(float(x))
        for g, basis in enumerate(rotations):
            probs = np.abs(np.kron(basis, basis) @ state) ** 2
            output[i, g] = probs / probs.sum()
    return output


def exact_features(probabilities):
    return (np.asarray(probabilities) @ OUTCOME_SIGNS).reshape(-1, 9)


def sample_records(probabilities, shots, rng):
    """Shared outcomes retain correlations of singles and pair within a group.

    Sampling uses the Born-rule joint distribution, not independent Gaussian
    errors per feature. Settings and input acquisitions are independent.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    counts = np.empty(probabilities.shape, dtype=np.int64)
    for i in range(len(probabilities)):
        for g in range(3):
            counts[i, g] = rng.multinomial(shots, probabilities[i, g])
    features = (counts @ OUTCOME_SIGNS / shots).reshape(-1, 9)
    return counts, features


def pennylane_probabilities(inputs):
    """Independent library implementation for physics verification."""
    import pennylane as qml
    device = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(device)
    def circuit(x, setting):
        qml.RY(x, wires=0)
        qml.RZ(0.4 * x, wires=0)
        qml.RY(2 * x + 0.3, wires=1)
        qml.RZ(-0.7 * x, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.7, wires=0)
        qml.RY(-0.4, wires=1)
        qml.RZ(0.2, wires=1)
        qml.CZ(wires=[0, 1])
        for wire in range(2):
            if setting == 1:
                qml.adjoint(qml.S)(wires=wire)
            if setting in (0, 1):
                qml.Hadamard(wires=wire)
        return qml.probs(wires=[0, 1])

    return np.asarray([[circuit(float(x), g) for g in range(3)] for x in inputs])

