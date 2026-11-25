"""Same fixed feature map, expressed independently using Qiskit.

Qiskit arrays/count strings use |q1 q0>; project records use |q0 q1>.
Conversions below preserve physical qubit labels and explicitly reorder data.
This optional module does not alter the original experiments or saved results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

SETTINGS = ("XX", "YY", "ZZ")
# Rows correspond to project-order outcomes 00, 01, 10, 11.
SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def build_circuit(x, basis="ZZ", measure=False):
    """Return the exact project gate sequence, with optional basis/measurement.

    Default ZZ without measurement is the feature-map state preparation.
    X measurement uses H; Y uses S-dagger followed by H on each qubit.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    if basis not in SETTINGS:
        raise ValueError("basis must be XX, YY or ZZ")
    circuit = QuantumCircuit(2, name=f"features_{basis}")
    circuit.ry(x, 0)
    circuit.rz(0.4 * x, 0)
    circuit.ry(2 * x + 0.3, 1)
    circuit.rz(-0.7 * x, 1)
    circuit.cx(0, 1)
    circuit.ry(0.7, 0)
    circuit.ry(-0.4, 1)
    circuit.rz(0.2, 1)
    circuit.cz(0, 1)
    for qubit in (0, 1):
        if basis == "YY":
            circuit.sdg(qubit)
        if basis in ("XX", "YY"):
            circuit.h(qubit)
    if measure:
        circuit.measure_all()
    return circuit


def measurement_probabilities(inputs):
    """Return (inputs, 3 settings, 4 outcomes) in project |q0 q1> order."""
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("inputs must be a finite one-dimensional sequence")
    output = np.empty((len(values), 3, 4))
    for i, x in enumerate(values):
        for setting, basis in enumerate(SETTINGS):
            state = Statevector.from_instruction(build_circuit(x, basis))
            # Native indices 00,01,10,11 mean q1q0. Swap the middle entries.
            output[i, setting] = state.probabilities()[[0, 2, 1, 3]]
    return output


def decode_counts(native_counts):
    """Qiskit meas-register bitstrings -> project-order four-outcome counts."""
    counts = np.zeros(4, dtype=np.int64)
    for bits, count in native_counts.items():
        if len(bits) != 2 or any(bit not in "01" for bit in bits):
            raise ValueError("Expected two-bit strings from the meas register")
        counts[int(bits[::-1], 2)] += count
    return counts


def sample_features(x, shots=128, seed=42):
    """Acquire independent XX/YY/ZZ samples using Qiskit's CPU sampler.

    Returns (3,4) counts in project order and nine feature estimates.
    S shots is per setting: the total resource count per input is 3*S.
    Qiskit's random draws need not match the original NumPy count files.
    """
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer")
    counts = np.empty((3, 4), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for setting, basis in enumerate(SETTINGS):
        # Distinct sampler seeds prevent reusing a random stream across bases.
        sampler = StatevectorSampler(seed=int(rng.integers(0, 2**32)))
        result = sampler.run([build_circuit(x, basis, measure=True)], shots=int(shots)).result()[0]
        counts[setting] = decode_counts(result.data.meas.get_counts())
    features = (counts @ SIGNS / shots).reshape(9)
    return counts, features
