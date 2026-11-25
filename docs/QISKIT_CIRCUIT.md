# Qiskit version of the project circuit

This is an independent Qiskit expression of the same fixed circuit used by the semester experiment. It is an additional CPU simulator demonstration and cross-check. The recorded regression experiments still use the original NumPy implementation, checked against PennyLane; they were not rerun using Qiskit.

## Install and run

From the repository root, using the existing environment:

```bash
uv pip install --python .venv/bin/python -r requirements-qiskit.txt
.venv/bin/python scripts/qiskit_circuit.py --x 0.5 --shots 128 --seed 42 --check
.venv/bin/python -m pytest -q tests/test_qiskit_circuit.py
```

On a standard pip-enabled virtual environment, `python -m pip install -r requirements-qiskit.txt` is an alternative. Tested SDK version: Qiskit 2.6.2. Qiskit Aer, cloud credentials, a GPU and quantum hardware are not required.

The demonstration prints the fixed circuit, three measured circuits, joint counts and nine features. `--check` compares all three basis distributions at 101 inputs against the original simulator. S shots is per setting, so the execution ledger per input is 3S. These new sampler draws are not expected to reproduce the original saved NumPy counts bit-for-bit.

## Minimal circuit code

```python
from qiskit import QuantumCircuit

x = 0.5
qc = QuantumCircuit(2)
qc.ry(x, 0)
qc.rz(0.4 * x, 0)
qc.ry(2 * x + 0.3, 1)
qc.rz(-0.7 * x, 1)
qc.cx(0, 1)
qc.ry(0.7, 0)
qc.ry(-0.4, 1)
qc.rz(0.2, 1)
qc.cz(0, 1)
print(qc.draw(output="text"))
```

For computational/Z measurement, append `qc.measure_all()`. For X measurements, apply H on both qubits before measurement. For Y, apply S-dagger and then H on each qubit before measurement. Build a separate circuit for each setting; do not append all basis measurements to the same prepared state.

The complete implementation is `src/qiskit_demo/circuit.py`:

- `build_circuit(x, basis="ZZ", measure=False)`: exact gate sequence plus optional measurement rotations.
- `measurement_probabilities(inputs)`: all three exact basis distributions in the project's outcome order.
- `sample_features(x, shots=128, seed=42)`: Qiskit StatevectorSampler acquisition, joint counts and nine finite-shot feature estimates.
- `decode_counts`: converts Qiskit's bitstrings to the original convention.

## Bit order matters

Physical qubit labels remain q0 and q1. Qiskit statevector arrays and the measured two-bit register strings use q1q0 ordering, whereas the original project uses q0q1. The probability conversion swaps indices 1 and 2: `[0,2,1,3]`. Count strings are reversed before indexing project records. Reversing display order alone is not a substitute for this conversion.

The tests compare complex state amplitudes, all three basis probabilities, a deliberately asymmetric count example, sampled feature agreement, repeatability, totals and measurement-circuit construction.

## Interpretation for the viva

“We added a Qiskit implementation of the same fixed circuit and checked it against the original simulator. The established regression results remain those of the original NumPy/PennyLane-checked pipeline. The Qiskit demonstration runs locally on a CPU; it is not a hardware experiment.”

This SDK translation does not turn the project-defined circuit into an architecture taken from a reference paper, and it does not establish quantum advantage.

Official documentation:

- [Statevector and probabilities](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)
- [StatevectorSampler](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorSampler)
- [Qiskit bit ordering](https://quantum.cloud.ibm.com/docs/en/qiskit/bit-ordering)
