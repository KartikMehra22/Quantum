"""Draw and sample the project's circuit using Qiskit; no hardware credentials."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from qiskit_demo.circuit import SETTINGS, build_circuit, measurement_probabilities, sample_features
except ModuleNotFoundError as error:
    if error.name and error.name.startswith("qiskit") and error.name != "qiskit_demo":
        raise SystemExit("Install the optional dependency: uv pip install --python .venv/bin/python -r requirements-qiskit.txt") from error
    raise
from qml_project.quantum import FEATURE_NAMES, measurement_probabilities as numpy_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.5)
    parser.add_argument("--shots", type=int, default=128, help="Shots per measurement setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="Check all bases at 101 inputs against the original simulator")
    args = parser.parse_args()
    if args.shots < 1 or not np.isfinite(args.x):
        parser.error("Use finite x and a positive shot count")
    if args.check:
        inputs = np.linspace(-np.pi, np.pi, 101)
        qiskit_values = measurement_probabilities(inputs)
        reference = numpy_probabilities(inputs)
        np.testing.assert_allclose(qiskit_values, reference, atol=1e-13, rtol=0)
        print(f"CHECK PASSED: 101 inputs, 3 bases; max probability difference={np.max(np.abs(qiskit_values-reference)):.3g}")
    print("Fixed two-qubit feature map (without measurement):")
    print(build_circuit(args.x).draw(output="text", fold=-1))
    for basis in SETTINGS:
        print(f"\n{basis} measurement circuit:")
        print(build_circuit(args.x, basis, measure=True).draw(output="text", fold=-1))
    counts, features = sample_features(args.x, args.shots, args.seed)
    print("\nJoint counts in PROJECT order |q0 q1>: 00, 01, 10, 11")
    for basis, row in zip(SETTINGS, counts):
        print(f"{basis}: {row.tolist()}")
    print("\nNine finite-shot feature estimates:")
    for name, value in zip(FEATURE_NAMES, features):
        print(f"{name:5s}: {value:+.6f}")
    print(f"CPU simulation only. Total simulated executions per input: {3 * args.shots}")
    print("This demonstration does not rerun the regression study or change saved results.")


if __name__ == "__main__":
    main()
