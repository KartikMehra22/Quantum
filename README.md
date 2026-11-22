# Finite-shot quantum feature regression

A completed, CPU-based semester experiment comparing **ordinary least squares (OLS)** and **validation-tuned ridge regression** on the same two-qubit measurement features. This is an empirical study of established methods, not a new quantum algorithm or a claim of quantum advantage.

## Start here

- `docs/METHODOLOGY.md`: exact circuit, data splits, tuning and measurement budget.
- `docs/VIVA.md`: short explanation, likely questions and honest answers.
- `docs/RESULTS.md`: automatically generated numerical findings.
- `docs/VERIFICATION.md`: what was checked and what the checks cannot establish.
- `docs/AUDIT_FINDINGS.md`: weaknesses found, fixes applied and unchanged numerical conclusions.
- `docs/REFERENCES.md`: verified background sources and their relevance.
- `docs/PAPER_COMPARISON.md`: precise differences from the papers; no claimed numerical reproduction.
- `figures/`: vector PDF and high-resolution PNG figures, including circuit and workflow.
- `results/semester/`: original metrics, test predictions, inputs, measurement counts, seeds, configuration and provenance.

## What was tested?

| Item | Fixed setting |
|---|---|
| Quantum circuit | Two qubits, fixed gates; no variational training |
| Features | X0, X1, X0X1, Y0, Y1, Y0Y1, Z0, Z1, Z0Z1 |
| Joint measurement settings | XX, YY, ZZ; each costs the stated shot count |
| Shots per setting | 32, 128, 512, plus an exact simulator reference |
| Targets | sin(x); sin(2x) + 0.3 cos(3x), x uniform on [-pi, pi] |
| Data splits | 64 training, 32 validation, 128 held-out test inputs |
| Primary study | Clean labels |
| Secondary sensitivity check | Independent Gaussian noise SD 0.1 on train/validation labels; test target is the noiseless function |
| Repetitions | 5 dataset seeds x 5 independent measurement seeds |
| Models | OLS and Ridge, identical input records and train-only scaling |
| Tuning | 8 predefined positive lambdas, validation MSE only; no test-based choices |
| Outcome | Test MSE, coefficient norm, feature-estimation MSE and variation |

The stored study has **600 finite-shot fits and 40 exact-feature fits**, not 640 independent datasets. Exact features are not repeated across measurement seeds. Error bars average the five acquisitions within each dataset first, then show sample SD over five dataset means. No confidence or significance claim is made from these error bars.

## Installation (Python 3.11 or newer; tested with 3.12.14)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the main libraries. `requirements-lock.txt` records all installed versions from the original environment. The experiment itself uses NumPy and scikit-learn; PennyLane independently checks the circuit in the tests. A dedicated GPU and physical quantum device are unnecessary for this small simulation.

## Run the checks and reproduce

From the repository root:

```bash
python -m pytest -q
python scripts/check_results.py
python scripts/deep_audit.py --deep-physics
python scripts/run_study.py --output results/reproduction
python scripts/make_figures.py --results results/reproduction --figures figures/reproduction
python scripts/compare_runs.py results/semester results/reproduction
```

The command-line runner validates the configuration and refuses to write into any nonempty output directory, including incomplete runs. Use a fresh output directory for each rerun. It writes seeds, raw joint-outcome counts and the configuration before reporting any result. Exact numerical agreement is checked with a tolerance for floating-point library differences; the bundled checksum audit checks the original saved bytes. The independent deep audit uses NumPy linear algebra rather than the experiment's sklearn fitting helper, checks all prediction vectors and summaries, and can check all 1,120 inputs against PennyLane.

To regenerate the checked-in figures and numerical findings from the original results:

```bash
python scripts/make_figures.py
python scripts/summarize_results.py
```

Paper preparation is intentionally deferred. This repository focuses on the implementation, preserved results and verification.

## Important interpretation

Ridge is **not consistently better**. Its gains are usually modest, and it worsens some conditions. More shots reduce average error in these particular experiments; that is not a theorem about every quantum learning model. The exact-feature reference is a simulator diagnostic, not a free measurement strategy available on real hardware. Synthetic one-dimensional tasks and a fixed small circuit cannot establish useful quantum advantage or reproduce all mechanisms from the cited papers.

The measurement budget is a **simulated resource count**, not actual QPU executions. Multinomial sampling draws joint outcome counts directly from Born probabilities. Reusing these counts for both methods and all classical tuning candidates requires no additional simulated quantum acquisitions.

## Repository map

```text
configs/semester.json       Frozen experimental settings
src/qml_project/quantum.py  Explicit statevector, joint sampling, PennyLane cross-check
src/qml_project/models.py   Train-only scaler, OLS and validation-tuned Ridge
src/qml_project/experiment.py  Seeds, targets, acquisitions and metrics
src/qml_project/analysis.py    Hierarchical descriptive aggregation
scripts/                   Run, audit, reproduce and generate report materials
tests/                     Independent physics and regression checks
results/semester/raw/       Inputs, exact probabilities, labels and joint counts
figures/                   Plots, workflow and circuit
docs/                      Methodology, results and evaluation preparation
```

See docs/REFERENCES.md for the five research references and official scikit-learn documentation. The literature supports the motivation; this code is not a full reproduction of those papers.
