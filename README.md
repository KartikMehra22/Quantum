# Finite-shot quantum feature regression

## Technical completion and demonstration

The implementation, two experimental studies, saved-model prediction and reproducibility tools are complete. The core study has **54 tests**, with **four additional optional Qiskit checks**. Submission materials are now included: a five-page IEEE report, native presentation source and speaker notes, and a complete presentation/viva guide.

With the environment active, start here:

```bash
python scripts/defense_demo.py --refit
python scripts/predict.py --method FullCov --task sine --shots 32 --x -0.5 0.2 0.9
python scripts/verify_project.py
```

The first command refits and checks all four readouts for one fixed recorded case. The second predicts at new inputs using a saved model and simulated feature measurements. The third runs the full suite, both independent scientific audits, fresh reproductions and inference demonstrations, then records the result in `results/project_verification.json`. Temporary reproductions are removed after comparison; canonical experiment data are preserved.

See [technical completion and demo instructions](docs/TECHNICAL_COMPLETION.md) and [the four-method notebook](notebooks/02_measurement_aware_walkthrough.ipynb). New prediction inputs are restricted to the evaluated domain [-pi, pi]; this is not a deployment or quantum-advantage claim.

## Qiskit circuit demonstration

An independent Qiskit implementation of the same fixed circuit is available in [src/qiskit_demo/circuit.py](src/qiskit_demo/circuit.py), with a [runnable example](scripts/qiskit_circuit.py) and [setup/bit-ordering guide](docs/QISKIT_CIRCUIT.md).

```bash
uv pip install --python .venv/bin/python -r requirements-qiskit.txt
.venv/bin/python scripts/qiskit_circuit.py --x 0.5 --shots 128 --seed 42 --check
.venv/bin/python -m pytest -q tests/test_qiskit_circuit.py
```

This additional CPU sampler example supports XX, YY and ZZ measurements and correctly converts Qiskit's bit ordering to the original project convention. It does not replace the original experimental implementation, regenerate recorded scores or claim hardware execution. The four additional tests are skipped when the optional Qiskit dependency is absent. The report's 54-test count refers to the original recorded scientific verification.

## Presentation and submission materials (24 November)

- [Five-page IEEE report](output/pdf/outliers_ieee_report.pdf)
- [Editable LaTeX source](report/main.tex) and [build notes](report/README.txt)
- [Presentation source and export guidance](presentation/README.md)
- [Six-slide speaker notes](presentation/speaker_notes.txt)
- [Complete presentation and viva guide](docs/PRESENTATION_AND_VIVA.md): timed script, definitions, mathematics, results, code locations, runnable demo, literature boundaries and likely questions.

The editable deck is saved in the presentation viewer delivered in the project chat; export it there as PowerPoint for offline use. The repository contains its native authoring snapshot, not an exported `.pptx`. Enrollment numbers are limited to the presentation material and excluded from the report and viva guide.

## Measurement-aware extension (23 November)

The project now also implements **DiagonalCov** and **FullCov** readouts, using an empirical measurement-noise penalty in addition to ridge shrinkage. This extension was evaluated separately on **20 fresh datasets**, with five acquisitions per dataset and **32 validation candidates per regularized method**. The original two-method study below remains intact.

The result is mixed rather than a general improvement: FullCov has slightly worse primary clean-label mean MSE than the stronger Ridge baseline in five of six finite-shot settings and improves one by about 0.7%. Differences are small and are not presented as significant. The off-diagonal ablation also gives mixed outcomes. No test-based retuning was performed to force a win.

- [Extension guide and viva explanation](docs/EXTENSION_GUIDE.md)
- [Recorded evaluation protocol](docs/EXTENSION_PROTOCOL.md)
- [All four-method numerical results](docs/EXTENSION_RESULTS.md)
- Implementation: `src/measurement_extension/covariance.py`, `models.py`, `experiment.py`, `analysis.py`.
- Evidence: `results/measurement_extension/verification.json`, `reproduction_verification.json`, raw counts and fitted coefficient/prediction arrays.
- Figures: `figures/measurement_extension/`.

With the project environment active, reproduce the extension using a fresh destination:

```bash
python scripts/run_extension.py --output results/extension_rerun
python scripts/compare_extension_runs.py results/measurement_extension results/extension_rerun
python scripts/audit_extension.py --deep-physics
```

To regenerate the four comparison figures and the technical results note:

```bash
python scripts/summarize_extension.py
```

The original extension was checked with 41 tests; the current full suite has **54 tests**, including saved-model inference and walkthrough checks. The extension's independent audit checks 5,120 evaluated readouts, all 124,160 validation-candidate fits, 300 training-noise covariance estimates reconstructed from expanded shots, and 4,480 quantum inputs. This is a new implementation/evaluation for the project, not an invented covariance-regularization method. The final submission materials are linked above.

## Original two-method semester baseline

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

The implementation and preserved results are accompanied by the submission materials linked above.

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
