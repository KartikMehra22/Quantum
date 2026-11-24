# Explaining the semester project

For the complete current submission, use [PRESENTATION_AND_VIVA.md](PRESENTATION_AND_VIVA.md).

The project now has a four-method extension. Use `EXTENSION_GUIDE.md` for the current pitch, method comparison and fresh-seed results. The material below explains the preserved original two-method baseline.

## A 45-second explanation

Our project studies how limited quantum measurements affect regression. A fixed two-qubit circuit converts each input into nine measurable features. These features are estimated from 32, 128 or 512 shots per measurement setting. We compare ordinary linear regression and ridge regression using exactly the same data and measurement outcomes. Ridge limits coefficient magnitude, so we test whether it makes prediction more reliable. We also compare against exact simulator features. More shots reduced mean prediction error in our tested settings, while ridge gave small improvements in some cases and performed worse in others. This is a reproducible comparison of existing methods, not a new quantum algorithm.

## Questions you should be able to answer

**What is a shot?** One preparation, circuit execution and measurement. Our circuit has three separate measurement settings, so S shots means 3*S executions per input.

**What is quantum about the project?** The feature map is a two-qubit quantum circuit and the measurement distribution follows the Born rule. The linear readout is classical. The whole small circuit is simulated on a classical CPU.

**Are you training the quantum circuit?** No. Its gates remain fixed. Only the classical readout weights and intercept are fitted.

**What are the inputs and outputs?** An input x in [-pi,pi]; a continuous target sin(x) or sin(2x)+0.3*cos(3x). The circuit returns nine expectation-value estimates used by the classical model.

**How are nine features measured with two qubits?** Each of the XX, YY and ZZ settings gives two single-qubit values and their pair product. Three settings times three estimates gives nine features.

**Why not add random Gaussian noise to the features?** Actual joint-outcome sampling preserves outcome probabilities, bounded observables and correlations. Gaussian noise would be a different measurement model.

**What does ridge do?** It adds lambda times the squared coefficient norm to the mean training squared error. This shrinks coefficients, adding bias in exchange for potentially less sensitivity. It does not remove quantum hardware errors.

**How was lambda chosen?** From a fixed list using the validation set only. The test set was not used to choose it. The scaler was fitted only on training features.

**Is ridge always better?** No. For clean labels, it is slightly better on the sine task at all three finite shot counts, but worse on mixed harmonics at 32 shots. Exact numbers are in docs/RESULTS.md. We must report the negative result too.

**Why use exact features?** As a diagnostic reference to separate sampling error from imperfect representation or learning. Exact features are available to the simulator, not free to a real quantum device.

**What are your contributions?** A verified two-qubit simulator; a fair, reproducible experiment; saved raw data, seeds and resource accounting; actual comparisons and an honest interpretation.

**How many papers and theorems did you evaluate?** Five research references motivate the study, plus official software documentation. We experimentally evaluate two methods; we do not prove a theorem or reproduce all five papers.

**How many experiments?** 600 finite-shot fitted models and 40 exact-reference models. There are only five independent dataset realizations; measurement repeats within a dataset are correlated through the shared data. We do not treat all model fits as independent trials.

**Did you use a GPU or quantum hardware?** No. The two-qubit experiment runs on a CPU. The recorded shot count represents simulated measurement resources, not real QPU executions.

**Can this establish quantum advantage?** No. The circuit is tiny and classically simulable, and we did not compare quantum and classical feature families.

**What would you do next?** More datasets, repeated validation, other fixed circuits and stronger regularization baselines. These are possible extensions, not claims in the current report.

## Demonstration order

1. Show figures/quantum_circuit.png and explain the fixed gates.
2. Show figures/workflow.png and explain the split and validation step.
3. Show figures/feature_precision.png: more shots improve measurement precision.
4. Show figures/test_mse_clean.png: compare both models including the failure case.
5. Run `python scripts/check_results.py` to reconstruct the results.
6. Open the raw records and the summary table if asked about reproducibility.
