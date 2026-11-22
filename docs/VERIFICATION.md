# Verification record and remaining limits

This file describes checks actually performed on 22 November 2026. Passing checks reduce known implementation risks; they do not prove that all possible errors are absent or that a scientific claim generalizes.

## Automated checks: 26 passed after the deeper audit

The original eight scientific tests below are retained. The expanded suite adds checks of duplicate result rejection, dataset-level weighting, known full-rank and rank-deficient linear solutions using the independent verifier, ten invalid configuration cases, default condition coverage, protection of partial outputs, a full alternate-configuration run/plot/audit, and appropriate floating-point portability tolerances that still reject meaningful changes.

1. **Independent circuit implementation:** normalized states and nonnegative, normalized Born probabilities; agreement with an independently expressed PennyLane circuit at five inputs, absolute tolerance 1e-13.
2. **Correlated records:** a known correlated distribution has identical single-qubit estimates and a deterministic +1 pair estimate; its empirical single-observable variance agrees with the independent-shot prediction within a stated tolerance.
3. **Sampling and reconstruction:** high-shot feature estimates approach exact values; estimates reconstructed from joint counts agree with stored features.
4. **Training-only preprocessing and ridge objective:** scaler statistics match the training set; ridge coefficients match a separately solved regularized linear system; intercept is unpenalized.
5. **Seed reproducibility:** repeated seeded measurements give identical counts; invalid zero-shot sampling is rejected.
6. **Measurement-basis/sign check:** all nine exact features match direct Pauli-operator expectations at 101 deterministic random inputs, independent of the basis-rotation/outcome decoder.
7. **Ridge selection:** the selected lambda minimizes independently recomputed validation loss on the supplied grid; ties select the smaller penalty regardless of grid order.
8. **OLS implementation:** coefficients and intercept match an independently assembled augmented least-squares solution.

The actual test runner output is preserved in `results/tests.xml`. Tests are required, not silently skipped if PennyLane is missing.

## Recorded-result reconstruction: passed

`python scripts/check_results.py` verifies:

- Experiment source hashes match the recorded version.
- 104 original saved files match their SHA-256 checksums.
- All 75 unique measurement acquisitions contain nonnegative joint counts whose totals equal the configured shot count for every input and setting.
- Stored features reconstruct from the joint counts.
- Both methods have paired keys for all 320 data/task/noise/budget/acquisition conditions.
- All 640 fitted model MSEs and selected lambdas reconstruct from the saved features and labels.

The machine-readable outcome is `results/semester/verification.json`.

## Fresh independent execution: passed

A fresh full execution using the same fixed configuration reproduced all 640 metric rows and all 100 raw input/label/acquisition arrays. The maximum absolute difference in numerical metrics in the original environment was 0.0. Across different numerical-library platforms, the comparison script permits small floating-point tolerances. Timestamps and elapsed time are not expected to reproduce.

The machine-readable outcome is `results/semester/reproduction_verification.json`. The second run itself is excluded from Git to avoid duplicating data; it can be regenerated using the README commands.

## Independent deep audit: passed

`python scripts/deep_audit.py --deep-physics` additionally checks all 1,120 recorded inputs against PennyLane; independently reconstructs 640 OLS/Ridge fits using NumPy least squares and regularized solves rather than the training helper; verifies all 640 saved test prediction vectors, train/validation/test losses, coefficient norms, rank/condition diagnostics, 32 summary groups and 16 paired comparisons; regenerates all 75 joint-count acquisitions from their seeds; checks the full condition/file key sets and the shared-resource ledger. See `results/semester/deep_audit.json`.

The audit found weak verification coverage and hard-coded plotting assumptions, not a changed experimental conclusion. Plotting now reads the recorded seeds, split boundary, tasks and budgets. Input/output guards preserve incomplete runs and reject unsupported settings. The summary uncertainty description now remains correct for different repetition counts. A fresh run with these changes matches the original committed baseline numerically; the baseline remains available at Git commit `9414b8b`.

## Experimental safeguards

- Test targets do not select scalers, penalties or circuit gates.
- Both models see identical records; no extra acquisition is granted to Ridge.
- Lambda-grid settings and both target functions were fixed before running the study.
- Primary clean-label results and secondary noisy-label results are separate.
- Error bars average acquisitions within datasets before computing SD over five independent dataset seeds. There are not 25 independent datasets.
- Both improvements and failures are retained; Ridge does not consistently win.
- Measurement-resource counts include all three settings and validation/test acquisitions, while counting shared records only once.

## What this does not establish

No ranked-conference novelty, theorem, quantum advantage, hardware performance or general superiority of Ridge is demonstrated. The study has two synthetic functions, one circuit, one split size and five dataset realizations. Small validation sets can select a poor penalty. Changing methods after viewing test outcomes would require a new, separately labelled evaluation. Reference checks establish source identity and relevant scope, not an exhaustive review of the field.

Paper drafting is intentionally deferred at the user's request. The repository contains implementation notes and evidence, not a final submission paper.
