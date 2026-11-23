# Technical project completion

The code and experiments are complete for the agreed semester scope. This handoff adds a runnable demonstration, saved-model inference and an end-to-end verification command. It intentionally does not include a final paper or presentation.

## What is complete

- Fixed two-qubit circuit, joint XX/YY/ZZ sampling and nine features.
- Original OLS/Ridge study, preserved unchanged.
- Four-method fresh-seed extension: OLS, Ridge, DiagonalCov and FullCov.
- Training-only noise covariance, standardization and validation selection.
- All raw records, coefficients, scalers, predictions, result tables and plots.
- Independent audits of physics, covariance, candidate fits and summaries.
- New-input prediction from recorded models, with no penalty retuning.
- A refit demonstration and executable four-method notebook.
- 54 tests and one command for all technical verification, also run by GitHub Actions.

## Run the demonstration

From the repository root, activate the project environment:

```bash
source .venv/bin/activate
python scripts/defense_demo.py --refit
```

This selects the first fresh dataset and acquisition in the recorded configuration, the sine task, clean labels and 32 shots per setting. It refits all four methods using the original training/validation split, checks their predictions against the saved case, prints test MSE and selected penalties, and shows the first five test inputs in fixed order. It is a one-case demonstration, not a replacement for the 20-dataset summary.

Other recorded cases:

```bash
python scripts/defense_demo.py --task mixed_harmonics --shots 512 --refit
python scripts/defense_demo.py --shots 0 --refit
```

Shots=0 selects the exact-feature simulator reference; it is not a hardware measurement mode.

## Predict at new inputs

```bash
python scripts/predict.py --method FullCov --task sine --shots 32 --x -0.5 0.2 0.9
```

The command loads a recorded fitted readout, generates fresh feature measurements from the fixed circuit, and applies the stored scaler, coefficients and intercept. It does not learn from the supplied inputs or their targets. JSON output identifies the training dataset, selected lambda/gamma, sampling seed and simulated measurement cost. Set `--sampling-seed` to reproduce a measurement draw. Use the same seed and shot count when comparing predictions across methods on common simulated measurements.

Available methods are OLS, Ridge, DiagonalCov and FullCov. The evaluated domain is [-pi,pi]; unsupported extrapolation is rejected. A single new-input prediction is not evidence of aggregate accuracy or deployment readiness. Both quantum state evolution and measurement sampling are CPU simulations.

The API is `measurement_extension.inference.load_readout`. Its `predict_features` method accepts nine-feature vectors; `predict_inputs` encodes raw x values and generates measured features. Loaded coefficient/scaler arrays are read-only to avoid accidental mutation.

## Verify the entire technical project

```bash
python scripts/verify_project.py
```

This executes the test suite, both independent stored-result audits, fresh original and extension experiments in temporary directories, their reproduction comparisons, the refit demo and prediction CLI. It stops on failure and writes a receipt to `results/project_verification.json`. Fresh temporary outputs are removed after checking. Original results and source snapshots are preserved.

For a separate receipt destination:

```bash
python scripts/verify_project.py --report /tmp/quantum-check.json
```

GitHub Actions uses the same command and retains its receipt as a workflow artifact. Local and Linux comparisons permit small floating-point rounding differences while keeping saved-byte hashes and integer-count checks exact.

## Walkthrough and main locations

| Purpose | Location |
|---|---|
| Current four-method notebook | notebooks/02_measurement_aware_walkthrough.ipynb |
| Saved-model loading and inference | src/measurement_extension/inference.py |
| Refit demonstration | scripts/defense_demo.py |
| Prediction CLI | scripts/predict.py |
| Complete technical check | scripts/verify_project.py |
| Inference and notebook tests | tests/test_inference.py |
| All numerical findings | docs/EXTENSION_RESULTS.md |
| Method, limitations and viva | docs/EXTENSION_GUIDE.md |
| Original and extension raw evidence | results/semester/; results/measurement_extension/ |

## Outcome to explain

More shots improve average predictions in the tested settings. FullCov improves only one of six primary clean-label means relative to the stronger Ridge baseline, by about 0.71%; the other primary means are slightly worse. The ablation and noisy-label outcomes are mixed. No significance, new-method novelty or quantum advantage is established.

The completed technical contribution is the implementation, controlled evaluation and reproducible evidence, including the failure cases. The added inference tools do not change or improve those measured findings. Future algorithm changes need a separate protocol and fresh evaluation rather than retuning against these inspected test results.
