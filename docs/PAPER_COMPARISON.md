# What we compare, and what we do not reproduce

Checked against primary paper methods and relevant discussion on 22 November 2026. This is a technical scope note, not a paper draft. Differences in circuits, targets, observables, sample sizes and inference measurements prevent direct numerical comparisons of reported MSEs.

| Reference | What is actually studied there | Our experiment | Permitted interpretation |
|---|---|---|---|
| Kreplin and Roth, *Reduction of finite sampling noise in quantum neural networks* (Quantum, 2024) | Trainable quantum neural networks with output-variance regularization; shot-based inference and hardware examples. | Fixed two-qubit circuit, classical OLS and coefficient-norm ridge penalty. | We investigate related sampling uncertainty. We neither implement their variance objective nor reproduce their variance-reduction factors. [Primary paper](https://arxiv.org/html/2306.01639), Secs. 2, 3 and 4. |
| Hu et al., *Fundamental Limits and Eigentasks* (PRX, 2023) | Feature combinations characterized by signal/noise and eigentask filtering. | Nine fixed observable features; no eigentask transform or selection. | A motivation for noise-aware features, not a baseline we can claim to beat. [Primary record](https://arxiv.org/abs/2307.16083). |
| Hu and Gao, *Generalization Error ... Sampling Noise* (2024 preprint) | Finite-data/finite-shot generalization analysis and task-dependent eigentask truncation under a defined theoretical model. | Fixed sample count, two synthetic functions and a validation-selected ridge grid. | We do not validate their analytical generalization formula or its optimality conditions. [Primary paper](https://arxiv.org/html/2410.14654), Secs. III-IV. |
| Lo Monaco et al., *Who can sample forever?* (2026 preprint) | Quantum linear regression with measurement-probability features; shot noise can populate directions absent from ideal feature space. Its discussion leaves a systematic regularization comparison for future work. | Local-observable expectations plus an intercept, with a full-rank feature matrix. | Our rank diagnostics do not exhibit that paper's noise-only direction mechanism. Decreasing error with shots in our setting neither proves nor contradicts their regime-dependent result. [Primary paper](https://arxiv.org/html/2610.12015v1), Secs. IV and IX. |
| Lo Monaco et al., *Triple descent ...* (2026 preprint) | Measurement-outcome dimension and interpolation regimes, with explicit ridge analysis. | Feature count remains nine and training size remains 64; no dimension sweep across interpolation. | We cannot demonstrate triple descent or claim to introduce ridge into QML. [Primary paper](https://arxiv.org/html/2611.00272), Secs. IV.2-IV.4. |

## Direct comparisons actually run

We fit OLS and Ridge to identical records at 32, 128 and 512 shots, plus exact-feature references. The classical regression weights are trained; the quantum gates are fixed. Primary clean-label results and the separate noisy-label sensitivity check are retained. The circuit is CPU-simulated rather than run on a physical QPU.

In the saved experiment, every augmented feature matrix has rank 10 (nine features plus intercept). Standardized exact-feature condition numbers range from approximately 82.23 to 112.50; finite-shot values range from approximately 5.15 to 25.54. Thus sampling reduced this particular condition-number diagnostic while increasing mean prediction error relative to exact features. A condition number alone does not establish better statistical features or reliable predictions. These are descriptive diagnostics for this setup, not a new theory.

Ridge modestly reduces primary mean MSE in five of six finite-shot task/budget conditions and worsens mixed harmonics at 32 shots. These differences do not establish significance or superiority over variance regularization, eigentasks or the papers' methods. No published numerical result is presented as if our experiment reproduced it.

## Realistic semester claim

"We implemented and independently checked a small quantum-feature regression simulation, evaluated two established classical readouts under matched shot budgets, preserved raw acquisitions, and documented conditional regularization benefits and failure cases."

A paper reproduction would require matching the original feature model, targets, dimensions, train/test acquisition regimes, estimator conventions and uncertainty analysis. That work is not part of the completed study and should not be described as done.
