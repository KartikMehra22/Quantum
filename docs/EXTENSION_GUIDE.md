# Measurement-aware extension: implementation, findings and viva

## What was added

The project now tests two measurement-aware classical readouts in addition to OLS and Ridge:

- **DiagonalCov:** penalty uses individual feature-estimation variances.
- **FullCov:** penalty also retains covariance between features extracted from the same measurement record.

The quantum circuit remains fixed. These are adaptations of an existing covariance-regularized readout principle, not new inventions or a reproduction of Baumann et al.'s full forecasting system. See https://arxiv.org/html/2604.28160, Sec. III-C (Cov-EV).

## Why the estimator is different from ordinary feature covariance

For each fixed input, repeated shots fluctuate. We reconstruct single-shot vectors v=(z0,z1,z0*z1) from the four joint-outcome counts. For S shots, the unbiased covariance of their MEAN is

`C_mean = [sum(v v^T) - S*m*m^T] / [S*(S-1)]`.

This is not the covariance of the feature values across different inputs. It is measurement uncertainty conditional on an input. The three independent measurement settings give three 3x3 blocks and zero cross-setting blocks. Training-only averaging estimates a representative noise penalty. The diagonal ablation drops the off-diagonal entries; both methods use exactly the same other settings.

Feature scaling must also transform noise: `C_scaled = D^-1 C_mean D^-1`, using only training-feature standard deviations. The intercept is unpenalized. All methods use the same measurements, so the covariance calculations add classical work but no extra simulated shots.

## Objective and tuning

`Loss = mean training squared error + lambda*||w||^2 + gamma*w^T C_scaled w`.

OLS has no penalty. Ridge uses 32 lambdas, including every lambda available to the covariance methods at gamma=0. Each new method uses eight lambdas crossed with gamma=[0,0.1,1,10], also 32 candidates. Validation MSE chooses parameters, with no test-based model selection. Equal candidate counts do not imply identical model-family complexity; this limitation is explicit.

Gamma=0 turns off the added covariance term. For exact features C=0, DiagonalCov and FullCov reduce to Ridge on their coarser lambda grid; any exact-reference difference from the finer Ridge grid is not evidence of a covariance effect.

## Fresh evaluation

The protocol was recorded in Git before the new held-out evaluation. It uses dataset seeds 1001-1020 and acquisition seeds 2101-2105, disjoint from the old study. There are 20 independent datasets, each with 64 training, 32 validation and 128 test inputs. Both targets, three shot counts and the clean/noisy-label distinction are retained. Acquisition repeats are averaged within dataset before calculating descriptive means and SDs.

There are 5,120 evaluated readouts: 4,800 finite-shot and 320 exact references. The 124,160 internal candidate fits, including OLS, are not independent statistical trials. There are 300 unique acquisitions and 45,158,400 simulated executions, shared across methods, tasks, labels and tuning. This is a resource ledger, not physical QPU execution or runtime.

An inherited configuration description initially said five dataset means, despite the numeric seed list and protocol specifying twenty. That descriptive field was corrected before interpreting scores, and the experiment was rerun unchanged. All 5,120 metric rows and fitted-model arrays match exactly. This is documented in `metadata_correction_verification.json`; no numerical setting or result was changed to improve performance.

## Actual outcome

FullCov improves primary clean-label mean test MSE versus Ridge in **one of six** finite-shot task/budget settings, by approximately **0.71%** (mixed harmonics, 512 shots). In the other five it is slightly worse, by approximately 0.12-0.76%. It improves the separate noisy-label mean in four of six finite-shot settings, also generally by small amounts, and worsens the other two.

Retaining off-diagonal terms beats the diagonal ablation in only three of six primary means. Therefore, this setup does not establish that including measured correlations improves accuracy. Across the six primary settings, FullCov chooses gamma=0 in approximately 43-57% of individual dataset/acquisition fits. Selection sometimes switches off the covariance penalty.

These are descriptive results without significance claims. Plausible limitations include covariance estimation error, limited validation data, the extra tuning dimension and an already competitive Ridge baseline. We have not experimentally isolated a unique causal explanation for the failures. Results were not used to redesign the method or search for a favourable seed.

## Verification

- Combined suite: 41 tests, including expanded-record covariance equality, theoretical covariance in expectation, PSD/block structure, standardization, unpenalized intercept, gamma=0 Ridge equivalence, diagonal/full equality under diagonal noise, fresh-seed validation and a full alternate-configuration pipeline.
- Independent extension audit: reconstructs training covariances using explicitly expanded shots and refits candidates via augmented least squares instead of the production normal-equation solver.
- All 5,120 readouts and 124,160 validation candidates checked; all stored weights, intercepts, scalers and predictions checked.
- All 4,480 inputs checked against PennyLane; all 400 raw files checked and all 300 acquisitions regenerated from seeds.
- A fresh reproduction matches all numerical metrics and fitted arrays; max absolute metric difference is 0.0 in the original environment.
- The original study and its independent audit still pass unchanged.

Checks cannot establish general scientific validity, significance or absence of every possible error. See the machine-readable evidence in `results/measurement_extension/`.

## Where to look

| Component | Location |
|---|---|
| Frozen design | configs/measurement_extension.json; docs/EXTENSION_PROTOCOL.md |
| Noise estimator and scaling | src/measurement_extension/covariance.py |
| Four readouts and candidate grids | src/measurement_extension/models.py |
| Fresh data, acquisition, fitting and provenance | src/measurement_extension/experiment.py |
| Dataset-level summaries and paired contrasts | src/measurement_extension/analysis.py |
| Runner and configuration guards | scripts/run_extension.py; scripts/validate_extension.py |
| Independent checks | scripts/audit_extension.py |
| Reproduction comparison | scripts/compare_extension_runs.py |
| Plots and numerical note | scripts/summarize_extension.py |
| New unit/integration tests | tests/test_measurement_extension.py |
| Complete evaluation rows | results/measurement_extension/metrics.csv |
| Coefficients, intercepts, scalers and test predictions | results/measurement_extension/fitted_models.npz |
| Original counts and probabilities | results/measurement_extension/raw/ |
| All numerical tables | docs/EXTENSION_RESULTS.md |
| Four figures | figures/measurement_extension/ |

## Faculty pitch

"We extended our quantum-feature regression baseline with measurement-aware regularization. Ordinary Ridge shrinks coefficients uniformly, whereas our added penalty uses estimated measurement variance and covariance. We implemented a diagonal ablation to test whether correlations provide additional benefit. We evaluated four methods on twenty fresh datasets under matched shot budgets and matched validation-candidate counts. Independent tests verify the measurement statistics, linear algebra and saved predictions. The covariance-aware method did not consistently outperform a strong Ridge baseline, so our contribution is the verified implementation and controlled evaluation, including its negative findings."

## Questions to expect

- **Did you invent covariance regularization?** No; it is an existing approach adapted to this feature map.
- **Why estimate covariance from training counts only?** To fit the penalty without using held-out measurements or targets.
- **What does the diagonal ablation remove?** Correlations between shot-derived features, while retaining individual variances.
- **Why no guaranteed improvement?** A more detailed penalty can itself be estimated imperfectly and can add bias or validation-selection uncertainty. Our data do not isolate one cause.
- **Is gamma=0 a success of covariance modelling?** It is a useful fallback, but not evidence that covariance helped.
- **Is the project still useful with negative results?** It demonstrates implementation, mathematical verification, fair comparison and the boundary of an existing method in this setting.
- **What comes next?** A separately designed study with more robust validation, other feature maps or matched reproduction of one reference. Any design chosen after these results needs a new evaluation, not test-set reuse.

Paper drafting remains deferred.
