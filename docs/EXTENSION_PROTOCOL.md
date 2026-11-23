# Measurement-aware extension: fixed evaluation protocol

Recorded before running the new held-out evaluation on 23 November 2026. This is a repository protocol, not a formal external preregistration. Original two-method results and original qml_project modules remain unchanged.

## Question and contribution

Does an empirically estimated measurement-noise covariance penalty improve held-out prediction relative to a well-tuned Ridge baseline under matched shot budgets? Does retaining within-setting off-diagonal noise covariance help relative to the diagonal ablation?

Covariance-regularized readouts are existing methods (see Baumann et al., arXiv:2604.28160v2, Sec. III-C, Cov-EV). This is an implementation/adaptation and evaluation, not a new invention or a full reproduction of their forecasting study. https://arxiv.org/html/2604.28160

## Fixed design

- Same two-qubit circuit, nine features, two synthetic targets, 64/32/128 train/validation/test inputs and shots per setting 32/128/512.
- Fresh dataset seeds 1001-1020; fresh acquisition seeds 2101-2105. These are disjoint from the previously inspected study. No extension method or parameter is selected using the previous or fresh test MSE.
- Primary condition: clean train/validation labels. Separate sensitivity: Gaussian label noise SD 0.1. Test target always the noiseless function.
- Four methods: OLS, Ridge, DiagonalCov, FullCov. Exact-feature references use zero measurement covariance and are reported separately; their penalty grids can differ in resolution and they need not return identical selected fits.
- Primary contrast: FullCov versus Ridge on clean-label finite-shot MSE. Ablation contrast: FullCov versus DiagonalCov. OLS remains a control.

## Noise estimator and objective

For input i and setting g, use its existing S joint-outcome counts to reconstruct sums of v=(z0,z1,z0*z1) and vv^T. The unbiased sample-mean covariance is [sum(vv^T)-S*m*m^T]/[S*(S-1)]. It measures repeated-measurement uncertainty at a fixed input, not the distribution of features across inputs.

Average these matrices over TRAINING inputs only. Different setting acquisitions are independent in this simulator, so cross-setting blocks are zero. FullCov retains each 3x3 within-setting block; DiagonalCov removes off-diagonal terms. Propagate through train-only StandardScaler: C_scaled=D^-1*C*D^-1. Tiny numerical negative eigenvalues are clipped at zero; a negative eigenvalue below -1e-12 is an error.

Fit mean training squared error + lambda*||w||^2 + gamma*w^T*C_scaled*w with an unpenalized intercept. This is an empirical regularizer, not a claim that this objective is an unbiased corrected training risk. No feature-noise subtraction or deployment-adaptation theorem is claimed.

## Fixed tuning and fairness

- Ridge: 32 unique lambdas, formed by 26 logarithmically spaced values from 1e-6 to 10 plus the original eight lambdas. Both endpoint duplicates are removed. This includes every gamma=0 covariance candidate.
- Each covariance method: the original eight lambdas [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10] crossed with gamma [0,0.1,1,10], giving 32 candidates.
- Each regularized method has the same number of validation candidates. Their model families and grid resolutions still differ; equal candidate counts are not proof of identical model-selection complexity. Both covariance methods use exactly the same grids. A covariance model choosing gamma=0 cannot have a strictly better minimum validation loss than the richer Ridge grid except for numerical rounding; such choices are reported and never attributed to covariance.
- Gamma=0 reduces the covariance model to Ridge at the same lambda, checked by tests. Ties prefer smaller gamma, then smaller lambda. Ridge ties prefer smaller lambda. No train+validation refit after selection.
- Estimate covariance using training measurements only; no oracle covariance, validation counts or test counts feed the fitted penalty. Both baselines and new methods share the same raw records. All candidate fits are classical and require no extra simulated quantum shots.

## Recorded outcomes and uncertainty

Save all predictions, chosen parameters, coefficients, scalers, training penalties, losses, raw inputs/labels/counts, config, source snapshots/hashes and software versions. Average five acquisitions within each dataset first. Report means and sample SD over twenty dataset means, with paired differences and the number of dataset means favouring each method. SDs are not confidence intervals; no significance claim is planned. Keep all failures and gamma=0 choices.

Expected evaluated readouts: 4 tasks/noise combinations * 20 datasets * (15 finite acquisitions + 1 exact reference) * 4 methods = 5,120. Of these 4,800 are finite-shot and 320 exact-reference evaluations. Internal validation candidates are additional fits, not additional independent datasets. Unique finite acquisitions: 300. Simulated shot ledger: 20*5*224*3*(32+128+512)=45,158,400, reused across methods, targets, labels and tuning.

## Verification before interpreting results

Check covariance against explicitly expanded outcomes, theoretical covariance in expectation, PSD/symmetry/block structure, scaling, intercept, gamma=0 Ridge equivalence, diagonal/full equivalence for diagonal noise and independent validation selection. Independently reconstruct penalties and fitted weights using a different linear-algebra route; validate predictions and summaries. Audit the original study unchanged. Run a fresh extension reproduction and compare saved outputs.

No post-test change will be presented as a prespecified primary experiment. A negative or mixed result will be retained. GPU and physical QPU access are not needed for this scope. Paper writing remains deferred.
