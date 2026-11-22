# Exact experimental specification

## Circuit and features

Starting from |00>, apply, in order: RY(x) on q0; RZ(0.4x) on q0; RY(2x+0.3) on q1; RZ(-0.7x) on q1; CNOT(q0,q1); RY(0.7) on q0; RY(-0.4) on q1; RZ(0.2) on q1; CZ(q0,q1). These constants were fixed in the configuration/code before the main experiment. There is no parameter optimization.

Basis ordering is |q0 q1> = |00>, |01>, |10>, |11>. Measure both qubits in X, Y and Z in three separate settings. To measure X rotate by H; to measure Y apply S-dagger followed by H; to measure Z apply nothing. Born probabilities are the squared amplitudes after these rotations. The independent PennyLane implementation uses the same gate sequence without calling the NumPy statevector code.

For outcome bits b0,b1, assign z0=(-1)^b0 and z1=(-1)^b1. One record supplies z0, z1 and z0*z1. Average S records for three features in each setting. Across three settings there are nine features. Sharing outcomes within a group preserves measurement covariance. Draw counts from the joint four-outcome multinomial distribution rather than adding arbitrary noise to exact features.

For each individual +/-1 observable, the conditional sample-mean variance is (1-f(x)^2)/S under independent shots. The feature precision figure compares empirical mean squared feature errors to the average of this prediction. It does not assume different features in a setting are independent.

## Data and labels

Use 224 independent uniform inputs on [-pi,pi] per data seed, sliced into 64 train, 32 validation and 128 test inputs. No split is moved after seeing performance. Targets are sin(x) and sin(2x)+0.3*cos(3x). In the secondary noise condition only, independent N(0,0.1^2) noise is added to train and validation labels; test labels always denote the known noiseless function. This is latent-function estimation, not prediction of noisy observed test labels.

Primary results use clean labels. The noisy-label condition is explicitly separated as sensitivity analysis; it is not pooled to manufacture an improvement. Input and measurement records are reused across tasks and noise conditions, so comparisons across those conditions are correlated.

## Models and leakage control

StandardScaler is fitted to the training features only, separately for each acquisition and shot condition. Validation and test features are transformed with that fitted scaler. OLS uses scikit-learn LinearRegression. Ridge uses its SVD solver and an unpenalized intercept. Both see identical records.

Minimize mean training squared error plus lambda*||w||^2. Scikit-learn uses summed squared error plus alpha*||w||^2, so alpha=n_train*lambda. Lambda candidates are [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]. Minimize validation MSE and break exact ties in favour of smaller lambda. Do not refit on train+validation after selection, to retain an identical training sample count. No test input is used to fit the scaler; no test target is used to select a parameter.

## Repetitions and accounting

Data seeds: 11,22,33,44,55. Acquisition seeds: 101,202,303,404,505. Each (data seed, acquisition seed, shots) key has independently generated outcomes, including independent inputs/splits within that acquisition. Different shot budgets are not nested prefixes. Both models are paired by shared records.

The 25 acquisition/dataset combinations are not 25 independent datasets. First average acquisitions for each dataset; report the mean and sample SD over the five dataset-level means. These bars show descriptive dispersion, not confidence intervals. Exact reference models have one acquisition-free fit per dataset and are never duplicated to imply more repetitions.

Every input uses three settings, so a finite-shot condition costs 3*S*(64+32+128) simulated executions. Shared records do not get charged again for another model, target, noise condition or lambda. Across all 75 unique acquisitions the budget is 11,289,600 simulated shots. This is a resource ledger, not a hardware runtime estimate.

## Verification

Unit tests check state normalization and PennyLane probability agreement, direct Pauli-operator expectations at 101 random inputs, joint single/pair consistency, empirical measurement variance, convergence to ideal features, training-only scaling, ridge's normalization and intercept, independent validation-grid selection including ties, the OLS solution against augmented least squares, and seed reproducibility. The result auditor checks source hashes, checksums, nonnegative counts, per-setting totals, reconstructed features, paired method keys and refits all 640 models from saved records. A second full run checks numerical reproducibility. See generated test XML, verification JSON and reproduction verification JSON for actual outcomes.

## Independent audit extension

The deeper audit uses separately implemented NumPy fits and checks every saved test vector, train/validation/test loss, coefficient norm, feature error, rank/condition diagnostic, summary mean/SD and paired comparison. It regenerates raw counts from seeds, checks complete condition/file sets and checks all recorded inputs against PennyLane. Configuration-driven plotting is tested end-to-end on changed seeds, split sizes, targets and budgets. See AUDIT_FINDINGS.md and PAPER_COMPARISON.md for the corrections and interpretation boundaries.

## Limits

Small synthetic tasks, one fixed circuit, one split size and five independent datasets limit generalization. The ridge grid is finite and validation estimates can be noisy. There is no adaptive acquisition, error mitigation, hardware noise, learned circuit or classical-versus-quantum advantage comparison. Small differences should not be described as significant. A larger circuit or additional baselines is future work, not a completed result.
