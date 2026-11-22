# Deep inspection: 22 November 2026

## Outcome

The numerical conclusions survive the deeper inspection. This is a two-method simulation comparison; it does not reproduce the cited papers' numerical findings. The original baseline remains at commit `9414b8b`. The post-audit rerun matches its 640 metric rows, 640 test prediction vectors and 100 raw arrays numerically, with maximum absolute metric difference 0.0 in the recorded environment. See `results/semester/baseline_comparison.json`.

## Weaknesses found and addressed

1. **The original audit reused the experiment's fitting helper.** This established repeatability but could share a fitting bug. Added a separate NumPy implementation of scaling, centered OLS, regularized linear solves, intercept handling and validation-grid selection. It checks all predictions and train/validation/test losses, not just test MSE.
2. **Counting two rows per group was weaker than checking method identities and the full design.** Added unique condition-key checks, exact expected coverage, explicit OLS/Ridge identities, matching prediction keys and rejection of missing/extra raw files.
3. **Some result fields and derived summaries were not independently audited.** Added coefficient norms, feature errors, alpha scaling, rank and condition numbers, split budgets, all 32 summary groups and all 16 paired comparisons.
4. **The original runner could reuse a directory from an incomplete run.** The CLI now refuses any nonempty destination and validates tasks, feature order, dimensions, split sizes, seeds, shot counts, lambda values, label-noise levels and domain bounds before starting.
5. **Figures hard-coded the original seeds, splits and budgets.** They now use the saved configuration. An integration test changes tasks, seeds, repetition counts, split sizes and budgets, then runs the model, plots and independent audit end-to-end.
6. **Uncertainty wording hard-coded five datasets.** It now describes dataset-level aggregation without assuming a fixed repetition count. This description-only change to the package is recorded by a fresh metadata/source manifest; the study design and numerical methods are unchanged.
7. **An augmented minimum-norm OLS verifier would use a different intercept convention in underdetermined cases.** The independent implementation now centers X and y explicitly, with a test of rank-deficient features. The canonical study is full rank and its results were unaffected.

8. **The new deep auditor initially demanded exact regenerated floating-point inputs across architectures.** Linux reported a maximum difference of 4.44e-16 from the saved macOS inputs. Recomputed floating inputs and labels now use rtol=1e-13 and atol=1e-14, with a test that rejects a 1e-6 change and nonfinite values. Saved-file hashes and integer outcome counts remain exact. No data or model calculations were changed by this portability fix.

## Additional evidence

- 26 automated tests passed, including negative cases that must fail.
- PennyLane independently checked Born probabilities for all 1,120 recorded inputs across the three settings.
- All 75 joint-count acquisitions regenerate from their seeds.
- Both the original fitted-result audit and independent linear-algebra audit pass.
- A fresh post-audit reproduction matches all 640 metric rows, all 640 prediction vectors and all 100 raw arrays.
- The default experiment retains five independent dataset realizations, each with five acquisition repeats. The 640 fits are not 640 independent statistical trials.

## Scientific interpretation

All default augmented feature matrices have rank 10. Exact-feature condition numbers are approximately 82.23-112.50; finite-shot values are approximately 5.15-25.54. Sampling therefore does not create the rank-changing instability described in some motivating papers in this particular setup. Smaller condition numbers alone do not imply better predictive accuracy.

Higher shots reduce primary mean prediction error here. Ridge gives small primary mean improvements in five of six finite-shot task/budget conditions and worsens one. Neither significance nor universal superiority is established. See `PAPER_COMPARISON.md` for the checked boundaries against each reference.

## Honest progress and version history

The 16 November discussion established the semester scope and draft introductory material. The implementation baseline and the deeper audit were created on 22 November. Git commits record actual work dates. No code development is claimed for unrecorded dates, and no historical commit timestamps are fabricated to satisfy a count target.

The unit tests and audits are strong evidence about the cases checked, not a guarantee that every possible error or limitation has been eliminated. Paper drafting remains deferred.
