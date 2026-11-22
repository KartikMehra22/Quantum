# Measured results

Generated directly from recorded result JSON. This is a numerical note, not a paper draft.

Test MSE is against the noiseless target. Values are averaged within dataset first, then across five dataset means. Sample SD bars are descriptive, not confidence intervals.

## Primary clean-label study

| Task | Shots per setting | OLS MSE | Ridge MSE | Ridge relative MSE reduction |
|---|---:|---:|---:|---:|
| mixed_harmonics | Exact | 0.000292 | 0.000312 | -6.98% |
| mixed_harmonics | 32 | 0.085070 | 0.089468 | -5.17% |
| mixed_harmonics | 128 | 0.027094 | 0.026630 | +1.71% |
| mixed_harmonics | 512 | 0.008182 | 0.008088 | +1.15% |
| sine | Exact | 0.000115 | 0.000118 | -2.72% |
| sine | 32 | 0.046958 | 0.046050 | +1.93% |
| sine | 128 | 0.014806 | 0.014654 | +1.02% |
| sine | 512 | 0.003988 | 0.003874 | +2.88% |

## Separate noisy-label sensitivity check

| Task | Shots per setting | OLS MSE | Ridge MSE | Ridge relative MSE reduction |
|---|---:|---:|---:|---:|
| mixed_harmonics | Exact | 0.002768 | 0.002829 | -2.23% |
| mixed_harmonics | 32 | 0.085566 | 0.089452 | -4.54% |
| mixed_harmonics | 128 | 0.029033 | 0.028277 | +2.61% |
| mixed_harmonics | 512 | 0.010512 | 0.010157 | +3.38% |
| sine | Exact | 0.001376 | 0.002108 | -53.19% |
| sine | 32 | 0.048535 | 0.046985 | +3.19% |
| sine | 128 | 0.015269 | 0.015063 | +1.35% |
| sine | 512 | 0.005281 | 0.005302 | -0.41% |

Positive relative reduction means lower MSE for Ridge; negative means Ridge was worse. Percentages near a small exact-reference baseline can look large even when the absolute difference is small.

## Validation and accounting

- 640 fitted models: 600 finite-shot and 40 exact references, not 640 independent datasets.
- 75 unique acquisitions; 11,289,600 simulated executions shared across both models, targets, label conditions and tuning candidates.
- Main simulation/fit runtime 0.97 seconds in the recorded environment, excluding setup, tests, figures and documentation; not a QPU runtime estimate.
- See verification.json and reproduction_verification.json for the independent reconstruction and fresh-run comparison.

## Interpretation

Mean test MSE decreases with more shots for both models on both tasks in the primary study. Ridge improves five of six finite-shot task/budget means, usually modestly, and worsens mixed harmonics at 32 shots. The secondary noise condition also gives mixed results. This does not establish statistical significance, general superiority, a new method or quantum advantage.
