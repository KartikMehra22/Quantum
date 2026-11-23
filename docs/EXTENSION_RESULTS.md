# Measurement-aware extension: measured results

Generated from preserved fresh-seed results. This is a technical results note, not a paper draft.

20 independent dataset realizations; 5 acquisitions averaged within each dataset. Means and sample SDs are descriptive, not confidence intervals.

Ridge, DiagonalCov and FullCov each search 32 validation candidates. Ridge includes every gamma=0 option available to the new methods. DiagonalCov and FullCov share identical grids. No method uses test targets to select its penalty.

## Primary clean-label results

| Task | Shots/setting | OLS MSE | Ridge MSE | DiagonalCov MSE | FullCov MSE | FullCov reduction vs Ridge |
|---|---:|---:|---:|---:|---:|---:|
| sine | 32 | 0.045718 | 0.045066 | 0.045272 | 0.045170 | -0.23% |
| sine | 128 | 0.014218 | 0.014126 | 0.014235 | 0.014232 | -0.76% |
| sine | 512 | 0.004031 | 0.003990 | 0.003966 | 0.004004 | -0.35% |
| sine | Exact | 0.000119 | 0.000119 | 0.000119 | 0.000119 | -0.21% |
| mixed_harmonics | 32 | 0.095369 | 0.093236 | 0.093350 | 0.093780 | -0.58% |
| mixed_harmonics | 128 | 0.028590 | 0.028609 | 0.028656 | 0.028643 | -0.12% |
| mixed_harmonics | 512 | 0.009048 | 0.009101 | 0.009018 | 0.009036 | +0.71% |
| mixed_harmonics | Exact | 0.000310 | 0.000306 | 0.000310 | 0.000310 | -1.05% |

## Separate label-noise sensitivity: SD 0.1

| Task | Shots/setting | OLS MSE | Ridge MSE | DiagonalCov MSE | FullCov MSE | FullCov reduction vs Ridge |
|---|---:|---:|---:|---:|---:|---:|
| sine | 32 | 0.047200 | 0.046290 | 0.046356 | 0.046316 | -0.06% |
| sine | 128 | 0.015986 | 0.015572 | 0.015640 | 0.015504 | +0.44% |
| sine | 512 | 0.005834 | 0.005583 | 0.005624 | 0.005579 | +0.07% |
| sine | Exact | 0.001739 | 0.001562 | 0.001521 | 0.001521 | +2.64% |
| mixed_harmonics | 32 | 0.097735 | 0.095135 | 0.095544 | 0.095077 | +0.06% |
| mixed_harmonics | 128 | 0.030471 | 0.030623 | 0.030579 | 0.030516 | +0.35% |
| mixed_harmonics | 512 | 0.010968 | 0.011112 | 0.011227 | 0.011290 | -1.60% |
| mixed_harmonics | Exact | 0.002411 | 0.002168 | 0.002164 | 0.002164 | +0.18% |

## Interpretation and limitations

Positive reduction means FullCov has a lower mean MSE; negative means it is worse. Small differences and descriptive SDs do not establish significance. Exact references are reported separately; with zero measurement covariance the new methods reduce to Ridge on their coarser lambda grid.

Nonzero-gamma selection is shown separately: gamma=0 choices are not evidence of a covariance benefit. Comparisons use the same data and shot records. Equal candidate counts do not make model families equally complex.

The covariance penalty is an adaptation of an existing approach. It is not a newly invented method, an unbiased risk correction, proof of quantum advantage, or a reproduction of a different paper's forecasting results.
