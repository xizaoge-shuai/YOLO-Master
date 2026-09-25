# Statistical appendix

- Best AP is selected on repeatedly inspected pilot validation; final AP also reported.
- Three seeds do not establish dataset-level uncertainty; no method superiority or equivalence claim.
- t95 and paired dz assume an adequate paired-difference distribution; n=3 cannot establish normality.
- Exact sign-flip assumes paired-difference symmetry/exchangeability; smallest two-sided p is .25.
- All eight metric/reference/budget contrasts form one exploratory Holm correction family.
- Reconstruction MSE is pre-update on queried TRAINING images, not held-out feature recovery.
- Raw predictions not reevaluated; existing custom D1 metric is not official full-dataset AP.
- Timing anomaly retained; no claim of corrector speedup at p25.

| Budget | Ref | Metric | Paired delta | Assumption-dependent t95 | Exact p | Holm p | Positive seeds |
|---:|---|---|---:|---|---:|---:|---:|
| 10 | mix | best | +0.1794 | [-0.9765, +1.3354] | 0.500 | 1.000 | 2/3 |
| 10 | mix | final | +0.4559 | [-0.6610, +1.5727] | 0.500 | 1.000 | 2/3 |
| 10 | transport | best | -0.2270 | [-0.4616, +0.0076] | 0.250 | 1.000 | 0/3 |
| 10 | transport | final | -0.1067 | [-0.9167, +0.7032] | 0.750 | 1.000 | 1/3 |
| 25 | mix | best | +0.2556 | [-1.0048, +1.5159] | 0.500 | 1.000 | 2/3 |
| 25 | mix | final | +0.3875 | [-0.7863, +1.5613] | 0.500 | 1.000 | 2/3 |
| 25 | transport | best | -0.1116 | [-0.7163, +0.4932] | 0.750 | 1.000 | 1/3 |
| 25 | transport | final | +0.0391 | [-0.6403, +0.7185] | 1.000 | 1.000 | 2/3 |

Do not count epochs as independent repetitions or interpret p>.05 as equivalence.
Absolute paired AP-point differences are the main effect size; paired Cohen dz is in analysis.json.
No normality claim follows from a low-power n=3 normality test.
Archived hashes and arithmetic consistency passed; this does not rerun prediction-based evaluation.
