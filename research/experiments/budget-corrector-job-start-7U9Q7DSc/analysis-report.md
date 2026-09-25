# Paired-budget pilot analysis

Question: does reconstruction-only online correction improve detection beyond
using the same fresh features with C2-Mix or C2-Transport?

| Method | Best AP mean +/- SD | Final AP mean +/- SD | Train s mean | Train s median |
|---|---:|---:|---:|---:|
| c2-mix-p10 | 3.6920 +/- 0.2844 | 3.2369 +/- 0.2337 | 296.0 | 295.7 |
| c2-transport-p10 | 4.0984 +/- 0.1540 | 3.7995 +/- 0.1007 | 291.9 | 289.9 |
| c2-correct-p10 | 3.8714 +/- 0.1812 | 3.6928 +/- 0.2316 | 307.2 | 307.7 |
| c2-mix-p25 | 4.1635 +/- 0.2744 | 3.8934 +/- 0.3361 | 407.1 | 412.8 |
| c2-transport-p25 | 4.5306 +/- 0.1011 | 4.2418 +/- 0.2094 | 492.0 | 421.4 |
| c2-correct-p25 | 4.4190 +/- 0.2736 | 4.2809 +/- 0.2778 | 425.6 | 428.2 |

The current corrector has not established an advantage over Transport. Best-AP
differences are negative for all three p10 seeds and two of three p25 seeds.
Final AP is negative on average at p10, almost tied on average at p25.
Mean improvements over Mix alone do not isolate a benefit beyond geometric transport.

Training-query reconstruction improves substantially: the final-ten-epoch relative
MSE reduction is about 52-54% at p10 and 62% at p25. This supports successful fitting
of the dense reconstruction objective on training queries, not held-out feature
generalization or task-relevant recovery. Objective mismatch, background dominance,
and a changing corrected-feature distribution are hypotheses to test, not findings.

Transport p25 seed1 has an anomalous training AND validation slowdown. Its 312.7 s
validation total contrasts with about 66 s elsewhere. The cause is unobserved.
Keep its AP and raw timing; do not report the aggregate 0.902 paired timing ratio
as an algorithmic speedup. A fresh interleaved timing control with resource telemetry
is required. Median timing is a sensitivity check only.

## Claim candidates

- Claim: reconstruction-only correction has no demonstrated AP advantage over matched Transport here.
  - Evidence: 18 raw 100-epoch CSVs, paired differences in analysis.json, budget-comparison figure.
  - Allowed: no consistent advantage on this fixed small pilot.
  - Forbidden: correction can never work, or the methods are statistically equivalent.
  - Uncertainty: three seeds and 100 development-validation images.
  - Next check: object-aware/task-aware diagnostics and a full VisDrone pilot.
  - Decision: keep.
- Claim: dense feature error falls while detector gains fail to follow.
  - Evidence: correction.jsonl and reconstruction-and-timing figure.
  - Allowed: a training-query feature/detection mismatch motivates a task-aware hypothesis.
  - Forbidden: proven foreground suppression or proven out-of-domain overfitting.
  - Uncertainty: no foreground/background or held-out feature decomposition yet.
  - Next check: matched unseen-image/augmentation feature probes, object-size and fixed-head analyses.
  - Decision: keep with qualifier.
- Claim: Correct is faster than Transport at p25.
  - Evidence: aggregate is dominated by a slowdown in one baseline run.
  - Allowed: timings need controlled remeasurement.
  - Forbidden: a 10% algorithmic speedup.
  - Uncertainty: no contemporaneous resource telemetry.
  - Next check: retain old data and repeat matched timing controls.
  - Decision: discard as a method claim.

## Decisions

Begin full-dataset loader/evaluator work and full VisDrone baseline profiling now.
Do not postpone those prerequisites until the corrector wins. Include this current
corrector as a diagnostic control in the first full VisDrone seed, then expand only
a short list. Limit additional small-pilot method searches to two specified changes:
object-balanced reconstruction, then the same loss plus task-aware supervision.
The raw archive is unchanged. No new training was launched by this analysis.
