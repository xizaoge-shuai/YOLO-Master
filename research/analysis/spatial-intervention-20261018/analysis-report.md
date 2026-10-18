# Spatial intervention analysis, 2026-10-18

This is a completed frozen-head diagnostic on 128 selected **training** images, with one seed-0 checkpoint per head recipe, six shared feature arms and six views. It is neither a new AP experiment nor independent seed/backbone/dataset replication. No parameters or buffers changed, no optimizer updates occurred, and native-loss parity error is zero. All downloaded raw hashes and row invariants passed.

## Questions and decisions

1. Does new padding explain shrink-view degradation? The paired padding oracle and cheap blank fill substantially repair the Mix25 classification and localization gradient failures at scale 0.85. This establishes sensitivity to the padding intervention in these frozen heads; it does not establish a general training cause or an AP gain.
2. Is the learned corrector doing more than filling blank space? Classification output alignment is nearly matched by blank fill, but Rec retains some parameter-gradient advantages. Output and parameter gradients must not be conflated. A cheap fill baseline is now mandatory in candidate training.
3. Where is the remaining localization opportunity? At scale 1.15, content replacement greatly improves localization parameter alignment; blank fill does essentially nothing because no new padding is created. At scale 0.85, content-only replacement leaves corrupted padding and can retain negative Mix25 P3 alignment. The next spatial study needs a padding-by-content factorial control before equal-token/equal-norm foreground/background interventions.

## Exact numerical summary

Mix25, one2many; mean of the two flip views within each image, then equal image means. Gradients use hard targets from the exact feature assignment. A cosine of one is the exact reference. Localization includes the reg_max=1 auxiliary normalized ltrb L1 term.

| Scale | Arm | Class output cosine | P3 box param cosine | Box-head param cosine |
|---|---|---:|---:|---:|
| 0.85 | Transport | 0.504042 | -0.224257 | 0.019620 |
| 0.85 | Blank fill | 0.972895 | 0.435897 | 0.370356 |
| 0.85 | Pad oracle | 0.973236 | 0.525524 | 0.405946 |
| 0.85 | Rec | 0.974375 | 0.505889 | 0.408537 |
| 0.85 | Content oracle | 0.486333 | -0.180066 | 0.145379 |
| 0.85 | Exact | 1.000000 | 1.000000 | 1.000000 |
| 1.15 | Transport | 0.965050 | 0.542088 | 0.439836 |
| 1.15 | Blank fill | 0.965050 | 0.542088 | 0.439836 |
| 1.15 | Pad oracle | 0.965050 | 0.542088 | 0.439836 |
| 1.15 | Rec | 0.965612 | 0.553150 | 0.453441 |
| 1.15 | Content oracle | 0.998727 | 0.954236 | 0.944803 |
| 1.15 | Exact | 1.000000 | 1.000000 | 1.000000 |

## Paired effect summaries

- mix25, scale 0.85, output_cls_cosine, blank_fill-transport: +0.468852; descriptive 95% prefix-bootstrap interval [+0.433472, +0.503962].
- mix25, scale 0.85, output_cls_cosine, rec-blank_fill: +0.001481; descriptive 95% prefix-bootstrap interval [+0.001003, +0.001942].
- mix25, scale 0.85, p3_box_cosine, rec-blank_fill: +0.069992; descriptive 95% prefix-bootstrap interval [+0.016629, +0.125894].
- mix25, scale 0.85, p3_box_cosine, rec-padding_oracle: -0.019634; descriptive 95% prefix-bootstrap interval [-0.047057, +0.007582].
- online-geom, scale 0.85, p3_box_cosine, blank_fill-transport: -0.068777; descriptive 95% prefix-bootstrap interval [-0.125353, -0.011211].
- mix25, scale 1.15, p3_box_cosine, content_oracle-transport: +0.412148; descriptive 95% prefix-bootstrap interval [+0.334957, +0.499376].

## Interpretation and limits

- New padding occupies 444/1600 tokens (27.75%) at scale 0.85. Its mean shares of Transport SSE in layers 3, 7 and 11 are approximately 94.68%, 89.56% and 83.24%. A large global MSE reduction can therefore arise mainly outside valid image content; it is not evidence of recovered object geometry by itself.
- Feature errors are shared by the two heads, not two independent replications. Natural-area oracles are unequal in area/energy and require exact teacher features. They are diagnostic donors, not deployable methods or fair efficiency competitors.
- Cheap fill is not uniformly beneficial: at scale 0.85 it decreases Online100 P3 localization-parameter cosine by 0.068777 versus Transport (descriptive prefix-bootstrap interval [-0.125353, -0.011211]). Its Mix25 benefit must not be generalized to all heads or metrics.
- Blank fill uses one additional gray-image teacher query per stage and a 7,372,800-byte FP32 template. Its fill includes fractional interpolation support, whereas the padding oracle replaces only pure added-padding tokens; these masks are not identical.
- The global feature statistic here is summed-layer SSE divided by summed-layer reference energy; do not merge its values with earlier equal-layer normalized-MSE summaries or different image subsets.
- Views are not independent samples. One checkpoint per recipe, correlated image filenames and previously seen training images prevent population/seed significance claims. No p-values, statistical winner claims or equivalence claims are made. All intervals are descriptive, pointwise resampling sensitivity, not simultaneous confidence bands.
- Exact assignment is held fixed for gradient comparisons; native self-assigned losses and matching are separately recorded. These are frozen eval-state derivatives, not optimizer update trajectories. One2one shared gradients remain disconnected and undefined comparisons stay NA.
- Official AP, small-object AP, equal-time training, robustness and cross-domain/model claims remain untested by this campaign. Previous negative training results remain valid.
- Development took 1926.39 seconds (32.11 minutes); the initial smoke-based 3-4-hour estimate was too high. Timing changed substantially during execution; do not use this diagnostic runtime as a method efficiency benchmark.

## Claim candidates

- Claim: Shrink-view transport error is dominated by added padding in these cached features.
  - Source evidence: regional-statistics.csv, Transport scale 0.85, all three layers; figure 02a.
  - Allowed wording: New padding accounts for 83-95% of mean per-image Transport SSE at the tested shrink scale.
  - Forbidden stronger wording: Padding explains all detector degradation or all transforms.
  - Uncertainty: One backbone, training subset; no area/energy matching.
  - Next check: Padding-by-content factorial interventions, followed by matched residual controls.
  - Decision: keep, narrowly.
- Claim: A cheap blank template nearly matches Rec's classification output alignment here.
  - Source evidence: cell-statistics.csv and paired-contrasts.csv; figure 01a.
  - Allowed wording: The observed means are close, while Rec retains a small positive paired difference.
  - Forbidden stronger wording: The methods are statistically equivalent or achieve the same AP.
  - Uncertainty: No equivalence margin or training comparison; parameter gradients differ.
  - Next check: Include blank fill as a budget-matched training baseline after diagnostic controls.
  - Decision: revise the earlier broad representation-recovery story.
- Claim: Content restoration exposes localization headroom at enlargement.
  - Source evidence: P3 and box-head parameter comparisons at scale 1.15; figure 01e-f.
  - Allowed wording: The content oracle improves local gradient alignment under exact assignments.
  - Forbidden stronger wording: A learned content corrector already exists, is novel or will improve AP.
  - Uncertainty: Teacher oracle, unequal region energy, fixed eval heads; shrink-view interaction remains unresolved.
  - Next check: Factorial padding/content control, then equal-token/equal-norm residual interventions with padding fixed.
  - Decision: keep as a method-development lead.
