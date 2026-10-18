# Figure catalog

## Figure 01: spatial gradient interventions

- Files: figures/figure-01-spatial-gradient-interventions.{pdf,svg,png}.
- Purpose: separate padding-driven shrink degradation from residual localization mismatch; compare output derivatives with parameter derivatives.
- Six panels: rows are scales 0.85 and 1.15; columns are classification output, P3 localization parameters, and one2many box-head parameters. X: six intervention arms; Y: cosine to exact, with a shared range including negative values and the exact reference. Circles/orange: Mix25; squares/blue: Online100.
- Source: cell-statistics.csv; paired-contrasts.csv supports comparisons. Means use 128 equal image weights after paired flip averaging; whiskers are 95% filename-prefix bootstrap sensitivity intervals (46 groups, 2000 draws), not seed uncertainty. One checkpoint per recipe.
- Notice: cheap padding repairs Mix25 shrink failures; Rec remains imperfect for localization; the content oracle performs well at enlargement but does not remove shrink padding corruption.
- Decision: retain cheap fill as a baseline; resolve padding/content interactions before a new trainable candidate.
- Caption requirements: hard exact assignments, frozen eval-state derivatives, no optimizer/AP result, unequal oracle area/energy, one seed and NA policy.
- Checklist: exact=1; negative P3 cosine visible; no inference from overlapping intervals; no comparison of natural-area oracle magnitudes as causal importance.

## Figure 02: regional error budget

- Files: figures/figure-02-regional-error-budget.{pdf,svg,png}.
- Purpose: audit whether global feature MSE predominantly measures blank-space error.
- Panels a/b: x = DINO layers 3/7/11; y = mean per-image fraction of Transport SSE, stacked to 100%, at scales 0.85/1.15. Five distinct color/hatch regions. Panel c: x = six arms; y = summed-layer global relative MSE; scale indicated by color/shape, prefix-bootstrap intervals.
- Source: regional-statistics.csv and feature-statistics.csv. Stack heights are descriptive compositions; per-region uncertainty is available in CSV rather than overlapping stacked whiskers.
- Notice: padding dominates error at shrink scale; no new padding at enlargement. Feature-error gains alone do not establish task recovery.
- Decision: report region-normalized error and task-sensitive derivatives alongside global MSE.
- Caption requirements: pixel/feature support mask, mixed boundaries separate, regional sizes unequal, feature results shared by both heads, no AP claim.
- Checklist: stacks sum to 100; foreground is any visible GT-fragment patch overlap; zero-energy regional ratios remain undefined; template build cost is charged once.
