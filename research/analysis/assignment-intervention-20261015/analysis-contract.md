# Analysis contract, fixed before numerical contrasts/figures

- Question: under the same frozen detector, how much of raw-output gradient
  disagreement changes when the complete assignment or only hard matching is
  transplanted? Does reconstruction error improvement track localization-gradient repair?
- Source: completed 128-training-image development and 548-validation-image
  confirmation; six views; one checkpoint seed; Transport and Rec. Preserve all cells.
- Primary contrasts: AE minus AA gradient cosine under hard_common, separately
  classification/box and one2many/one2one, both methods. Native-soft is a separate
  sensitivity analysis. Also retain EE/EA and relative gradient errors.
- Sampling unit: first average views within each image. No independent-view counts.
  Missing metrics are excluded only when mathematically undefined, with counts.
- Uncertainty: report descriptive image SD and paired 2,000-draw image-bootstrap
  intervals. Because image filenames share prefixes, also provide cluster-resampling
  sensitivity using the first filename component, maintaining image-weighted means.
  Prefixes are not verified sequence metadata; neither interval captures seed
  variability or establishes population significance. No p-values or superiority claim.
- Figure 1: quantitative 2x2 panel grid, two loss branches by classification/box
  cosine; horizontal categories EE, EA, AE, AA; Transport blue, Rec orange.
  Hard-common is primary. Error bars are image-bootstrap reweighting intervals,
  explicitly not training-seed confidence intervals. Main conclusion to check:
  assignment transplant repairs classification agreement more than localization.
- Figure 2: quantitative 2x2 grid over scale: global relative feature MSE;
  one2many foreground Jaccard; one2one foreground Jaccard; small-target coverage
  gap (approximate minus exact) in both branches. Identity scale is retained as a control.
- Python/matplotlib is the saved backend. Export editable PDF/SVG and 300dpi PNG
  at 180mm width, roughly 130mm height; source data CSV and reproducible script.
  No generated/illustrative data. Preview exports and run source figure QA.
- Claim limits: frozen-head output-space gradients are not parameter updates or
  AP. Unit-positive BCE gradients necessarily become more similar when target labels
  are shared, so classification alignment alone is not a novel causal explanation.
  One2one feature detachment does not invalidate output-space diagnostics, but
  prevents interpreting them as backbone/adapter gradients.
- Next gate: independent frozen-head replication using the completed Mix25 head,
  same 548 images/labels/views, before designing a training intervention.
