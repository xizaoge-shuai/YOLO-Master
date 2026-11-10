# Figure catalog

All four figures are analysis drafts from the accompanying exact CSV values, not final manuscript layout. Eight final-epoch seed0 models; n=1 training replicate each; no error bars or inference tests.

- size-comparison: panels auxiliary AP and AR500 by original-pixel size, from size-metrics-auxiliary.csv. Supports size-specific comparisons; not official VisDrone size AP.
- tide-errors: independent error-oracle dAP50 and separate nonignored raw error counts, from tide-errors-auxiliary.csv. The oracles do not sum; raw counts depend on the shared export threshold.
- per-class-ap: pinned-reference class AP heatmap from per-class-metrics.csv. Same color scale across all methods.
- training-curves: train loss, val loss and internal diagnostic AP against epoch, from training-curves.csv. Sparse validation is not interpolated into official AP; no train AP was measured.

Exports include editable SVG/PDF and300dpi PNG. Current widths favor inspection of eight methods, not journal column dimensions; no TIFF is needed for these drafts. Font/SVG/DPI are configured. Source values, protocols, masks and labels must survive any later manuscript rearrangement.
