# Full VisDrone final-checkpoint offline evidence

Eight methods, each scored on the same 548 development images. All checkpoints are fixed final epoch100, training seed0. No model retraining or GPU inference was performed.

| Method | Official AP | Auxiliary AP-small | Auxiliary AR500-small |
|---|---:|---:|---:|
| transport25 | 9.6350 | 3.0694 | 11.3691 |
| blank25 | 9.9984 | 3.1329 | 11.2440 |
| views25 | 10.0912 | 3.4000 | 11.8379 |
| detail25 | 12.3453 | 5.4259 | 17.0169 |
| lowpass25 | 9.8509 | 3.1494 | 11.4258 |
| mix25 | 11.3526 | 4.2209 | 14.1763 |
| online100 | 11.3982 | 4.2094 | 14.5860 |
| vit-adapter-cache25 | 16.5855 | 9.0101 | 23.9035 |

Official AP is the pinned VisDrone occurrence-weighted metric. Size metrics use original-pixel COCO area ranges, uniform category averaging and COCO 101-point sampled precision, after the frozen VisDrone ignored-region prefilter. They are auxiliary custom metrics, not official VisDrone AP_S/AP_M/AP_L. The exact inclusive area boundaries and global500 export limit are in protocol.json.

## Supported observations

- Detail25 minus blank25: official AP +2.3469; auxiliary small AP +2.2930; auxiliary small AR500 +5.7728.
- Detail25 minus lowpass25: official AP +2.4943; auxiliary small AP +2.2765; auxiliary small AR500 +5.5911.
- Detail25 minus mix25: official AP +0.9927; auxiliary small AP +1.2050; auxiliary small AR500 +2.8406.
- Detail25 minus online100: official AP +0.9471; auxiliary small AP +1.2165; auxiliary small AR500 +2.4308.
- Detail25 minus vit-adapter-cache25: official AP -4.2402; auxiliary small AP -3.5843; auxiliary small AR500 -6.8866.

For Detail25 versus Blank25, the medium AP delta is +0.9358 and large AP delta is -0.6321. Thus the improvement is not uniform across object sizes. Adapter25 is stronger than Detail25 on overall and small-object AP.

## TIDE AP50 diagnostics

These are genuine tidecv1.0.1 error-fixing oracles at foregroundIoU0.5/backgroundIoU0.1 with max500. Oracle dAP values are independent and cannot be added into a decomposition. Ignore-region preprocessing and score-zero GT crowd handling are explicit. TIDE AP50 is an auxiliary macro metric, not the official VisDrone AP50.

| Error | Blank25 dAP50 | Detail25 dAP50 | Detail-minus-Blank | Blank count | Detail count |
|---|---:|---:|---:|---:|---:|
| Cls | 10.5402 | 13.2057 | +2.6655 | 33232 | 46993 |
| Loc | 7.8006 | 7.7085 | -0.0921 | 55116 | 40039 |
| Both | 0.8942 | 1.0034 | +0.1092 | 49310 | 43294 |
| Dupe | 1.0456 | 1.1245 | +0.0789 | 20601 | 20750 |
| Bkg | 0.7393 | 0.9438 | +0.2045 | 66504 | 68388 |
| Miss | 6.8354 | 5.2448 | -1.5907 | 8260 | 6232 |

Detail25 reduces missed-error count and Miss oracle dAP, and reduces raw localization-error count. The Loc oracle dAP changes only slightly, while classification and background oracle dAP increase. The data support improved recall and some spatial accuracy, not a blanket reduction of every error type. Raw counts use all exported scores>=.001 and exclude TIDE objects whose prediction is ignored; library totals including ignored objects are retained in each method JSON. Miss excludes GT corrected by other error oracles and is not the total false-negative count.

## Geometry quality and provenance

Detail25 contains 37 original saved predictions with negative width/height; 29 remain after the frozen ignored-region prefilter. Maximum original confidence is 0.03757677. The per-method invalid_prediction_geometry field refers to the prepared evaluator input, not the unfiltered raw export. Original prediction files are untouched. Official-reference recomputation uses the original inputs and exactly matches every saved endpoint within1e-9 AP points. For auxiliary COCO/TIDE only, negative sizes become zero; score, class, coordinates and number of predictions are preserved, so they remain degenerate FP candidates. CPU contracts verify this treatment. The initial strict-loader failure and its source hash are retained in previous-failure.json.

Each method JSON binds the final checkpoint, all548 prediction hashes, all548 original annotation hashes and training-curve source. Full original hash maps remain in the read-only original official-final files; compact sorted-map SHA256 values are stored here.

## Boundaries and next measurements

- There is one independent training seed per method. No confidence interval, p value, or robustness-over-seeds claim is supported by this analysis.
- Development validation is not a held-out test-dev submission.
- train_AP is missing because the training set was not separately scored. diagnostic_AP in training-curves.csv is internal macro validation, not official reference AP.
- Source runs have mixed shared-GPU timing conditions; convergence-versus-time curves do not establish a controlled training speedup.
- GT-only difficulty cohorts are fixed before feature inspection. They provide qualitative sample identities, not a new benchmark subset.
- Fourier/P3 feature analysis, full fresh-RGB inference latency and geometric robustness of the current four candidate models remain separate pending model-evaluation stages.
