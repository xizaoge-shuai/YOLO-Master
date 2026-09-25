# Experiment integrity review

- Reviewer: GPT-5.6-Sol ultra, fresh read-only agent /root/audit_budget_results.
- review_independence: same-family
- acceptance_status: provisional
- Overall verdict: WARN
- Scope: the 18-run budget-corrector-job-start-7U9Q7DSc pilot only.
- No retraining or independent AP recomputation from saved predictions was performed.

## A. Ground truth provenance: WARN

Detection uses dataset YOLO labels, not generated labels. The cache binds image
paths to VisDrone; d1_train_cached_detector.py:105-175 resolves and reads real
annotations, and :320-367 compares predictions with these labels. The view-cache
manifest hashes configuration and label files (:531-540). However, this is the
custom D1 COCO-style evaluator, not the official VisDrone protocol or pycocotools
(reference_args.json:17,21,40-42). Real-ground-truth provenance passes; official
benchmark claims are not supported.

## B. Score normalization: PASS

budget_corrector.py:415-421 multiplies raw AP by 100 for reporting. The active
metrics implementation integrates precision-recall over IoU 0.50:0.95
(ultralytics/utils/metrics.py:768-875,990-1000). No prediction-derived denominator
normalizes detection scores. Correction relative MSE uses teacher-target energy
(budget_corrector.py:76-83), a distinct model-teacher proxy.

## C. Results existence/arithmetic: PASS

All 18 planned runs have DONE and 100 finite epoch records. Summary best/final
values, comparison.json and displayed paired differences match the CSVs.
All 201 archive/source evidence hashes checked by the reviewer matched.
Plans were not counted as completed full-data results.

## D. Called metrics: PASS with provenance caveat

Worker -> patched pilot -> validation -> match_predictions -> DetMetrics is
active (budget_corrector.py:380-394; d1_online_compare.py:369-384;
d1_train_cached_detector.py:295-367; metrics.py:1143-1180,1223-1227).
No claimed metric was found to be dead code.

metrics.py is absent from the explicit source-hash list (provenance.json:3-17).
The reviewer verified that it matches the recorded Git commit (:20).
Future experimental manifests must hash the metric implementation directly.
Do not rewrite this old provenance to suggest it was originally hashed.

## E. Scope and fairness: WARN

- One fixed 400/100 split of 500 VisDrone training images, three training seeds,
  one split seed. Train/validation IDs are disjoint.
- Recomputing raw schedules confirmed identical schedules/query masks in all
  six budget-seed groups; summary checks are at budget_corrector.py:435-440.
- Validation uses original val_bank, no corrector or calibration
  (budget_corrector.py:86-102).
- Best AP is selected over 100 evaluations on the same 100-image development
  validation set (d1_online_compare.py:377-429), not unbiased held-out performance.
- Mix does not realize center-scale geometry on nonqueries; Correct vs Transport
  is the cleaner ablation (budget_corrector.py:202-210).
- Transport p25 seed1 has anomalous train/validation timing; the reported 0.902
  paired mean ratio does not establish a correction speedup. Correct is slower
  in two of the three individual pairs.

## F. Evaluation classification: PASS

- Detection AP: real_gt, custom pilot evaluator.
- Correction MSE: self-supervised/model-teacher proxy on queried training features.
- Online/cache parity: self-supervised consistency proxy.

## Claim impact

- Supported with scope qualifier: Correct has not shown a consistent detection
  advantage over same-budget Transport in this pilot.
- Supported with scope qualifier: Correct fits queried TRAINING teacher features.
- Unsupported: held-out/domain-transfer feature recovery from these MSE logs alone.
- Unsupported: algorithmic p25 timing speedup from the anomalous aggregate.
- Unsupported: official/full-data/general-model performance.

## Actions

1. Keep Transport as the primary correction reference; report both best and final.
2. Add held-out feature/object-size diagnostics before attributing a cause.
3. Remeasure matched timing with telemetry, retaining the old AP and time records.
4. Use official full splits/evaluation before benchmark claims; freeze test choices.
5. Directly fingerprint metrics.py and prediction/postprocessing code in new runs.

This review is same-family and provisional, not an independent third-party
replication. Archived artifacts do not include prediction/weight files; this
audit did not independently rerun detection evaluation.
