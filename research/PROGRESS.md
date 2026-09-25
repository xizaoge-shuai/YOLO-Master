# D1 experiment progress

## 2026-09-25: cached-feature augmentation baselines

- Completed: 10 configurations, 3 seeds each, 100 epochs, fixed 400/100 split.
- Actual head: mean_value; identical unaugmented validation cache.
- Strong cheap reference: two real cached views + geometric feature transport.
- Mean best AP: C1 2.5694; C2 3.6405; Transport C1 3.1775;
  Transport C2 3.7666; LOFF-TA-style C2 3.7675.
- Noise adds approximately 0.0009 mean best AP over Transport C2 in this pilot;
  this is not evidence of a reliable improvement.
- Next: 10%/25% matched-backbone-query C2-Mix / C2-Transport / C2-Correct.
- Source code and experimental evidence are recorded in separate commits.
- [Evidence and analysis](experiments/feature-baselines-20260925-job-6JnMcLAf/README.md).

Earlier online, fusion, mean-sweep and flip-cache runs remain historical
references. Their helper scripts are retained as run; this entry does not
claim that their weights/predictions have been independently re-evaluated.

## 2026-09-25: fixed-budget correction implementation and full-data audit

- Added six C2 cases: Mix / Transport / reconstruction-only Correct at 10% and 25%.
  Formal campaign: 100 epochs, seeds 0/1/2; fixed 400/100 pilot.
- Training queries have paired identities, transforms and masks across arms.
  Actual teacher images, setup parity, correction time, checkpoints and schedule hashes are logged.
- Background launcher runs smoke before training and archives/commits lightweight progress on completion or failure.
- Validation: 17 related tests passed; new-file Ruff and launcher syntax checks passed.
  Repository-wide checks still report 2,768 existing Ruff issues, five format mismatches;
  codespell is not installed. Evidence: validation/budget-corrector-20260925/.
- GPU smoke/full campaign had not been run at this implementation commit.
- User confirmed target reference is YOLO-Master CVPR 2026. Paper recipe and differences:
  [benchmark protocol](YOLO_MASTER_BENCHMARK_PROTOCOL.md).
- Full VisDrone files: 6,471 train / 548 val / 1,610 test-dev; one zero-height training box
  requires an explicit derived-annotation exclusion before full training.
- COCO: 5,000 val images and annotations present; training images/JSON absent despite train list.
  Full VisDrone caches require streaming; COCO C1 raw features alone require about 406 GiB.

## Fixed-budget correction: job-smoke-1bpNRqmT

- Status: FAILED exit=1; inspect job.log and worker log
- Evidence: [archive](experiments/budget-corrector-job-smoke-1bpNRqmT/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-smoke-1bpNRqmT
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

## Fixed-budget correction: job-smoke-ajfAh00N

- Status: SMOKE_SUCCEEDED
- Evidence: [archive](experiments/budget-corrector-job-smoke-ajfAh00N/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-smoke-ajfAh00N
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

- Post-smoke verification: all six raw schedules match their saved hashes; query schedules
  match across arms. Corrector checkpoints contain trained weights and optimizer state
  (50,560 parameters; 5/12 updates at p10/p25). Formal 18-run campaign not launched here.
- Pilot provenance: all 500 images come from official VisDrone training, internally split 400/100.

## Fixed-budget correction: job-start-7U9Q7DSc

- Status: SUCCEEDED
- Evidence: [archive](experiments/budget-corrector-job-start-7U9Q7DSc/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-start-7U9Q7DSc
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

## 2026-09-25: paired-budget results audited; full-data roadmap fixed

- Source job: runs/paper/budget-corrector/job-start-7U9Q7DSc; raw archive commit 8d01693.
- Verified 201 archived evidence hashes and 18 complete 100-epoch runs. Local/server analysis
  matches in 256 numeric fields to absolute/relative tolerance 1e-12. No AP rerun from predictions.
- Correct minus Transport best AP: p10 -0.2270, p25 -0.1116;
  final AP: p10 -0.1067, p25 +0.0391. Current reconstruction-only correction has no
  demonstrated stable AP advantage over same-budget Transport.
- Last-ten-epoch training-query relative MSE reductions: about 52-54% (p10), 62% (p25).
  This establishes training-query fit only; held-out feature utility remains unmeasured.
- Transport-p25-s1 slowdown affects training and validation; keep all original results.
  No algorithmic speedup claim from the aggregate 0.902 ratio. Remeasure with telemetry.
- Fresh GPT-5.6-Sol ultra reviewer: same-family/provisional WARN, not independent replication.
  AP GT and query pairing pass; custom pilot scope, best-on-validation selection,
  timing anomaly and explicit metrics.py hash omission require qualifiers/actions.
- Analysis, statistical appendix, figures and audit:
  [budget analysis](experiments/budget-corrector-job-start-7U9Q7DSc/analysis-report.md).
- [Next experiments](NEXT_EXPERIMENTS_AFTER_BUDGET_CORRECTOR.md): start full VisDrone
  streaming/evaluation/baseline preparation now; first full seed includes the current corrector
  as a diagnostic control. One bounded held-out/object-aware diagnosis, then selected
  COCO and second-backbone studies. No new GPU training launched in this analysis turn.
- New analysis Ruff checks pass. Repository-wide pre-existing 2,768 Ruff issues and five
  format mismatches persist; codespell remains unavailable. Evidence in analysis-validation/.
