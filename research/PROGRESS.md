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
