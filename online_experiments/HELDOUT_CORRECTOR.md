# Frozen full-validation corrector diagnostic

This diagnostic uses the final epoch-100 Correct25 corrector and one fixed
Online-Geom detector from the completed full VisDrone campaign. It evaluates
548 validation images at five scales (1, .85, 1.15, .75, 1.25), with and without
horizontal flipping. It makes 5,480 exact teacher-image evaluations and no
parameter updates. Validation is used for mechanism development; this is not
an independent test-set or cross-dataset generalization result.

The original and true-flipped features are anchors. Spatial transport scales
the selected anchor; the reconstruction corrector adds its learned residual.
Foreground masks retain every positive-area visible fragment of transformed
YOLO-format boxes, including subpixel boxes. Detection loss independently
retains the existing training target filters. Fractional masks take maximum
per-box overlap, not exact union area. Ignored-region official AP is not
computed here.

All arms use identical image ordering and batch=4. Loss is an image-count-
weighted mean of batch target-score-normalized E2EDetectLoss components;
its numerical value is not guaranteed invariant to changing batch size.
Training timing comparisons must not use this diagnostic's elapsed time.

## Start a new diagnostic

Run only when a new evaluation is intended; each invocation creates a new run.
The original training launcher uses `$HOME/cache/huggingface`; preserve this
environment setting or offline model loading will fail. GPU1 was unavailable
to CUDA on 2026-09-27, despite appearing idle in nvidia-smi.

```bash
cd ~/YOLO-Master
SOURCE_JOB="$(cat runs/paper/full-visdrone/latest-job.txt)"
mkdir -p runs/paper/heldout-corrector
JOB="$(mktemp -d "$PWD/runs/paper/heldout-corrector/job-XXXXXXXX")"
printf '%s\n' "$JOB" > runs/paper/heldout-corrector/latest-job.txt
printf 'RUNNING starting\n' > "$JOB/status.txt"

nohup env CUDA_VISIBLE_DEVICES=0 HF_HOME="$HOME/cache/huggingface" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/users/zhjia/miniconda3/envs/yolomaster/bin/python -u \
  online_experiments/diagnose_full_corrector.py \
  --source-job "$SOURCE_JOB" --output "$JOB" --batch 4 \
  > "$JOB/job.log" 2>&1 < /dev/null &
echo "$!" > "$JOB/pid.txt"
```

## Inspect

```bash
cd ~/YOLO-Master
JOB="$(cat runs/paper/heldout-corrector/latest-job.txt)"
cat "$JOB/status.txt"
tail -n 20 "$JOB/job.log"
if [[ -f "$JOB/comparison.txt" ]]; then
    cat "$JOB/comparison.txt"
fi
```

Expected outputs: provenance.json, per-image.csv (10,960 rows), summary.json
(20 rows), comparison.txt and status.txt. Relative MSE and loss ratios are
diagnostic metrics, not AP. Compare nonidentity scales separately from scale1;
the true-flip scale1 anchor equals its exact teacher by construction. Preserve
failed runs and archive new progress with source hashes and a scoped Git commit.
