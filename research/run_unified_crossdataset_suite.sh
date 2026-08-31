#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null || true

cd ~/YOLO-Master || exit 1

ROOT="$PWD"

RUN_ROOT="$ROOT/runs/paper/unified-crossdataset"
DEV_ROOT="$RUN_ROOT/context-development"
FINAL_ROOT="$RUN_ROOT/frozen-method"
RANK_ROOT="$RUN_ROOT/rank-ablation"

mkdir -p \
  "$RUN_ROOT" \
  "$DEV_ROOT" \
  "$FINAL_ROOT" \
  "$RANK_ROOT" \
  "$ROOT/research/results"

MODEL="Tooony133/dinov3-vits16-pretrain-lvd1689m"
REVISION="fc6921f7a0b44d5b33ab4482cfed5443db6ccd81"

VIS_CACHE="$ROOT/runs/d1/p1/visdrone500/cache500"
VIS_DATASET="$ROOT/runs/d1/p1/visdrone500/source/visdrone500-p1.yaml"
VIS_CHECKPOINT="$ROOT/runs/d1/p1/visdrone500/gpu-resident/cached-preload-gpu-e100-lr1e-3-s0-20260825-090718/weights/best.pt"

COCO_CACHE="$ROOT/runs/d1/admission-824/cache100"
COCO_DATASET="coco128.yaml"
COCO_CHECKPOINT="$ROOT/runs/d1/p1/coco128-pilot/gpu-resident/cached-preload-gpu-e100-lr1e-3-s0-20260825-082142/weights/best.pt"

echo "================================================"
echo "Unified cross-dataset suite"
echo "================================================"
echo "python: $(which python)"
echo "branch: $(git branch --show-current)"
echo "commit: $(git rev-parse HEAD)"
echo

for FILE in \
  "$VIS_CACHE/manifest.json" \
  "$VIS_DATASET" \
  "$VIS_CHECKPOINT" \
  "$COCO_CACHE/manifest.json" \
  "$COCO_CHECKPOINT"
do
  if [ ! -f "$FILE" ]; then
    echo "FAIL: missing $FILE"
    exit 1
  fi
done

echo "Input files: PASS"

# ============================================================
# Stage A
# Global architecture development.
#
# IMPORTANT:
# context is selected ONLY according to calibration-set
# coefficient validation loss.
# Held-out detection AP is NOT used for architecture selection.
# ============================================================

echo
echo "================================================"
echo "STAGE A: global context development"
echo "================================================"

for CONTEXT in \
  point \
  local3 \
  local5 \
  multiscale \
  global \
  multiscale_global
do

  OUT="$DEV_ROOT/$CONTEXT"

  echo
  echo "----------------------------------------"
  echo "context=$CONTEXT"
  echo "----------------------------------------"

  rm -rf "$OUT"

  python research/train_l3_defect_predictor.py \
    --cache "$VIS_CACHE" \
    --dataset "$VIS_DATASET" \
    --checkpoint "$VIS_CHECKPOINT" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --layers 3 7 11 \
    --augmentations \
      hflip \
      zoom_in15 \
    --rank 64 \
    --hidden 128 \
    --context "$CONTEXT" \
    --train-count 400 \
    --val-count 100 \
    --calibration-count 50 \
    --calibration-seed 0 \
    --eval-count 8 \
    --positions-per-image 128 \
    --epochs 80 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed 0 \
    --device cuda:0 \
    --dtype fp16 \
    --conf 0.001 \
    --output "$OUT" \
    > "$DEV_ROOT/${CONTEXT}.log" \
    2>&1

  CODE=$?

  if [ "$CODE" -ne 0 ]; then
    echo "FAIL: context=$CONTEXT exit=$CODE"
    tail -n 80 "$DEV_ROOT/${CONTEXT}.log"
    exit "$CODE"
  fi

  echo "PASS: context=$CONTEXT"

done


# ============================================================
# Choose ONE global context.
# Selection criterion = average minimum coefficient validation
# loss across BOTH augmentations.
# No held-out AP is used.
# ============================================================

python - <<'PY'
import json
from pathlib import Path

root = Path(
    "runs/paper/unified-crossdataset/"
    "context-development"
)

records = []

for context in (
    "point",
    "local3",
    "local5",
    "multiscale",
    "global",
    "multiscale_global",
):
    path = root / context / "summary.json"

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    history = data[
        "training_history"
    ]

    aug_losses = []

    for aug, rows in history.items():
        best = min(
            float(row["val_loss"])
            for row in rows
        )

        aug_losses.append(best)

    mean_loss = (
        sum(aug_losses)
        /
        len(aug_losses)
    )

    records.append(
        (
            mean_loss,
            context,
            aug_losses,
        )
    )

records.sort()

best_loss, best_context, _ = records[0]

(root / "context-selection.tsv").write_text(
    "context\tmean_calibration_val_loss\n"
    +
    "\n".join(
        f"{context}\t{loss:.10f}"
        for loss, context, _
        in records
    )
    +
    "\n",
    encoding="utf-8",
)

(root / "BEST_CONTEXT").write_text(
    best_context + "\n",
    encoding="utf-8",
)

print("Context ranking:")

for loss, context, aug_losses in records:
    print(
        context,
        f"{loss:.8f}",
        aug_losses,
    )

print(
    "SELECTED_GLOBAL_CONTEXT=",
    best_context,
)

print(
    "SELECTION_LOSS=",
    best_loss,
)
PY

BEST_CONTEXT="$(
  cat "$DEV_ROOT/BEST_CONTEXT"
)"

echo
echo "Frozen context: $BEST_CONTEXT"


# ============================================================
# Stage B
# Frozen method across TWO datasets.
#
# Same:
# rank=64
# hidden=128
# context=$BEST_CONTEXT
# epochs=80
# lr=1e-3
# wd=1e-4
# augmentations=hflip,zoom
#
# Calibration ratio is fixed at 12.5%.
# VisDrone: 50/400
# COCO128: 10/80
# ============================================================

echo
echo "================================================"
echo "STAGE B: frozen cross-dataset evaluation"
echo "================================================"

for SEED in 0 1 2
do

  # ---------------- VisDrone500 ----------------

  OUT="$FINAL_ROOT/visdrone500-s${SEED}"

  rm -rf "$OUT"

  echo
  echo "VisDrone500 seed=$SEED"

  python research/train_l3_defect_predictor.py \
    --cache "$VIS_CACHE" \
    --dataset "$VIS_DATASET" \
    --checkpoint "$VIS_CHECKPOINT" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --layers 3 7 11 \
    --augmentations \
      hflip \
      zoom_in15 \
    --rank 64 \
    --hidden 128 \
    --context "$BEST_CONTEXT" \
    --train-count 400 \
    --val-count 100 \
    --calibration-count 50 \
    --calibration-seed "$SEED" \
    --eval-count 100 \
    --positions-per-image 128 \
    --epochs 80 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed "$SEED" \
    --device cuda:0 \
    --dtype fp16 \
    --conf 0.001 \
    --output "$OUT" \
    > "$FINAL_ROOT/visdrone500-s${SEED}.log" \
    2>&1

  CODE=$?

  if [ "$CODE" -ne 0 ]; then
    echo "FAIL: VisDrone seed=$SEED"
    tail -n 100 \
      "$FINAL_ROOT/visdrone500-s${SEED}.log"
    exit "$CODE"
  fi

  echo "PASS: VisDrone seed=$SEED"


  # ---------------- COCO128 ----------------

  OUT="$FINAL_ROOT/coco128-s${SEED}"

  rm -rf "$OUT"

  echo
  echo "COCO128 seed=$SEED"

  python research/train_l3_defect_predictor.py \
    --cache "$COCO_CACHE" \
    --dataset "$COCO_DATASET" \
    --checkpoint "$COCO_CHECKPOINT" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --layers 3 7 11 \
    --augmentations \
      hflip \
      zoom_in15 \
    --rank 64 \
    --hidden 128 \
    --context "$BEST_CONTEXT" \
    --train-count 80 \
    --val-count 20 \
    --calibration-count 10 \
    --calibration-seed "$SEED" \
    --eval-count 20 \
    --positions-per-image 128 \
    --epochs 80 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed "$SEED" \
    --device cuda:0 \
    --dtype fp16 \
    --conf 0.001 \
    --output "$OUT" \
    > "$FINAL_ROOT/coco128-s${SEED}.log" \
    2>&1

  CODE=$?

  if [ "$CODE" -ne 0 ]; then
    echo "FAIL: COCO128 seed=$SEED"
    tail -n 100 \
      "$FINAL_ROOT/coco128-s${SEED}.log"
    exit "$CODE"
  fi

  echo "PASS: COCO128 seed=$SEED"

done


# ============================================================
# Stage C
# Rank ablation.
#
# Same rank candidates are tested on BOTH datasets.
# Final method remains rank=64 regardless of per-dataset result.
# ============================================================

echo
echo "================================================"
echo "STAGE C: cross-dataset rank ablation"
echo "================================================"

for RANK in 32 64 96
do

  OUT="$RANK_ROOT/visdrone500-r${RANK}-s0"

  rm -rf "$OUT"

  python research/train_l3_defect_predictor.py \
    --cache "$VIS_CACHE" \
    --dataset "$VIS_DATASET" \
    --checkpoint "$VIS_CHECKPOINT" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --layers 3 7 11 \
    --augmentations \
      hflip \
      zoom_in15 \
    --rank "$RANK" \
    --hidden 128 \
    --context "$BEST_CONTEXT" \
    --train-count 400 \
    --val-count 100 \
    --calibration-count 50 \
    --calibration-seed 0 \
    --eval-count 100 \
    --positions-per-image 128 \
    --epochs 80 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed 0 \
    --device cuda:0 \
    --dtype fp16 \
    --conf 0.001 \
    --output "$OUT" \
    > "$RANK_ROOT/visdrone500-r${RANK}-s0.log" \
    2>&1

  [ "$?" -eq 0 ] || exit 1


  OUT="$RANK_ROOT/coco128-r${RANK}-s0"

  rm -rf "$OUT"

  python research/train_l3_defect_predictor.py \
    --cache "$COCO_CACHE" \
    --dataset "$COCO_DATASET" \
    --checkpoint "$COCO_CHECKPOINT" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --layers 3 7 11 \
    --augmentations \
      hflip \
      zoom_in15 \
    --rank "$RANK" \
    --hidden 128 \
    --context "$BEST_CONTEXT" \
    --train-count 80 \
    --val-count 20 \
    --calibration-count 10 \
    --calibration-seed 0 \
    --eval-count 20 \
    --positions-per-image 128 \
    --epochs 80 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed 0 \
    --device cuda:0 \
    --dtype fp16 \
    --conf 0.001 \
    --output "$OUT" \
    > "$RANK_ROOT/coco128-r${RANK}-s0.log" \
    2>&1

  [ "$?" -eq 0 ] || exit 1

  echo "PASS: rank=$RANK both datasets"

done


# ============================================================
# Stage D
# Final aggregation.
# ============================================================

echo
echo "================================================"
echo "STAGE D: aggregation"
echo "================================================"

python - <<'PY'
import csv
import json
import statistics
from pathlib import Path

root = Path(
    "runs/paper/unified-crossdataset"
)

final_root = (
    root / "frozen-method"
)

context = (
    root
    / "context-development"
    / "BEST_CONTEXT"
).read_text(
    encoding="utf-8"
).strip()

records = []

for dataset in (
    "visdrone500",
    "coco128",
):
    for seed in (
        0,
        1,
        2,
    ):
        directory = (
            final_root
            /
            f"{dataset}-s{seed}"
        )

        summary = json.loads(
            (
                directory
                /
                "summary.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        rows = list(
            csv.DictReader(
                (
                    directory
                    /
                    "results.csv"
                ).open(
                    encoding="utf-8"
                )
            )
        )

        for row in rows:
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "context": context,
                    "rank": 64,
                    "hidden": 128,
                    "augmentation": row[
                        "augmentation"
                    ],
                    "mode": row[
                        "mode"
                    ],
                    "map50_95": float(
                        row[
                            "metrics/mAP50-95(B)"
                        ]
                    ),
                    "ap_gap_recovery": float(
                        row[
                            "ap_gap_recovery"
                        ]
                    ),
                    "loss_gap_recovery": float(
                        row[
                            "loss_gap_recovery"
                        ]
                    ),
                    "parameters": int(
                        summary[
                            "predictor_parameters_total"
                        ]
                    ),
                }
            )


output = Path(
    "research/results/"
    "unified_crossdataset_summary.csv"
)

with output.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(records)


groups = {}

for row in records:
    key = (
        row["dataset"],
        row["augmentation"],
        row["mode"],
    )

    groups.setdefault(
        key,
        [],
    ).append(
        row["ap_gap_recovery"]
    )


print()
print("=" * 100)

print(
    f'{"dataset":14s}'
    f'{"augmentation":14s}'
    f'{"mode":14s}'
    f'{"mean recovery":>16s}'
    f'{"std":>12s}'
    f'{"min":>12s}'
    f'{"max":>12s}'
)

print("-" * 100)

for key in sorted(groups):
    dataset, aug, mode = key
    vals = groups[key]

    std = (
        statistics.stdev(vals)
        if len(vals) > 1
        else 0.0
    )

    print(
        f'{dataset:14s}'
        f'{aug:14s}'
        f'{mode:14s}'
        f'{statistics.mean(vals):16.4f}'
        f'{std:12.4f}'
        f'{min(vals):12.4f}'
        f'{max(vals):12.4f}'
    )

print("=" * 100)

print(
    "GLOBAL_CONTEXT=",
    context,
)

print(
    "CSV=",
    output,
)
PY


# ============================================================
# Rank-ablation summary
# ============================================================

python - <<'PY'
import csv
from pathlib import Path

root = Path(
    "runs/paper/"
    "unified-crossdataset/"
    "rank-ablation"
)

output = Path(
    "research/results/"
    "unified_rank_ablation.csv"
)

records = []

for dataset in (
    "visdrone500",
    "coco128",
):
    for rank in (
        32,
        64,
        96,
    ):
        path = (
            root
            /
            f"{dataset}-r{rank}-s0"
            /
            "results.csv"
        )

        rows = list(
            csv.DictReader(
                path.open(
                    encoding="utf-8"
                )
            )
        )

        for row in rows:
            if row["mode"] not in (
                "transport",
                "mean_only",
                "oracle",
                "predicted",
            ):
                continue

            records.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "augmentation": row[
                        "augmentation"
                    ],
                    "mode": row["mode"],
                    "map50_95": row[
                        "metrics/mAP50-95(B)"
                    ],
                    "ap_gap_recovery": row[
                        "ap_gap_recovery"
                    ],
                    "loss_gap_recovery": row[
                        "loss_gap_recovery"
                    ],
                }
            )

with output.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(records)

print(
    "PASS:",
    output,
)
PY


# ============================================================
# Record experiment log
# ============================================================

cat >> research/EXPERIMENT_LOG.md <<EOF

## Unified cross-dataset frozen-method experiment

The predictor architecture was selected once using calibration-set coefficient validation loss, without using held-out detection AP.

Frozen configuration:

- context: ${BEST_CONTEXT}
- rank: 64
- hidden width: 128
- predictor epochs: 80
- learning rate: 1e-3
- weight decay: 1e-4
- augmentations: horizontal flip and 15% zoom
- calibration ratio: 12.5%

The identical method configuration was evaluated on VisDrone500 and COCO128 with three calibration/training seeds. Dataset-specific method tuning was not used.

Machine-readable results:
- research/results/unified_crossdataset_summary.csv
- research/results/unified_rank_ablation.csv
EOF


# ============================================================
# Git snapshot to USER fork.
# ============================================================

git add \
  research/train_l3_defect_predictor.py \
  research/run_unified_crossdataset_suite.sh \
  research/EXPERIMENT_LOG.md \
  research/results/unified_crossdataset_summary.csv \
  research/results/unified_rank_ablation.csv

git diff --cached --check

if ! git diff --cached --quiet
then
  git commit -m \
    "research: validate frozen defect correction across datasets"

  git push origin \
    research-equivariance-defect
fi

echo
echo "================================================"
echo "UNIFIED CROSS-DATASET SUITE: PASS"
echo "================================================"
