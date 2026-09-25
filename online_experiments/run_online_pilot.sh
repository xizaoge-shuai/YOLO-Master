#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${YOLO_REPO:-$HOME/YOLO-Master}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:-smoke}"
case "$STAGE" in
    check|smoke|train) ;;
    *) echo "Usage: bash run_online_pilot.sh check|smoke|train" >&2; exit 2 ;;
esac

CACHE="${D1_CACHE:-$PWD/runs/d1/p1/visdrone500/cache500}"
DATASET="${D1_DATASET:-$PWD/runs/d1/p1/visdrone500/source/visdrone500-p1.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
TEACHER_DTYPE="${TEACHER_DTYPE:-fp32}"
OUT="${OUT:-$PWD/runs/paper/online-pilot/${STAGE}-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$OUT"

for FILE in \
    "$CACHE/manifest.json" \
    "$DATASET" \
    scripts/d1_train_cached_detector.py \
    scripts/d1_build_feature_cache.py
do
    if [[ ! -f "$FILE" ]]; then
        echo "Missing required file: $FILE" >&2
        exit 1
    fi
done

git rev-parse HEAD | tee "$OUT/commit.txt"
git status --short > "$OUT/git-status.txt"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__, "CUDA", torch.cuda.is_available())'

COMMON=(
    --cache "$CACHE"
    --dataset "$DATASET"
    --train-count 400
    --val-count 100
    --split-seed 0
    --batch "$BATCH"
    --device cuda:0
    --lr 0.001
    --weight-decay 0.0005
    --aux-weight 0
    --teacher-dtype "$TEACHER_DTYPE"
)

python "$SCRIPT_DIR/d1_online_compare.py" "${COMMON[@]}" \
    --mode online --augment none --parity-only \
    --output "$OUT/parity" 2>&1 | tee "$OUT/parity.log"

if [[ "$STAGE" == check ]]; then
    echo "Check complete: $OUT"
    exit 0
fi

if [[ "$STAGE" == smoke ]]; then
    EPOCHS=1
    SEEDS=0
else
    SEEDS="${SEEDS:-0}"
fi

for SEED in $SEEDS; do
    for CONFIG in offline:none online:none online:hflip online:geom; do
        MODE="${CONFIG%%:*}"
        AUG="${CONFIG##*:}"
        NAME="${MODE}-${AUG}-s${SEED}"

        python "$SCRIPT_DIR/d1_online_compare.py" "${COMMON[@]}" \
            --mode "$MODE" \
            --augment "$AUG" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --output "$OUT/$NAME" \
            2>&1 | tee "$OUT/$NAME.log"
    done
done

python "$SCRIPT_DIR/summarize_online.py" "$OUT" | tee "$OUT/comparison.txt"
echo "Results: $OUT"
