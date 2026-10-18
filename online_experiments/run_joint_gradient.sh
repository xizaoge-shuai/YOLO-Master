#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="$ROOT/runs/paper/joint-gradient"
PYTHON="${JOINT_PYTHON:-/data/users/zhjia/miniconda3/envs/yolomaster/bin/python}"
ACTION="${1:-status}"
mkdir -p "$BASE"
if [[ "$ACTION" == status ]]; then
    if [[ ! -f "$BASE/latest-job.txt" ]]; then echo 'No joint-gradient job started.'; exit 0; fi
    JOB="$(cat "$BASE/latest-job.txt")"
    echo "JOB=$JOB"
    cat "$JOB/status.txt"
    tail -n 8 "$JOB/job.log"
    for STAGE in smoke development; do
        if [[ -f "$JOB/$STAGE/progress.json" ]]; then echo "$STAGE"; cat "$JOB/$STAGE/progress.json"; fi
    done
    if [[ -f "$JOB/development/comparison.txt" ]]; then cat "$JOB/development/comparison.txt"; fi
    if [[ -f "$JOB/development/screening.json" ]]; then cat "$JOB/development/screening.json"; fi
    if [[ -f "$JOB/archive-status.json" ]]; then cat "$JOB/archive-status.json"; fi
    exit 0
fi
if [[ "$ACTION" == start ]]; then
    exec 9>"$BASE/campaign.lock"
    if ! flock -n 9; then echo 'Already running; use status.'; exit 1; fi
    STAMP="$(TZ=Asia/Shanghai date +%Y%m%d)"
    JOB="$(mktemp -d "$BASE/job-$STAMP-XXXXXXXX")"
    printf '%s\n' "$JOB" > "$BASE/latest-job.txt"
    printf 'QUEUED\n' > "$JOB/status.txt"
    nohup bash online_experiments/run_joint_gradient.sh worker "$JOB" > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    echo "JOB=$JOB PID=$!"
    exit 0
fi
if [[ "$ACTION" != worker || $# != 2 || ! -e /proc/$$/fd/9 ]]; then
    echo 'Usage: bash online_experiments/run_joint_gradient.sh {start|status}'
    exit 2
fi
JOB="$2"
trap 'code=$?; if (( code != 0 )); then printf "FAILED exit=%s; inspect job.log\n" "$code" > "$JOB/status.txt"; fi' EXIT
export HF_HOME="${HF_HOME:-$HOME/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${JOINT_GPU:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for STAGE in smoke development; do
    COUNT=2
    if [[ "$STAGE" == development ]]; then COUNT=128; fi
    printf 'RUNNING %s\n' "$STAGE" > "$JOB/status.txt"
    "$PYTHON" -u online_experiments/diagnose_joint_gradient.py \
        --source-job "$ROOT/runs/paper/full-visdrone/job-start-eDmH0odX" \
        --output "$JOB/$STAGE" --count "$COUNT" --seed 20261014
    # Verify smoke before advancing; archive errors are explicit and stop progression.
    "$PYTHON" -u online_experiments/archive_joint_gradient.py \
        --job "$JOB" --stage "$STAGE" --push > "$JOB/archive-$STAGE.log" 2>&1
done
printf 'SUCCEEDED\n' > "$JOB/status.txt"
