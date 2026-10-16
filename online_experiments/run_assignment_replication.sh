#!/usr/bin/env bash
# Same diagnostic computation and fixed validation set; change only the frozen head.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="$ROOT/runs/paper/assignment-intervention"
PYTHON="${ASSIGNMENT_PYTHON:-/data/users/zhjia/miniconda3/envs/yolomaster/bin/python}"
ACTION="${1:-status}"
mkdir -p "$BASE"
if [[ "$ACTION" == status ]]; then
    if [[ ! -f "$BASE/latest-replication.txt" ]]; then
        echo "No Mix25-head replication has been started."
        exit 0
    fi
    JOB="$(cat "$BASE/latest-replication.txt")"
    echo "JOB=$JOB"
    cat "$JOB/status.txt"
    tail -n 8 "$JOB/job.log"
    if [[ -f "$JOB/confirmation/progress.json" ]]; then cat "$JOB/confirmation/progress.json"; fi
    if [[ -f "$JOB/confirmation/comparison.txt" ]]; then cat "$JOB/confirmation/comparison.txt"; fi
    if [[ -f "$JOB/archive-status.json" ]]; then cat "$JOB/archive-status.json"; fi
    exit 0
fi
if [[ "$ACTION" == start ]]; then
    exec 9>"$BASE/replication.lock"
    if ! flock -n 9; then echo "Replication is already running; use status."; exit 1; fi
    STAMP="$(TZ=Asia/Shanghai date +%Y%m%d)"
    JOB="$(mktemp -d "$BASE/job-replication-mix25-$STAMP-XXXXXXXX")"
    printf '%s\n' "$JOB" > "$BASE/latest-replication.txt"
    printf 'QUEUED\n' > "$JOB/status.txt"
    nohup bash online_experiments/run_assignment_replication.sh worker "$JOB" > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    echo "JOB=$JOB PID=$!"
    exit 0
fi
if [[ "$ACTION" != worker || $# != 2 || ! -e /proc/$$/fd/9 ]]; then
    echo "Usage: bash online_experiments/run_assignment_replication.sh {start|status}"
    exit 2
fi
JOB="$2"
trap 'code=$?; if (( code != 0 )); then printf "FAILED exit=%s; inspect job.log\n" "$code" > "$JOB/status.txt"; fi' EXIT
export HF_HOME="${HF_HOME:-$HOME/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${ASSIGNMENT_GPU:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
printf 'RUNNING Mix25-head confirmation\n' > "$JOB/status.txt"
"$PYTHON" -u online_experiments/diagnose_assignment_intervention.py \
    --source-job "$ROOT/runs/paper/full-visdrone/job-start-eDmH0odX" \
    --output "$JOB/confirmation" --split val --count 0 --seed 20261014 --head mix25
printf 'SUCCEEDED\n' > "$JOB/status.txt"
if ! "$PYTHON" -u online_experiments/archive_assignment_intervention.py \
    --job "$JOB" --stages confirmation --once > "$JOB/archive.log" 2>&1; then
    echo "Computation succeeded; result archiving needs attention: $JOB/archive.log"
fi
