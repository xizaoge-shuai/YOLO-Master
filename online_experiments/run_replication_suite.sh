#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="$PWD/runs/paper/replication-suite"
PYTHON="${REPLICATION_PYTHON:-$(dirname "$PWD")/miniconda3/envs/yolomaster/bin/python}"
ACTION="${1:-status}"
mkdir -p "$BASE"
if [[ "$ACTION" == status ]]; then
    if [[ ! -f "$BASE/latest-job.txt" ]]; then echo 'No replication suite started.'; exit 0; fi
    JOB="$(cat "$BASE/latest-job.txt")"
    echo "JOB=$JOB"
    cat "$JOB/status.txt"
    tail -n 5 "$JOB/job.log"
    LOG="$(find "$JOB" -maxdepth 1 -name 'train-*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$LOG" ]]; then echo "${LOG##*/}"; tail -n 2 "$LOG"; fi
    if [[ -f "$JOB/comparison.txt" ]]; then cat "$JOB/comparison.txt"; fi
    exit 0
fi
if [[ "$ACTION" == publish ]]; then
    "$PYTHON" -u online_experiments/replication_campaign.py publish --job "$(cat "$BASE/latest-job.txt")"
    exit 0
fi
if [[ "$ACTION" == start || "$ACTION" == resume ]]; then
    exec 9>"$BASE/campaign.lock"
    if ! flock -n 9; then echo 'Already running; use status.'; exit 1; fi
    if [[ "$ACTION" == resume ]]; then
        JOB="$(cat "$BASE/latest-job.txt")"
        if [[ "$(cat "$JOB/status.txt")" == SUCCEEDED ]]; then echo 'Already complete.'; exit 0; fi
    else
        JOB="$(mktemp -d "$BASE/job-$(TZ=Asia/Shanghai date +%Y%m%d)-XXXXXXXX")"
    fi
    printf '%s\n' "$JOB" > "$BASE/latest-job.txt"
    printf 'QUEUED\n' > "$JOB/status.txt"
    nohup bash online_experiments/run_replication_suite.sh worker "$JOB" >> "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    echo "JOB=$JOB PID=$!"
    exit 0
fi
if [[ "$ACTION" != worker || $# != 2 || ! -e /proc/$$/fd/9 ]]; then
    echo 'Usage: bash online_experiments/run_replication_suite.sh {start|resume|status|publish}'
    exit 2
fi
export HF_HOME="${HF_HOME:-$(dirname "$PWD")/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${REPLICATION_GPU:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PYTHON" -u online_experiments/replication_campaign.py worker --job "$2" --push
