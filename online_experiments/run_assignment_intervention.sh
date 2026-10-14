#!/usr/bin/env bash
# Serial, resumable-by-new-job diagnostics; preserve failed jobs and existing runs.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="$ROOT/runs/paper/assignment-intervention"
PYTHON="${ASSIGNMENT_PYTHON:-/data/users/zhjia/miniconda3/envs/yolomaster/bin/python}"
SOURCE="${ASSIGNMENT_SOURCE:-$ROOT/runs/paper/full-visdrone/job-start-eDmH0odX}"
if [[ -z "${HF_HOME:-}" && -z "${HF_HUB_CACHE:-}" && -d "$HOME/cache/huggingface/hub" ]]; then
    export HF_HOME="$HOME/cache/huggingface"
fi
ACTION="${1:-status}"
mkdir -p "$BASE"

if [[ "$ACTION" == status ]]; then
    if [[ ! -f "$BASE/latest-job.txt" ]]; then
        echo "No assignment-intervention job has been launched."
        exit 0
    fi
    JOB="$(cat "$BASE/latest-job.txt")"
    echo "JOB=$JOB"
    cat "$JOB/status.txt"
    tail -n 8 "$JOB/job.log"
    for stage in smoke development confirmation; do
        if [[ -f "$JOB/$stage/progress.json" ]]; then
            echo "$stage:"
            cat "$JOB/$stage/progress.json"
        fi
        if [[ -f "$JOB/$stage/comparison.txt" ]]; then
            echo "$stage results:"
            cat "$JOB/$stage/comparison.txt"
        fi
    done
    exit 0
fi

if [[ "$ACTION" == start ]]; then
    exec 9>"$BASE/active.lock"
    if ! flock -n 9; then
        echo "An assignment-intervention worker is already active. Use status."
        exit 1
    fi
    JOB="$(mktemp -d "$BASE/job-$(date +%Y%m%d)-XXXXXXXX")"
    printf '%s\n' "$JOB" > "$BASE/latest-job.txt"
    printf 'QUEUED\n' > "$JOB/status.txt"
    # Worker inherits the acquired flock descriptor; no gap between start and worker.
    nohup bash online_experiments/run_assignment_intervention.sh worker "$JOB" > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    echo "JOB=$JOB PID=$!"
    echo "Watch: bash online_experiments/run_assignment_intervention.sh status"
    exit 0
fi

if [[ "$ACTION" != worker || $# != 2 ]]; then
    echo "Usage: bash online_experiments/run_assignment_intervention.sh {start|status}"
    exit 2
fi
JOB="$2"
if [[ ! -e /proc/$$/fd/9 ]]; then
    echo "Worker must be launched through start (active lock required)."
    exit 2
fi
trap 'code=$?; if (( code != 0 )); then printf "FAILED exit=%s: inspect job.log and stage traceback.txt\n" "$code" > "$JOB/status.txt"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${ASSIGNMENT_GPU:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
printf 'RUNNING tests\n' > "$JOB/status.txt"
"$PYTHON" -m pytest online_experiments/test_assignment_intervention.py -q -o addopts= > "$JOB/tests.log" 2>&1
for stage in smoke development confirmation; do
    case "$stage" in
        smoke) split=train; count=2 ;;
        development) split=train; count=128 ;;
        confirmation) split=val; count=0 ;;
    esac
    printf 'RUNNING %s\n' "$stage" > "$JOB/status.txt"
    echo "START $stage $(date -Is)"
    "$PYTHON" -u online_experiments/diagnose_assignment_intervention.py \
        --source-job "$SOURCE" --output "$JOB/$stage" --split "$split" --count "$count" \
        --seed 20261014 --head online-geom > "$JOB/$stage.log" 2>&1
    echo "END $stage $(date -Is)"
done
printf 'SUCCEEDED\n' > "$JOB/status.txt"
echo "SUCCEEDED $JOB"
