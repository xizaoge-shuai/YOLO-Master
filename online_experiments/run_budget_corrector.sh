#!/usr/bin/env bash
# Run from the existing yolomaster environment; only GPU0 is selected by Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="$ROOT/runs/paper/budget-corrector"
mkdir -p "$BASE"
ACTION="${1:-start}"
if [[ "$ACTION" == "_worker" ]]; then
    JOB="$2"
    SOURCE="$3"
    KIND="$4"
    PYTHON="$5"
    ARGS=()
    [[ "$KIND" == smoke ]] && ARGS+=(--smoke-only)
    set +e
    "$PYTHON" -u online_experiments/budget_corrector.py --job "$JOB" --source "$SOURCE" "${ARGS[@]}"
    RC=$?
    set -e
    if (( RC != 0 )); then
        printf 'FAILED exit=%s; inspect job.log and worker log\n' "$RC" > "$JOB/status.txt"
    fi
    # Formal progress is committed on success or failure; smoke is archived manually.
    if [[ "$KIND" != smoke ]]; then
        set +e
        "$PYTHON" online_experiments/archive_budget_corrector.py --job "$JOB" > "$JOB/archive.log" 2>&1
        ARC=$?
        set -e
        if (( ARC == 0 )); then
            printf 'COMMITTED\n' > "$JOB/archive-status.txt"
        else
            printf 'ARCHIVE_FAILED exit=%s; inspect archive.log\n' "$ARC" > "$JOB/archive-status.txt"
        fi
    fi
    exit "$RC"
fi
if [[ "$ACTION" == start || "$ACTION" == smoke ]]; then
    if [[ "$ACTION" == start ]]; then
        MARKER="$BASE/latest-job.txt"
    else
        MARKER="$BASE/latest-smoke.txt"
    fi
    if [[ -f "$MARKER" ]]; then
        OLD="$(cat "$MARKER")"
        if [[ -f "$OLD/pid.txt" && -f "$OLD/status.txt" ]] && grep -q '^RUNNING' "$OLD/status.txt"; then
            PID="$(cat "$OLD/pid.txt")"
            if kill -0 "$PID" 2>/dev/null; then
                printf 'Existing job still running: %s\n' "$OLD"
                exit 0
            fi
        fi
    fi
    SOURCE="$(cat "$ROOT/runs/paper/flip-cache/latest-job.txt")"
    PYTHON="$(python -c 'import sys; print(sys.executable)')"
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Activate yolomaster with CUDA"'
    JOB="$(mktemp -d "$BASE/job-${ACTION}-XXXXXXXX")"
    printf '%s\n' "$JOB" > "$MARKER"
    printf 'RUNNING launching\n' > "$JOB/status.txt"
    nohup bash "$ROOT/online_experiments/run_budget_corrector.sh" _worker "$JOB" "$SOURCE" "$ACTION" "$PYTHON" > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    printf 'JOB=%s\nPID=%s\n' "$JOB" "$(cat "$JOB/pid.txt")"
    exit 0
fi
JOB="${2:-}"
if [[ -z "$JOB" ]]; then JOB="$(cat "$BASE/latest-job.txt")"; fi
case "$ACTION" in
    status)
        printf 'JOB=%s\n' "$JOB"
        cat "$JOB/status.txt"
        tail -n 12 "$JOB/job.log"
        [[ ! -f "$JOB/comparison.txt" ]] || cat "$JOB/comparison.txt"
        [[ ! -f "$JOB/archive-status.txt" ]] || cat "$JOB/archive-status.txt"
        [[ ! -f "$JOB/archive.log" ]] || tail -n 3 "$JOB/archive.log"
        ;;
    watch) tail -n 30 -f "$JOB/job.log" ;;
    archive) python online_experiments/archive_budget_corrector.py --job "$JOB" ;;
    *) printf 'Usage: bash online_experiments/run_budget_corrector.sh {start|smoke|status|watch|archive} [JOB]\n'; exit 2 ;;
esac
