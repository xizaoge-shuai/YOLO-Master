#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="$ROOT/runs/paper/full-visdrone"
mkdir -p "$BASE"
if [[ -z "${HF_HOME:-}" && -z "${HF_HUB_CACHE:-}" && -d "$HOME/cache/huggingface/hub" ]]; then
    export HF_HOME="$HOME/cache/huggingface"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
ACTION="${1:-start}"
if [[ "$ACTION" == _worker ]]; then
    JOB="$2"
    KIND="$3"
    PYTHON="$4"
    ARGS=()
    [[ "$KIND" != smoke ]] || ARGS+=(--smoke)
    set +e
    "$PYTHON" -u online_experiments/full_visdrone_campaign.py --job "$JOB" "${ARGS[@]}"
    RC=$?
    if (( RC != 0 )); then
        printf 'FAILED exit=%s; inspect job.log and case log\n' "$RC" > "$JOB/status.txt"
    fi
    "$PYTHON" online_experiments/full_visdrone_campaign.py --job "$JOB" --archive > "$JOB/archive.log" 2>&1
    ARC=$?
    if (( ARC == 0 )); then
        printf 'COMMITTED\n' > "$JOB/archive-status.txt"
    else
        printf 'ARCHIVE_FAILED exit=%s; inspect archive.log\n' "$ARC" > "$JOB/archive-status.txt"
    fi
    exit "$RC"
fi
if [[ "$ACTION" == start || "$ACTION" == smoke || "$ACTION" == resume ]]; then
    exec 9>"$BASE/.launch.lock"
    if ! flock -n 9; then
        printf 'A full-visdrone job is already running; use status.
'
        exit 1
    fi
    KIND="$ACTION"
    MARKER="$BASE/latest-job.txt"
    [[ "$ACTION" != smoke ]] || MARKER="$BASE/latest-smoke.txt"
    if [[ "$ACTION" == resume ]]; then
        JOB="${2:-$(cat "$BASE/latest-job.txt")}"
        KIND="$(python - "$JOB" <<'PY'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1]) / "plan.json").read_text())
print("smoke" if p["smoke"] else "start")
PY
)"
        if grep -q '^SUCCEEDED' "$JOB/status.txt"; then
            printf 'This job is already complete: %s
' "$JOB"
            exit 0
        fi
        if [[ -f "$JOB/pid.txt" ]] && kill -0 "$(cat "$JOB/pid.txt")" 2>/dev/null; then
            printf 'This job still has a live process: %s
' "$JOB"
            exit 1
        fi
    elif [[ -f "$MARKER" ]]; then
        OLD="$(cat "$MARKER")"
        if [[ -f "$OLD/pid.txt" ]] && grep -q '^RUNNING' "$OLD/status.txt" && kill -0 "$(cat "$OLD/pid.txt")" 2>/dev/null; then
            printf 'Already running: %s\n' "$OLD"
            exit 0
        fi
    fi
    PYTHON="$(python -c 'import sys; print(sys.executable)')"
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda"); print("GPU0 CUDA PASS:",torch.cuda.get_device_name(0))'
    if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
        printf 'GPU0 already has compute processes; inspect nvidia-smi before launching.\n'
        exit 1
    fi
    if [[ "$ACTION" != resume ]]; then
        JOB="$(mktemp -d "$BASE/job-${ACTION}-XXXXXXXX")"
        printf '%s\n' "$JOB" > "$MARKER"
    fi
    printf 'RUNNING launching\n' > "$JOB/status.txt"
    nohup bash "$ROOT/online_experiments/run_full_visdrone.sh" _worker "$JOB" "$KIND" "$PYTHON" > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    printf 'JOB=%s\nPID=%s\n' "$JOB" "$(cat "$JOB/pid.txt")"
    exit 0
fi
JOB="${2:-$(cat "$BASE/latest-job.txt")}"
case "$ACTION" in
    status)
        printf 'JOB=%s\n' "$JOB"
        cat "$JOB/status.txt"
        tail -n 12 "$JOB/job.log"
        python - "$JOB" <<'PY'
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
for phase in ("profile", "train"):
    for path in sorted((job / phase).glob("*/summary.json")):
        s = json.loads(path.read_text())
        print(f"{phase}/{path.parent.name}: epoch {s['completed_epochs']}/{s['epochs']} best_AP={s['best_AP']:.4f} last_AP={s['last_AP']:.4f}")
PY
        [[ ! -f "$JOB/eta.json" ]] || cat "$JOB/eta.json"
        [[ ! -f "$JOB/comparison.txt" ]] || cat "$JOB/comparison.txt"
        [[ ! -f "$JOB/archive-status.txt" ]] || cat "$JOB/archive-status.txt"
        [[ ! -f "$JOB/archive.log" ]] || tail -n 3 "$JOB/archive.log"
        ;;
    watch) tail -n 30 -f "$JOB/job.log" ;;
    archive) python online_experiments/full_visdrone_campaign.py --job "$JOB" --archive ;;
    *) printf 'Usage: bash online_experiments/run_full_visdrone.sh {start|smoke|resume|status|watch|archive} [JOB]\n'; exit 2 ;;
esac
