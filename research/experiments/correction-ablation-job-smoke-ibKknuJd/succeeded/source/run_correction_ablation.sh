#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="$ROOT/runs/paper/correction-ablation"
mkdir -p "$BASE"
PYTHON="${ABLATION_PYTHON:-/data/users/zhjia/miniconda3/envs/yolomaster/bin/python}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z "${HF_HOME:-}" && -z "${HF_HUB_CACHE:-}" ]]; then
    export HF_HOME="$HOME/cache/huggingface"
fi
ACTION="${1:-status}"
if [[ "$ACTION" == _worker ]]; then
    JOB="$2"
    SOURCE="$3"
    MODE="$4"
    ARGS=()
    [[ "$MODE" != smoke ]] || ARGS+=(--smoke)
    exec "$PYTHON" -u online_experiments/correction_ablation_campaign.py \
        --job "$JOB" --source "$SOURCE" "${ARGS[@]}"
fi
if [[ "$ACTION" == start || "$ACTION" == smoke || "$ACTION" == resume ]]; then
    exec 9>"$BASE/.launch.lock"
    flock -n 9 || { echo 'An ablation worker is active; use status.'; exit 1; }
    MODE="$ACTION"
    MARKER="$BASE/latest-job.txt"
    if [[ "$ACTION" == resume ]]; then
        JOB="${2:-$(cat "$MARKER")}"
        if grep -q '^SUCCEEDED' "$JOB/status.txt"; then echo "Already complete: $JOB"; exit 0; fi
        SOURCE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_job"])' "$JOB/plan.json")"
        MODE="$("$PYTHON" -c 'import json,sys; print("smoke" if json.load(open(sys.argv[1]))["smoke"] else "start")' "$JOB/plan.json")"
    else
        if [[ "$ACTION" == smoke ]]; then
            SOURCE="$(cat runs/paper/full-visdrone/latest-smoke.txt)"
            MARKER="$BASE/latest-smoke.txt"
        else
            SOURCE="$(cat runs/paper/full-visdrone/latest-job.txt)"
            SMOKE="$(cat "$BASE/latest-smoke.txt")"
            grep -q '^SUCCEEDED' "$SMOKE/status.txt" || { echo 'Complete smoke first.'; exit 1; }
            "$PYTHON" - "$SMOKE" <<'PY'
import sys
from pathlib import Path
from online_experiments.full_visdrone_data import read, verify_sources
verify_sources(read(Path(sys.argv[1]) / "provenance.json"))
assert len(read(Path(sys.argv[1]) / "smoke-checks.json")) == 3
PY
        fi
        JOB="$(mktemp -d "$BASE/job-${ACTION}-XXXXXXXX")"
        printf '%s\n' "$JOB" > "$MARKER"
    fi
    if [[ "${ABLATION_ALLOW_SHARED_GPU:-0}" != 1 && -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
        echo 'GPU0 has compute processes; leaving them untouched.'
        exit 1
    fi
    "$PYTHON" -c 'import torch; x=torch.ones(8,device="cuda"); torch.cuda.synchronize(); print("GPU0 CUDA PASS")'
    printf 'RUNNING launching\n' > "$JOB/status.txt"
    nohup bash "$ROOT/online_experiments/run_correction_ablation.sh" _worker "$JOB" "$SOURCE" "$MODE" \
        > "$JOB/job.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$JOB/pid.txt"
    printf 'JOB=%s\nPID=%s\n' "$JOB" "$(cat "$JOB/pid.txt")"
    exit 0
fi
JOB="${2:-$(cat "$BASE/latest-job.txt")}"
case "$ACTION" in
    status)
        echo "JOB=$JOB"
        cat "$JOB/status.txt"
        tail -n 12 "$JOB/job.log"
        "$PYTHON" - "$JOB" <<'PY'
import json,sys
from pathlib import Path
for p in sorted((Path(sys.argv[1])/"train").glob("*/summary.json")):
    s=json.loads(p.read_text())
    print(f"{p.parent.name}: epoch {s['completed_epochs']}/{s['epochs']} custom_last_AP={s['last_AP']:.4f} queries={s['query_images']}")
PY
        [[ ! -f "$JOB/comparison.txt" ]] || cat "$JOB/comparison.txt"
        [[ ! -f "$JOB/archive-status.txt" ]] || cat "$JOB/archive-status.txt"
        ;;
    watch) tail -n 25 -f "$JOB/job.log" ;;
    *) echo 'Usage: run_correction_ablation.sh {smoke|start|resume|status|watch} [JOB]'; exit 2 ;;
esac
