#!/usr/bin/env bash
set -euo pipefail
cd "${YOLO_REPO:-$HOME/YOLO-Master}"

MODE="${1:-all}"
JOB="${2:?missing absolute job directory}"
case "$MODE" in all|smoke|train) ;; *) exit 2 ;; esac
mkdir -p "$JOB"
PHASE=prepare

finish() {
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
        printf 'SUCCEEDED\n' > "$JOB/status.txt"
    else
        printf 'FAILED phase=%s exit=%s\n' "$PHASE" "$rc" > "$JOB/status.txt"
    fi
    cat "$JOB/status.txt"
}
trap finish EXIT

phase() {
    PHASE="$1"
    printf 'RUNNING %s\n' "$PHASE" > "$JOB/status.txt"
    printf '\n[%s] %s\n' "$(date -Is)" "$PHASE"
}

export PYTHONUNBUFFERED=1
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export HF_ENDPOINT=https://huggingface.co
unset ALL_PROXY all_proxy HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

phase weights
python -u - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, constants
from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor

cache = Path(os.environ.get("D1_CACHE", "runs/d1/p1/visdrone500/cache500"))
identity = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))["identity"]

print("Python:", sys.executable)
print("HF cache:", constants.HF_HUB_CACHE)
print("Model:", identity["model_id"], "Revision:", identity["revision"])

def load_offline():
    return DINOv3MultiLevelExtractor(
        identity["model_id"], revision=identity["revision"],
        layers=tuple(int(x) for x in identity["layers"]),
        device="cpu", dtype="fp32", local_files_only=True,
    )

try:
    extractor = load_offline()
except OSError as exc:
    print("Offline load failed; checking proxy before downloading:", exc)
    subprocess.run([
        "curl", "--proxy", os.environ["HTTPS_PROXY"],
        "--connect-timeout", "10", "--max-time", "30",
        "-f", "-L", "-sS", "-o", "/dev/null",
        "-w", "HTTP %{http_code}\n", "https://huggingface.co",
    ], check=True)
    snapshot_download(
        repo_id=identity["model_id"], revision=identity["revision"],
        local_files_only=False,
        allow_patterns=["*.json", "*.safetensors", "pytorch_model*.bin"],
        max_workers=2,
    )
    extractor = load_offline()

del extractor
print("WEIGHTS_READY: offline model loading passed.")
PY

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EPOCHS="${EPOCHS:-100}"
export SEEDS="${SEEDS:-0}"

if [[ "$MODE" == all || "$MODE" == smoke ]]; then
    phase smoke
    OUT="$JOB/smoke" bash online_experiments/run_online_pilot.sh smoke
fi

if [[ "$MODE" == all || "$MODE" == train ]]; then
    phase train
    OUT="$JOB/train" bash online_experiments/run_online_pilot.sh train
fi
