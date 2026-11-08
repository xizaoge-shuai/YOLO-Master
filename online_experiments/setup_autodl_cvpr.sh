#!/usr/bin/env bash
# Configure an isolated environment without changing SSH authentication or running training.
set -Eeuo pipefail
umask 022
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
TRANSFER="$DATA_ROOT/transfer"
REPO="$DATA_ROOT/YOLO-Master"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
PYTHON="$CONDA_ROOT/envs/yolomaster/bin/python"
mkdir -p "$TRANSFER" "$DATA_ROOT/cache/huggingface" "$DATA_ROOT/datasets"
exec > >(tee -a "$TRANSFER/setup.log") 2>&1
trap 'printf "FAILED line=%s exit=%s\n" "$LINENO" "$?" > "$TRANSFER/setup-status.txt"' ERR
printf 'RUNNING bootstrap\n' > "$TRANSFER/setup-status.txt"
export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export NO_PROXY=localhost,127.0.0.1,::1 PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
export HF_HOME="$DATA_ROOT/cache/huggingface" MPLCONFIGDIR="$DATA_ROOT/cache/matplotlib" TZ=Asia/Shanghai

curl -fLsS --retry 3 --connect-timeout 15 --max-time 60 \
    https://github.com/xizaoge-shuai/YOLO-Master -o /dev/null
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    if [[ -e "$CONDA_ROOT" ]]; then
        printf 'Refusing to overwrite incomplete existing environment: %s\n' "$CONDA_ROOT" >&2
        exit 2
    fi
    installer="$TRANSFER/Miniconda3-py310_26.9.4-0-Linux-x86_64.sh"
    curl -fL --retry 4 --retry-delay 3 --connect-timeout 20 --max-time 1800 -C - \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_26.9.4-0-Linux-x86_64.sh -o "$installer"
    printf '258005644606ee77d2ed31bcff9dd14975885f601a1fd55d06e4628d0a047f89  %s\n' "$installer" \
        | sha256sum -c -
    bash "$installer" -b -p "$CONDA_ROOT"
fi
if [[ ! -x "$PYTHON" ]]; then
    "$CONDA_ROOT/bin/conda" create -y -n yolomaster --override-channels -c conda-forge python=3.10.20 pip
fi
printf 'RUNNING clone_code\n' > "$TRANSFER/setup-status.txt"
if [[ ! -d "$REPO/.git" ]]; then
    if [[ -e "$REPO" ]]; then
        printf 'Refusing to overwrite existing non-git directory: %s\n' "$REPO" >&2
        exit 3
    fi
    git clone --depth 1 --single-branch --branch research/paper-baselines-20261107 \
        https://github.com/xizaoge-shuai/YOLO-Master.git "$REPO"
    git -C "$REPO" switch -c research/autodl-migration-20261109
fi
git -C "$REPO" rev-parse HEAD > "$TRANSFER/source-base-commit.txt"
printf 'RUNNING install_torch\n' > "$TRANSFER/setup-status.txt"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.13.0' torchvision
printf 'RUNNING install_project_dependencies\n' > "$TRANSFER/setup-status.txt"
"$PYTHON" -m pip install 'numpy==2.2.6' 'scipy==1.15.3' 'transformers==5.15.1' \
    opencv-python-headless pillow pandas matplotlib pyyaml requests tqdm psutil \
    pycocotools scikit-learn huggingface_hub safetensors ultralytics-thop polars lap ruff codespell
"$PYTHON" -m pip install --no-deps -e "$REPO"
cat > "$TRANSFER/env.sh" <<EOF
export PATH="$CONDA_ROOT/envs/yolomaster/bin:$CONDA_ROOT/bin:\$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="$HF_HOME"
export MPLCONFIGDIR="$MPLCONFIGDIR"
export TZ=Asia/Shanghai
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY=http://127.0.0.1:17897
export NO_PROXY=localhost,127.0.0.1,::1
EOF
"$PYTHON" -m pip freeze > "$TRANSFER/pip-freeze.txt"
"$PYTHON" - "$TRANSFER/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
record = {
    "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
    "packages": {name: importlib.metadata.version(name) for name in
                 ("numpy", "scipy", "transformers", "torchvision", "pycocotools")},
    "gpu_inventory": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
    "verification": "imports and GPU inventory only; no model forward, training or subset evaluation",
}
assert record["python"] == "3.10.20"
assert record["torch"].startswith("2.13.0")
assert record["cuda_build"] == "13.0"
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
PY
printf 'SUCCEEDED environment_and_code\n' > "$TRANSFER/setup-status.txt"
