#!/usr/bin/env bash
# 一键在云端创建 PAD 环境并按镜像源安装依赖
# - 默认环境名：PAD
# - 依赖文件：requirements-cloud.txt
# - 使用 TUNA 镜像和 PyTorch CU118 镜像，适合无法直接联网的机器

set -euo pipefail

ENV_NAME="PAD"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Please install Miniconda/Anaconda first."
  exit 1
fi

case "${PYTHON_VERSION}" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "ERROR: PyTorch cu121 wheels require Python 3.9-3.12. Set PYTHON_VERSION to one of these."
    exit 1
    ;;
esac

# Prevent accidental overwrite of an existing env
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "ERROR: Conda env '${ENV_NAME}' already exists. Remove it with 'conda env remove -n ${ENV_NAME}' or set ENV_NAME to another name."
  exit 1
fi

echo "==> Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "==> Activating conda env"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "    Using Python at $(which python)"
python - <<'PY'
import sys
print(f"    Python version: {sys.version.split()[0]}")
PY

echo "==> Upgrading pip"
pip install --upgrade pip wheel

echo "==> Pinning setuptools for gym==0.21.0 build compatibility"
pip install "setuptools==65.5.1"

# Use TUNA mirrors by default (pip)
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/wheels/cu121 https://download.pytorch.org/whl/cu121"

# Install gym via conda to avoid building from source (gym 0.21.0 has setup.py quirks)
echo "==> Installing gym==0.21.0 via conda-forge (prebuilt wheel)"
conda install -y -n "${ENV_NAME}" -c conda-forge "gym==0.21.0"
# Re-activate to ensure conda-installed packages are on PATH
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Fix invalid requirement metadata in conda gym 0.21.0 (opencv-python (>=3.) -> (>=3.0))
echo "==> Patching gym METADATA for pip compatibility"
python - <<'PY'
import importlib.metadata
from pathlib import Path
import re

dist = importlib.metadata.distribution("gym")
meta_path = Path(dist._path) / "METADATA"  # _path is available on Distribution
text = meta_path.read_text()
# Fix invalid requirement specifiers like opencv-python (>=3.) under different extras
fixed = re.sub(r"opencv-python \(>=3\.\)(\s*; extra == '[^']+')", r"opencv-python (>=3.0)\1", text)
if text != fixed:
    meta_path.write_text(fixed)
    print(f"Patched {meta_path}")
else:
    print("No patch needed")
PY

echo "==> Installing dependencies from requirements-cloud.txt (mirror mode)"
# Disable build isolation globally to reuse pinned setuptools for other sdists
export PIP_NO_BUILD_ISOLATION=1
pip install -r requirements-cloud.txt

echo "==> Done."
echo "Activate the env next time with:"
echo "  conda activate ${ENV_NAME}"
