#!/usr/bin/env bash
# 一键在云端创建 PAD 环境并按镜像源安装依赖
# - 默认环境名：PAD
# - 依赖文件：requirements-cloud.txt
# - 使用 TUNA 镜像和 PyTorch CU118 镜像，适合无法直接联网的机器

set -euo pipefail

ENV_NAME="PAD"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Using Python interpreter: ${PYTHON_BIN}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Please install Python 3.9/3.10 first."
  exit 1
fi

echo "==> Creating virtualenv: ${ENV_NAME}"
"${PYTHON_BIN}" -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip setuptools wheel

echo "==> Installing dependencies from requirements-cloud.txt (mirror mode)"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/wheels/cu118 https://download.pytorch.org/whl/cu118"
pip install -r requirements-cloud.txt

echo "==> Done."
echo "Activate the env next time with:"
echo "  source ${ENV_NAME}/bin/activate"
