#!/usr/bin/env bash
# 一键在云端创建 PAD 环境并按镜像源安装依赖
# - 默认环境名：PAD
# - 依赖文件：requirements-cloud.txt
# - 使用 TUNA 镜像和 PyTorch CU118 镜像，适合无法直接联网的机器

set -euo pipefail

ENV_NAME="PAD"

# 优先用 Python 3.10（兼容 mujoco-py 和 Torch cu121），可通过 `PYTHON_BIN=/path/to/python3.10 bash setup_cloud_env.sh` 覆盖
if [ -z "${PYTHON_BIN:-}" ]; then
  for c in python3.10 python3.11 python3.12 python3.9 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      PYTHON_BIN="$c"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  echo "ERROR: No python interpreter found. Please install Python 3.10/3.11/3.12 (e.g., conda create -n pad310 python=3.10)."
  exit 1
fi

echo "==> Using Python interpreter: ${PYTHON_BIN}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Please install Python 3.9/3.10 first."
  exit 1
fi

echo "==> Detecting Python version"
PY_MAJOR_MINOR=$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
echo "    Found Python ${PY_MAJOR_MINOR}"
case "${PY_MAJOR_MINOR}" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "ERROR: PyTorch cu121 wheels require Python 3.9-3.12. Please install a compatible Python (e.g., conda create -n pad310 python=3.10) and rerun with PYTHON_BIN pointing to it."
    exit 1
esac

echo "==> Creating virtualenv: ${ENV_NAME}"
"${PYTHON_BIN}" -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip setuptools wheel

echo "==> Installing dependencies from requirements-cloud.txt (mirror mode)"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/wheels/cu121 https://download.pytorch.org/whl/cu121"
pip install -r requirements-cloud.txt

echo "==> Done."
echo "Activate the env next time with:"
echo "  source ${ENV_NAME}/bin/activate"
