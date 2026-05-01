#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_GEN_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-gen}"
PYTHON_VERSION="${UV_PYTHON_VERSION:-3.8}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
CSDF_DIR="$ROOT_DIR/dexgrasp_generation/thirdparty/CSDF"

export UV_CACHE_DIR
export UV_HTTP_TIMEOUT

uv python install "$PYTHON_VERSION"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"

UV_PIP=(env -u PYTHONPATH uv pip install --python "$ENV_DIR/bin/python")

"${UV_PIP[@]}" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  "torch==1.11.0+cu113" \
  "torchvision==0.12.0+cu113" \
  "torchaudio==0.11.0+cu113"

"${UV_PIP[@]}" \
  iopath \
  "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl"

"${UV_PIP[@]}" \
  tensorboard==2.11.2 \
  hydra-core==1.3.2 \
  transforms3d==0.4.1 \
  lxml==4.9.2 \
  trimesh==3.9.8 \
  scipy==1.10.0 \
  UMNN==1.68 \
  healpy==1.16.2 \
  plotly==5.14.1 \
  transformations \
  absl-py

"${UV_PIP[@]}" \
  torch-scatter==2.0.9 \
  torch-sparse==0.6.15 \
  torch-cluster==1.6.0 \
  torch-geometric==2.2.0 \
  --find-links https://data.pyg.org/whl/torch-1.11.0+cu113.html

"${UV_PIP[@]}" nvidia-cublas-cu11 nvidia-cusparse-cu11

if [[ ! -d "$CSDF_DIR/.git" ]]; then
  git clone https://github.com/wrc042/CSDF.git "$CSDF_DIR"
fi

"${UV_PIP[@]}" --no-build-isolation \
  -e "$ROOT_DIR/dexgrasp_generation/thirdparty/pytorch_kinematics" \
  -e "$ROOT_DIR/dexgrasp_generation/thirdparty/nflows"

env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' uv pip install \
  --python "$ENV_DIR/bin/python" \
  --no-build-isolation \
  -e "$CSDF_DIR"

cat <<EOF
Generation environment is ready.

Activate with:
  source "$ROOT_DIR/scripts/activate_uv_generation.sh"

Notes:
  - This setup uses Python 3.8 and a uv-managed virtualenv.
  - CSDF is built in CPU mode because this host only exposes CUDA 12.3 nvcc,
    while the PyTorch stack above is compiled against CUDA 11.3.
EOF
