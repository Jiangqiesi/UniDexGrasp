#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_GEN_H20_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-gen-h20}"
PYTHON_VERSION="${UV_GEN_H20_PYTHON_VERSION:-3.10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
CSDF_DIR="$ROOT_DIR/dexgrasp_generation/thirdparty/CSDF"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export UV_CACHE_DIR
export UV_HTTP_TIMEOUT

if [[ -f /opt/rh/gcc-toolset-11/enable ]]; then
  # gcc-toolset's enable script is not nounset-safe on TencentOS.
  set +u
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-11/enable
  set -u
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

uv python install "$PYTHON_VERSION"
if [[ "$FORCE_REINSTALL" == "1" ]]; then
  uv venv "$ENV_DIR" --python "$PYTHON_VERSION" --clear
elif [[ -x "$ENV_DIR/bin/python" ]]; then
  echo "Reusing existing virtualenv: $ENV_DIR"
else
  uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
fi

UV_PIP=(env -u PYTHONPATH uv pip install --python "$ENV_DIR/bin/python")
UV_PIP_UNINSTALL=(env -u PYTHONPATH uv pip uninstall --python "$ENV_DIR/bin/python")
PYTHON=(env -u PYTHONPATH "$ENV_DIR/bin/python")

have_module() {
  local module="$1"
  "${PYTHON[@]}" - "$module" <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

have_torch_h20() {
  "${PYTHON[@]}" <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__.startswith("2.4.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
PY
}

"${UV_PIP[@]}" -U pip wheel ninja packaging
"${UV_PIP[@]}" "setuptools<70"

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_torch_h20; then
  "${UV_PIP[@]}" \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    "torchaudio==2.4.1"
else
  echo "torch 2.4.1 CUDA 12.x already installed; skipping torch install."
fi

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_module hydra || ! have_module tensorboard || ! have_module trimesh; then
  "${UV_PIP[@]}" \
    tensorboard==2.17.1 \
    hydra-core==1.3.2 \
    numpy==1.26.4 \
    transforms3d==0.4.1 \
    lxml==5.3.0 \
    trimesh==4.4.9 \
    scipy==1.14.1 \
    UMNN==1.68 \
    healpy==1.17.3 \
    plotly==5.24.1 \
    transformations \
    absl-py \
    iopath \
    fvcore
else
  echo "base generation Python deps already installed; skipping."
fi

if "${PYTHON[@]}" <<'PY' >/dev/null 2>&1
import numpy as np
assert int(np.__version__.split(".", 1)[0]) < 2, np.__version__
PY
then
  echo "NumPy < 2 already installed; skipping NumPy compatibility pin."
else
  "${UV_PIP[@]}" "numpy==1.26.4"
fi

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_module torch_geometric || ! have_module torch_scatter || ! have_module torch_cluster; then
  "${UV_PIP[@]}" \
    torch-scatter \
    torch-cluster \
    torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.4.0+cu124.html

  "${UV_PIP[@]}" torch-geometric==2.6.1
else
  echo "PyG stack already installed; skipping."
fi

if [[ "${KEEP_PYG_OPTIONAL:-0}" != "1" ]]; then
  # TencentOS glibc is older than the PyG optional pyg-lib wheel expects.
  # UniDexGrasp generation uses fps/radius/PointConv, which need torch-cluster
  # and torch-scatter, not pyg-lib or torch-sparse.
  "${UV_PIP_UNINSTALL[@]}" pyg-lib torch-sparse || true
fi

if [[ ! -d "$CSDF_DIR/.git" ]]; then
  git clone https://github.com/wrc042/CSDF.git "$CSDF_DIR"
fi

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_module pytorch_kinematics || ! have_module nflows; then
  "${UV_PIP[@]}" --no-build-isolation \
    -e "$ROOT_DIR/dexgrasp_generation/thirdparty/pytorch_kinematics" \
    -e "$ROOT_DIR/dexgrasp_generation/thirdparty/nflows"
else
  echo "editable local generation deps already installed; skipping."
fi

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_module pytorch3d; then
  env -u PYTHONPATH \
    CUDA_HOME="$CUDA_HOME" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install --python "$ENV_DIR/bin/python" \
    --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
else
  echo "PyTorch3D already installed; skipping source build."
fi

"${UV_PIP[@]}" "setuptools<70"

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_module csdf; then
  env -u PYTHONPATH \
    CUDA_HOME="$CUDA_HOME" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install --python "$ENV_DIR/bin/python" \
    --no-build-isolation \
    -e "$CSDF_DIR"
else
  echo "CSDF already importable; skipping build."
fi

cat <<EOF
H20 generation environment is ready.

Activate with:
  source "$ROOT_DIR/scripts/activate_uv_generation_h20.sh"

Built with:
  CUDA_HOME=$CUDA_HOME
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
EOF
