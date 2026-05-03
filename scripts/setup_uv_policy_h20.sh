#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_H20_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy-h20}"
PYTHON_VERSION="${UV_POLICY_H20_PYTHON_VERSION:-3.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
THIRDPARTY_DIR="$ROOT_DIR/dexgrasp_policy/thirdparty"
ISAACGYM_ARCHIVE_PATH="$THIRDPARTY_DIR/IsaacGym_Preview_4_Package.tar.gz"
ISAACGYM_EXTRACT_DIR="$THIRDPARTY_DIR/isaacgym_preview4"
ISAACGYM_PACKAGE_DIR="$ISAACGYM_EXTRACT_DIR/isaacgym"
DEFAULT_ISAACGYM_PATH="$ROOT_DIR/dexgrasp_policy/thirdparty/isaacgym_preview4/isaacgym"
ISAACGYM_PATH="${ISAACGYM_PATH:-$DEFAULT_ISAACGYM_PATH}"
GYMTORCH_TORCH2_PATCH="$ROOT_DIR/scripts/patches/isaacgym_gymtorch_torch2.patch"
LEGACY_POINTNET2_DIR="$ROOT_DIR/dexgrasp_policy/thirdparty/Pointnet2_PyTorch"
POINTNET2_DIR="${POINTNET2_H20_DIR:-$ROOT_DIR/dexgrasp_policy/thirdparty/Pointnet2_PyTorch_h20}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export UV_CACHE_DIR
export UV_HTTP_TIMEOUT
export POINTNET2_SKIP_CUDA_VERSION_CHECK="${POINTNET2_SKIP_CUDA_VERSION_CHECK:-1}"

if [[ -f /opt/rh/gcc-toolset-11/enable ]]; then
  # gcc-toolset's enable script is not nounset-safe on TencentOS.
  set +u
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-11/enable
  set -u
fi

export PATH="$CUDA_HOME/bin:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  _CLEAN_LD_LIBRARY_PATH=""
  IFS=':' read -r -a _LD_PARTS <<< "$LD_LIBRARY_PATH"
  for _LD_PART in "${_LD_PARTS[@]}"; do
    [[ -z "$_LD_PART" ]] && continue
    [[ "$_LD_PART" == */stubs || "$_LD_PART" == */lib64/stubs ]] && continue
    _CLEAN_LD_LIBRARY_PATH="${_CLEAN_LD_LIBRARY_PATH:+$_CLEAN_LD_LIBRARY_PATH:}$_LD_PART"
  done
  export LD_LIBRARY_PATH="$_CLEAN_LD_LIBRARY_PATH"
fi

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
assert "sm_90" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
PY
}

ensure_isaacgym() {
  if [[ "$ISAACGYM_PATH" != "$DEFAULT_ISAACGYM_PATH" ]]; then
    return 0
  fi
  mkdir -p "$THIRDPARTY_DIR"
  if [[ ! -f "$ISAACGYM_ARCHIVE_PATH" ]]; then
    curl -L https://developer.nvidia.com/isaac-gym-preview-4 -o "$ISAACGYM_ARCHIVE_PATH"
  fi
  if [[ ! -d "$ISAACGYM_PACKAGE_DIR/python" ]]; then
    mkdir -p "$ISAACGYM_EXTRACT_DIR"
    tar -xzf "$ISAACGYM_ARCHIVE_PATH" -C "$ISAACGYM_EXTRACT_DIR"
  fi
}

patch_gymtorch_torch2() {
  local gymtorch_cpp="$ISAACGYM_PATH/python/isaacgym/_bindings/src/gymtorch/gymtorch.cpp"
  if [[ ! -f "$gymtorch_cpp" ]]; then
    echo "WARNING: cannot find Isaac Gym gymtorch.cpp at '$gymtorch_cpp'." >&2
    return 0
  fi
  if grep -q "torch::from_blob(data, torch::IntArrayRef(dimensions), options)" "$gymtorch_cpp"; then
    echo "Isaac Gym gymtorch.cpp already patched for PyTorch 2.x."
    return 0
  fi
  if ! command -v patch >/dev/null 2>&1; then
    echo "Missing 'patch' command; install patch/diffutils before setting up H20 policy." >&2
    exit 1
  fi
  echo "Patching Isaac Gym gymtorch.cpp for PyTorch 2.x ..."
  patch -d "$(dirname "$gymtorch_cpp")" -p0 < "$GYMTORCH_TORCH2_PATCH"
}

patch_pointnet2_h20_arch() {
  local setup_py="$POINTNET2_DIR/pointnet2_ops_lib/setup.py"
  local pointnet2_utils="$POINTNET2_DIR/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py"
  if [[ ! -f "$setup_py" || ! -f "$pointnet2_utils" ]]; then
    echo "WARNING: cannot find PointNet2 setup.py or pointnet2_utils.py under '$POINTNET2_DIR'." >&2
    return 0
  fi
  echo "Patching PointNet2 arch handling for H20 ..."
  "${PYTHON[@]}" - "$setup_py" "$pointnet2_utils" <<'PY'
from pathlib import Path
import re
import sys

setup_py = Path(sys.argv[1])
pointnet2_utils = Path(sys.argv[2])

setup_text = setup_py.read_text()
if not re.search(r"^import os$", setup_text, flags=re.MULTILINE):
    setup_text = setup_text.replace("import glob\n", "import glob\nimport os\n", 1)
setup_text = re.sub(
    r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*["\'][^"\']*["\']',
    'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")',
    setup_text,
)
setup_text = re.sub(
    r'os\.environ\.setdefault\("TORCH_CUDA_ARCH_LIST",\s*["\'][^"\']*["\']\)',
    'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")',
    setup_text,
)
if "TORCH_CUDA_ARCH_LIST" not in setup_text:
    marker = 'exec(open(osp.join("pointnet2_ops", "_version.py")).read())'
    setup_text = setup_text.replace(
        marker,
        marker + '\n\nos.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")',
        1,
    )
setup_py.write_text(setup_text)

utils_text = pointnet2_utils.read_text()
utils_text = re.sub(
    r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*["\'][^"\']*["\']',
    'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")',
    utils_text,
)
utils_text = re.sub(
    r'os\.environ\.setdefault\("TORCH_CUDA_ARCH_LIST",\s*["\'][^"\']*["\']\)',
    'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")',
    utils_text,
)
pointnet2_utils.write_text(utils_text)
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
  echo "torch 2.4.1 CUDA 12.x with sm_90 already installed; skipping torch install."
fi

if [[ "$FORCE_REINSTALL" == "1" ]] || \
   ! have_module gym || \
   ! have_module matplotlib || \
   ! have_module h5py || \
   ! have_module cv2 || \
   ! have_module transforms3d || \
   ! have_module addict || \
   ! have_module sorcery || \
   ! have_module psutil || \
   ! have_module pynvml || \
   ! have_module pytorch_lightning || \
   ! have_module tensorboard; then
  "${UV_PIP[@]}" \
    gym \
    matplotlib==3.5.1 \
    numpy==1.23.5 \
    tb-nightly \
    tqdm==4.66.5 \
    ipdb \
    "pytorch-lightning<2" \
    opencv-python \
    transforms3d==0.4.1 \
    addict \
    yapf \
    h5py \
    sorcery \
    psutil \
    pynvml \
    ninja
else
  echo "base policy Python deps already installed; skipping."
fi

"${UV_PIP[@]}" "setuptools<70"
"${UV_PIP[@]}" --no-deps --no-build-isolation -e "$ROOT_DIR/dexgrasp_policy"

if [[ -n "$ISAACGYM_PATH" ]]; then
  ensure_isaacgym
  if [[ -d "$ISAACGYM_PATH/python" ]]; then
    patch_gymtorch_torch2
    "${UV_PIP[@]}" --no-build-isolation -e "$ISAACGYM_PATH/python"
  else
    echo "ISAACGYM_PATH is set but '$ISAACGYM_PATH/python' does not exist." >&2
  fi
else
  cat <<'EOF'
Isaac Gym is not installed yet.
Set ISAACGYM_PATH to your local Isaac Gym directory and rerun this script to install it.
EOF
fi

if [[ ! -d "$POINTNET2_DIR/.git" ]]; then
  mkdir -p "$(dirname "$POINTNET2_DIR")"
  if [[ -d "$LEGACY_POINTNET2_DIR/.git" ]]; then
    git clone "$LEGACY_POINTNET2_DIR" "$POINTNET2_DIR"
  else
    git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$POINTNET2_DIR"
  fi
fi
patch_pointnet2_h20_arch

cat <<EOF
Building H20 PointNet2 with:
  ENV_DIR=$ENV_DIR
  POINTNET2_DIR=$POINTNET2_DIR
  CUDA_HOME=$CUDA_HOME
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
EOF

pushd "$POINTNET2_DIR/pointnet2_ops_lib" >/dev/null
env -u PYTHONPATH \
  CUDA_HOME="$CUDA_HOME" \
  FORCE_CUDA=1 \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  MAX_JOBS="${MAX_JOBS:-8}" \
  uv pip install --python "$ENV_DIR/bin/python" \
    --no-build-isolation \
    --force-reinstall \
    -e .
popd >/dev/null

pushd "$POINTNET2_DIR" >/dev/null
env -u PYTHONPATH uv pip install \
  --python "$ENV_DIR/bin/python" \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
  -e .
popd >/dev/null

cat <<EOF
H20 policy environment is ready.

Activate with:
  source "$ROOT_DIR/scripts/activate_uv_policy_h20.sh"

Built with:
  CUDA_HOME=$CUDA_HOME
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
  POINTNET2_DIR=$POINTNET2_DIR
EOF
