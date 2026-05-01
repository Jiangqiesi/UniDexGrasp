#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy}"
PYTHON_VERSION="${UV_PYTHON_VERSION:-3.8}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
DEFAULT_ISAACGYM_PATH="$ROOT_DIR/dexgrasp_policy/thirdparty/isaacgym_preview4/isaacgym"
ISAACGYM_PATH="${ISAACGYM_PATH:-$DEFAULT_ISAACGYM_PATH}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

export UV_CACHE_DIR
export UV_HTTP_TIMEOUT

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

have_torch_policy() {
  "${PYTHON[@]}" <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__.startswith("1.13.1"), torch.__version__
PY
}

if [[ "$FORCE_REINSTALL" == "1" ]] || ! have_torch_policy; then
  "${UV_PIP[@]}" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    "torch==1.13.1+cu117" \
    "torchvision==0.14.1+cu117" \
    "torchaudio==0.13.1+cu117"
else
  echo "torch 1.13.1+cu117 already installed; skipping."
fi

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

"${UV_PIP[@]}" "setuptools<70"

"${UV_PIP[@]}" --no-deps --no-build-isolation -e "$ROOT_DIR/dexgrasp_policy"

if [[ -n "$ISAACGYM_PATH" ]]; then
  if [[ -d "$ISAACGYM_PATH/python" ]]; then
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

cat <<EOF
Policy base environment is ready.

Activate with:
  source "$ROOT_DIR/scripts/activate_uv_policy.sh"

Re-run safety:
  This script is safe to re-run. It reuses the existing venv and skips
  torch download if already installed. Use FORCE_REINSTALL=1 to rebuild
  the venv from scratch.

Host-specific note:
  - This machine currently exposes CUDA 12.x nvcc.
  - Isaac Gym Preview 3/4 and pointnet2_ops are legacy CUDA-extension stacks;
    if you need to compile them, install a matching CUDA 11.x toolkit first.
  - For pointnet2_ops specifically, torch 1.13.1+cu117 expects a matching CUDA 11.7 toolchain.
  - Isaac Gym Preview 4 was validated on this host with torch 1.13.1+cu117.
EOF
