#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy}"
POINTNET2_DIR="${POINTNET2_DIR:-$ROOT_DIR/dexgrasp_policy/thirdparty/Pointnet2_PyTorch}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
POINTNET2_SKIP_CUDA_VERSION_CHECK="${POINTNET2_SKIP_CUDA_VERSION_CHECK:-1}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export UV_CACHE_DIR
export POINTNET2_SKIP_CUDA_VERSION_CHECK

# torch 1.13.x _get_cuda_arch_flags only knows up to sm_86 (Ampere).
# H20 is sm_90 (Hopper). Patch the supported arches list so 9.0 / 9.0+PTX
# passes the version check and nvcc gets the correct -gencode flags.
_NEEDS_90_PATCH=0
if echo "$TORCH_CUDA_ARCH_LIST" | grep -q "9\.0"; then
  _NEEDS_90_PATCH=1
fi

_patch_torch_cuda_arches() {
  local _torch_cpp="$("${PYTHON[@]}" -c 'import torch.utils.cpp_extension as x; print(x.__file__)')"
  if [[ ! -f "$_torch_cpp" ]]; then
    echo "WARNING: cannot find torch cpp_extension.py to patch" >&2
    return 0
  fi
  # Only patch if 9.0 is not already present in the supported arches definition.
  if grep -q "'9\.0'" "$_torch_cpp"; then
    echo "torch cpp_extension already patched for sm_90."
    return 0
  fi
  echo "Patching $_torch_cpp to accept sm_90 (Hopper) ..."
  sed -i.bak_sm90 "s/'8\.6'/'8.6', '9.0'/" "$_torch_cpp"
  if grep -q "'9\.0'" "$_torch_cpp"; then
    echo "Patched successfully."
  else
    echo "WARNING: automated patch failed. Falling back to 8.6+PTX (PTX is forward-compatible to sm_90)." >&2
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//9.0/8.6}"
    echo "TORCH_CUDA_ARCH_LIST adjusted to: $TORCH_CUDA_ARCH_LIST"
  fi
}
patch_torch_cuda_arches() {
  if [[ "$_NEEDS_90_PATCH" == "1" ]]; then
    _patch_torch_cuda_arches
  fi
}

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing policy env: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_policy.sh first." >&2
  exit 1
fi

if [[ ! -d "$POINTNET2_DIR/.git" ]]; then
  mkdir -p "$(dirname "$POINTNET2_DIR")"
  git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$POINTNET2_DIR"
fi

env -u PYTHONPATH uv pip install --python "$ENV_DIR/bin/python" "setuptools<70"

PYTHON=(env -u PYTHONPATH "$ENV_DIR/bin/python")
have_pointnet2() {
  "${PYTHON[@]}" <<'PY' >/dev/null 2>&1
import pointnet2_ops, pointnet2
PY
}

if [[ "${FORCE_REINSTALL:-0}" == "1" ]] || ! have_pointnet2; then
  patch_torch_cuda_arches
  cat <<EOF
Building pointnet2_ops with:
  ENV_DIR=$ENV_DIR
  CUDA_HOME=$CUDA_HOME
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

This requires a CUDA toolkit that matches the PyTorch CUDA version in the policy
environment. On this host we intentionally bypass the strict version check and
compile with the system CUDA toolkit using PTX fallback for Ada.
EOF

  pushd "$POINTNET2_DIR/pointnet2_ops_lib" >/dev/null
  env -u PYTHONPATH uv pip install \
    --python "$ENV_DIR/bin/python" \
    --no-build-isolation \
    -e .
  popd >/dev/null

  pushd "$POINTNET2_DIR" >/dev/null
  env -u PYTHONPATH uv pip install \
    --python "$ENV_DIR/bin/python" \
    --no-deps \
    --no-build-isolation \
    -e .
  popd >/dev/null
else
  echo "pointnet2_ops already importable; skipping build."
fi

cat <<EOF
PointNet2 is installed.
Verify with:
  source "$ROOT_DIR/scripts/activate_uv_policy.sh"
  python -c "import pointnet2_ops, pointnet2; print(pointnet2_ops.__file__)"
EOF
