#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_GEN_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-gen}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing environment: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_generation.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

source "$ENV_DIR/bin/activate"

SITE_PACKAGES="$("$ENV_DIR/bin/python" - <<'PY'
import site
for path in site.getsitepackages():
    if "site-packages" in path:
        print(path)
        break
PY
)"

TORCH_LIB="$SITE_PACKAGES/torch/lib"
NVIDIA_CUBLAS_LIB="$SITE_PACKAGES/nvidia/cublas/lib"
NVIDIA_CUSPARSE_LIB="$SITE_PACKAGES/nvidia/cusparse/lib"

export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_CUBLAS_LIB:$NVIDIA_CUSPARSE_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

echo "Activated generation env: $ENV_DIR"
