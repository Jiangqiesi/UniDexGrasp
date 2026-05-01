#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_GEN_H20_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-gen-h20}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing environment: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_generation_h20.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

source "$ENV_DIR/bin/activate"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"

echo "Activated H20 generation env: $ENV_DIR"
