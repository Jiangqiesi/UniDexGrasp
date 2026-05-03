#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_H20_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy-h20}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing environment: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_policy_h20.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

source "$ENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export CUDA_HOME
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

echo "Activated H20 policy env: $ENV_DIR"
