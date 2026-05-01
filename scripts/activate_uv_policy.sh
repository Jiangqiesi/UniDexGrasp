#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing environment: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_policy.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

source "$ENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

echo "Activated policy env: $ENV_DIR"
