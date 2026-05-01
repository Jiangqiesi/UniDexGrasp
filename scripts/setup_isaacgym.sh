#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${UV_POLICY_ENV_DIR:-$ROOT_DIR/.venvs/unidexgrasp-policy}"
THIRDPARTY_DIR="$ROOT_DIR/dexgrasp_policy/thirdparty"
ARCHIVE_PATH="$THIRDPARTY_DIR/IsaacGym_Preview_4_Package.tar.gz"
EXTRACT_DIR="$THIRDPARTY_DIR/isaacgym_preview4"
PACKAGE_DIR="$EXTRACT_DIR/isaacgym"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"

export UV_CACHE_DIR
export UV_HTTP_TIMEOUT

mkdir -p "$THIRDPARTY_DIR"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  curl -L https://developer.nvidia.com/isaac-gym-preview-4 -o "$ARCHIVE_PATH"
fi

if [[ ! -d "$PACKAGE_DIR/python" ]]; then
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing policy env: $ENV_DIR" >&2
  echo "Run scripts/setup_uv_policy.sh first." >&2
  exit 1
fi

env -u PYTHONPATH uv pip install --python "$ENV_DIR/bin/python" ninja "setuptools<70"
env -u PYTHONPATH uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -e "$PACKAGE_DIR/python"

cat <<EOF
Isaac Gym Preview 4 is installed from:
  $PACKAGE_DIR

Try:
  source "$ROOT_DIR/scripts/activate_uv_policy.sh"
  python -c "from isaacgym import gymapi; print(gymapi.acquire_gym())"
EOF
