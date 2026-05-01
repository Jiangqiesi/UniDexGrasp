#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_generation_train.sh <config_name> <exp_dir>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="$1"
EXP_DIR="$2"

source "$ROOT_DIR/scripts/activate_uv_generation.sh"
cd "$ROOT_DIR/dexgrasp_generation"

python ./network/train.py --config-name "$CONFIG_NAME" --exp-dir "./$EXP_DIR"
