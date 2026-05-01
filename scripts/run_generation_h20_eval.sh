#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-eval_config}"
EXP_DIR="${2:-eval_h20}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT_DIR/scripts/activate_uv_generation_h20.sh"
cd "$ROOT_DIR/dexgrasp_generation"

python ./network/eval.py --config-name "$CONFIG_NAME" --exp-dir "./$EXP_DIR"
