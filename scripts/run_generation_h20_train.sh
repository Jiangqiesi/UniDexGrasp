#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_generation_h20_train.sh <config_name> <exp_dir> [hydra_overrides...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="$1"
EXP_DIR="$2"
shift 2

source "$ROOT_DIR/scripts/activate_uv_generation_h20.sh"
cd "$ROOT_DIR/dexgrasp_generation"

GPU_LIST="${GENERATION_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_LIST="${GPU_LIST//[[:space:]]/}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
  NUM_GPUS="${#GPU_ARRAY[@]}"
else
  NUM_GPUS=1
fi

if (( NUM_GPUS > 1 )); then
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    ./network/train.py --config-name "$CONFIG_NAME" --exp-dir "./$EXP_DIR" "$@"
else
  python ./network/train.py --config-name "$CONFIG_NAME" --exp-dir "./$EXP_DIR" "$@"
fi
