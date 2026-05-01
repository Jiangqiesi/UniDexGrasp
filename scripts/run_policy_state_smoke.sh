#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"

source "$ROOT_DIR/scripts/activate_uv_policy.sh"
cd "$ROOT_DIR/dexgrasp_policy/dexgrasp"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
  --task=ShadowHandGrasp \
  --algo=ppo \
  --seed="$SEED" \
  --rl_device=cuda:0 \
  --sim_device=cuda:0 \
  --num_envs=1 \
  --episode_length=2 \
  --max_iterations=1 \
  --logdir=logs/smoke_state \
  --headless
