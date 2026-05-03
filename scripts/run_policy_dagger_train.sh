#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-20}"
EPISODE_LENGTH="${EPISODE_LENGTH:-250}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
LOGDIR="${LOGDIR:-logs/dagger_train}"
BACKBONE_TYPE="${BACKBONE_TYPE:-pn}"
EXPERT_MODEL="${EXPERT_MODEL:-$ROOT_DIR/dexgrasp_policy/example_model/model.pt}"
POLICY_ACTIVATE_SCRIPT="${POLICY_ACTIVATE_SCRIPT:-$ROOT_DIR/scripts/activate_uv_policy.sh}"

source "$POLICY_ACTIVATE_SCRIPT"
cd "$ROOT_DIR/dexgrasp_policy/dexgrasp"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
  --task=ShadowHandRandomLoadVision \
  --algo=dagger \
  --seed="$SEED" \
  --rl_device=cuda:0 \
  --sim_device=cuda:0 \
  --num_envs="$NUM_ENVS" \
  --episode_length="$EPISODE_LENGTH" \
  --max_iterations="$MAX_ITERATIONS" \
  --logdir="$LOGDIR" \
  --expert_model_dir="$EXPERT_MODEL" \
  --headless \
  --vision \
  --backbone_type="$BACKBONE_TYPE"
