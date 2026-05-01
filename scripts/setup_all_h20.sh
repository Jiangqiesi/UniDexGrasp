#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== UniDexGrasp H20 (Hopper sm_90) full setup ==="
echo "Root: $ROOT_DIR"
echo ""

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || {
  echo "nvidia-smi failed — check GPU driver." >&2
  exit 1
}
echo ""

echo "--- [1/4] Generation environment ---"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
  bash "$ROOT_DIR/scripts/setup_uv_generation_h20.sh"
echo ""

echo "--- [2/4] Policy base environment ---"
bash "$ROOT_DIR/scripts/setup_uv_policy.sh"
echo ""

echo "--- [3/4] Isaac Gym ---"
bash "$ROOT_DIR/scripts/setup_isaacgym.sh"
echo ""

echo "--- [4/4] PointNet2 (sm_90) ---"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
  bash "$ROOT_DIR/scripts/setup_pointnet2_policy.sh"
echo ""

echo "=== Setup complete ==="
echo "Warm the PTX JIT cache by running smoke tests:"
echo "  bash scripts/run_policy_state_smoke.sh"
echo "  bash scripts/run_policy_vision_smoke.sh"
