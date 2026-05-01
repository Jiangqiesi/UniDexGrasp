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

if [[ "${SKIP_GENERATION_DATA_CHECK:-0}" != "1" ]]; then
  python - "$CONFIG_NAME" "./$EXP_DIR" "$@" <<'PY'
import glob
import json
import os
import sys

from hydra import compose, initialize

config_name = sys.argv[1]
exp_dir = sys.argv[2]
overrides = [f"exp_dir={exp_dir}", *sys.argv[3:]]

with initialize(version_base=None, config_path="configs", job_name="data_preflight"):
    cfg = compose(config_name=config_name, overrides=overrides)

dataset_cfg = cfg["dataset"]
root_path = dataset_cfg["root_path"]
dataset_dir = dataset_cfg.get("dataset_dir", "DFCData")
mesh_data_dir = os.path.join(
    root_path,
    dataset_cfg.get("mesh_data_dir", os.path.join(dataset_dir, "meshes")),
)
pose_data_dir = os.path.join(
    root_path,
    dataset_cfg.get("pose_data_dir", os.path.join(dataset_dir, "poses")),
)
split_dir = os.path.join(root_path, "DFCData", "splits")
categories = dataset_cfg.get("categories", None)

if categories is None:
    split_paths = sorted(glob.glob(os.path.join(split_dir, "*.json")))
else:
    split_paths = [os.path.join(split_dir, f"{category}.json") for category in categories]

missing = []
checked_instances = 0
for split_path in split_paths:
    if not os.path.exists(split_path):
        missing.append(split_path)
        continue

    category = os.path.splitext(os.path.basename(split_path))[0]
    with open(split_path, "r") as f:
        split_data = json.load(f)

    for mode in ("train", "test"):
        for instance in split_data.get(mode, []):
            checked_instances += 1
            instance_mesh_dir = os.path.join(mesh_data_dir, category, instance)
            instance_pose_dir = os.path.join(pose_data_dir, category, instance)

            for filename in ("poses.npy", "pcs_table.npy"):
                path = os.path.join(instance_mesh_dir, filename)
                if not os.path.exists(path):
                    missing.append(path)

            if not os.path.isdir(instance_pose_dir):
                missing.append(instance_pose_dir)
            elif not any(name.endswith(".npz") for name in os.listdir(instance_pose_dir)):
                missing.append(os.path.join(instance_pose_dir, "*.npz"))

if missing:
    preview = "\n  ".join(missing[:20])
    extra = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
    raise SystemExit(
        "Generation data preflight failed: missing files/directories:\n"
        f"  {preview}{extra}\n\n"
        "For the error in tmp.log, the missing cache is usually pcs_table.npy.\n"
        "Generate it on the H20 server, for example:\n"
        f"  cd {os.getcwd()}\n"
        f"  python scripts/generate_object_table_pc.py --data_root_path {mesh_data_dir} --gpu_list 4 5 6 7 --n_cpu 8\n\n"
        "Set SKIP_GENERATION_DATA_CHECK=1 only if you intentionally want to bypass this check."
    )

print(f"Generation data preflight passed: {checked_instances} split instances checked.")
PY
fi

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
