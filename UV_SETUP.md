# UniDexGrasp `uv` Environment Notes

This repository has two incompatible Python environments:

- `dexgrasp_generation`: legacy geometry stack (`torch`, `pytorch3d`, `torch_geometric`, `CSDF`)
- `dexgrasp_policy`: Isaac Gym policy stack

Do not try to merge them into one environment.

## Host Snapshot

The current host was checked before setup:

- OS: Ubuntu 22.04 (`x86_64`)
- GPU: `NVIDIA RTX 6000 Ada Generation`
- Driver: `580.126.09`
- `nvidia-smi`: available
- System `nvcc`: CUDA `12.3`
- `uv`: available

## Why The Setup Differs From README

The upstream README files assume:

- conda
- Python 3.8
- older CUDA 11.x userland
- Linux wheels and extensions from the 2022-2023 ecosystem

On this host, `uv` is required and the machine only exposes CUDA 12.3 toolkit binaries. That means:

- `dexgrasp_generation` can be installed under `uv`, but `CSDF` must be built in CPU mode unless a CUDA 11.3 toolkit is added.
- `dexgrasp_policy` can be installed under `uv`, and on this host both Isaac Gym Preview 4 and `pointnet2_ops` were validated successfully with a patched PointNet2 build flow.

## Generation Environment

Create it with:

```bash
bash scripts/setup_uv_generation.sh
source scripts/activate_uv_generation.sh
```

What this script does:

- installs Python 3.8 with `uv`
- creates `.venvs/unidexgrasp-gen`
- installs `torch==1.11.0+cu113`
- installs `pytorch3d==0.7.2` from the official Linux wheel
- installs `torch_geometric` and its matching CUDA wheels
- installs `pytorch_kinematics`, `nflows`, and `CSDF`
- builds `CSDF` in CPU mode to avoid the host CUDA 12.3 vs PyTorch CUDA 11.3 toolchain mismatch

Verification command:

```bash
source scripts/activate_uv_generation.sh
python - <<'PY'
import torch, pytorch3d, torch_geometric, csdf
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(pytorch3d.__version__)
print(torch_geometric.__version__)
print(csdf.__file__)
PY
```

## Policy Environment

Create the base environment with:

```bash
bash scripts/setup_uv_policy.sh
```

If Isaac Gym has already been unpacked locally:

```bash
ISAACGYM_PATH=/path/to/isaacgym bash scripts/setup_uv_policy.sh
```

What this script does:

- installs Python 3.8 with `uv`
- creates `.venvs/unidexgrasp-policy`
- installs `torch==1.13.1+cu117`
- installs the Python dependencies from `dexgrasp_policy/setup.py`
- installs `dexgrasp_policy` itself in editable mode
- installs `ninja`
- installs Isaac Gym from `ISAACGYM_PATH/python` if a local package is present

Current limitation on this host:

- the machine only exposes CUDA 12.3 `nvcc`
- Isaac Gym Preview 4 itself was installed and validated successfully on this host with `torch 1.13.1+cu117`
- `pointnet2_ops` was built successfully on this host by bypassing PyTorch's strict CUDA version check and compiling with `TORCH_CUDA_ARCH_LIST=8.6+PTX`

Activate the base policy environment with:

```bash
source scripts/activate_uv_policy.sh
```

Install PointNet2 with:

```bash
bash scripts/setup_pointnet2_policy.sh
```

If you need to fetch and install Isaac Gym into the policy env:

```bash
bash scripts/setup_isaacgym.sh
```

## Known Good Commands Used On This Host

GPU visibility:

```bash
nvidia-smi
```

Generation environment smoke test:

```bash
source scripts/activate_uv_generation.sh
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.randn(4, device='cuda').mean().item())
PY
```
