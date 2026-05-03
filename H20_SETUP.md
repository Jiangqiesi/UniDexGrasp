# UniDexGrasp H20 Server Setup

这份文档面向一台 `8 * NVIDIA H20` 服务器，目标是最快把
UniDexGrasp 配到能跑 smoke test。它假设代码已经从旧机器迁移到新机器，
不在服务器上直接开发。

> 重要：UniDexGrasp 是 2023 年左右的老 CUDA / PyTorch / Isaac Gym 栈。
> H20 是 Hopper 架构 GPU，按 `sm_90` 处理。旧脚本里出现的
> `8.6+PTX` 是给 Ampere/Ada 类机器用的，不适合 H20。

## 已确认服务器信息

当前 H20 服务器：

```bash
Linux TENCENT64.site 5.4.241-1-tlinux4-0017.7 x86_64
TencentOS Server 3.2 (Final)
ID_LIKE="rhel fedora centos"
```

已知状态：

- `PROJECT_ROOT`：`/apdcephfs_hzlf/share_304819645/chenmingkang/FF/UniDexGrasp`
- `uv`：当前没有安装。
- `nvcc`：`/usr/local/cuda/bin/nvcc`
- CUDA toolkit：`12.8`，真实目录 `/usr/local/cuda-12.8`
- `/usr/local/cuda -> /etc/alternatives/cuda`
- Isaac Gym tarball：已在仓库里。
- generation 数据路径：本文件暂不处理。

服务器上先设置：

```bash
export PROJECT_ROOT=/apdcephfs_hzlf/share_304819645/chenmingkang/FF/UniDexGrasp
export CUDA128_HOME=/usr/local/cuda-12.8
```

## 0. 服务器基础检查

```bash
uname -a
cat /etc/os-release
which uv || true
which nvcc || true
nvcc --version || true
ls -l /usr/local | grep cuda || true
nvidia-smi
```

H20 服务器的驱动层只要 `nvidia-smi` 正常即可。你给出的机器是：

- Driver: `535.247.01`
- `nvidia-smi` reported CUDA: `12.8`
- GPU: `NVIDIA H20`
- Memory: about `97 GB` per GPU

这里的 `CUDA Version: 12.8` 是驱动可支持的最高 CUDA runtime，不等于系统
已经安装了 CUDA toolkit。编译 PointNet2、CSDF 这类扩展时真正看的是
`CUDA_HOME` / `nvcc`。

## 1. 系统依赖

这台机器是 TencentOS / RHEL 系，不用 `apt`。如果有 root 权限，优先用
`dnf` 或 `yum`：

```bash
dnf install -y \
  gcc gcc-c++ make git git-lfs curl rsync tmux htop unzip \
  cmake ninja-build pkgconf-pkg-config \
  mesa-libGL glib2 libXrender libXext libXi libXrandr libXcursor \
  libglvnd libglvnd-egl vulkan-loader vulkan-tools || \
yum install -y \
  gcc gcc-c++ make git git-lfs curl rsync tmux htop unzip \
  cmake ninja-build pkgconf-pkg-config \
  mesa-libGL glib2 libXrender libXext libXi libXrandr libXcursor \
  libglvnd libglvnd-egl vulkan-loader vulkan-tools

git lfs install
```

服务器 PATH 里已经有 `/opt/rh/gcc-toolset-11/root/usr/bin`，这很好。若后面
编译扩展时发现 `gcc --version` 太旧，先执行：

```bash
set +u
source /opt/rh/gcc-toolset-11/enable
set -u
gcc --version
g++ --version
```

`gcc-toolset-11/enable` 在 TencentOS 上不是 `set -u` 安全的；如果当前 shell 或
脚本开了 `set -u` / `set -euo pipefail`，必须像上面这样临时关闭 nounset。

安装 `uv`（如果已安装则跳过，不会重复下载覆盖）：

```bash
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

当前没有 CUDA 11.8，但有 CUDA 12.8。H20 路线统一用 CUDA 12.8 编译本地
CUDA 扩展，并设置 `TORCH_CUDA_ARCH_LIST=9.0+PTX` 生成 Hopper (`sm_90`) 目标。
Policy 环境不要继续依赖 `torch==1.13.1+cu117` 做 GPU 计算；它只能看到 H20，
但自身 CUDA kernel 不支持 `sm_90`。

## 2. 代码和数据

当前已知代码目标路径是：

```bash
export PROJECT_ROOT=/apdcephfs_hzlf/share_304819645/chenmingkang/FF/UniDexGrasp
cd "$PROJECT_ROOT"
pwd
```

如果之后需要重新从旧机器迁移，建议用 `rsync` 并排除虚拟环境、缓存和 git logs。

确认 Isaac Gym 包存在：

```bash
cd "$PROJECT_ROOT"
test -f dexgrasp_policy/thirdparty/IsaacGym_Preview_4_Package.tar.gz
```

如果这个文件不存在，`scripts/setup_isaacgym.sh` 会尝试从 NVIDIA 链接下载，
但该链接可能需要登录或返回 HTML，所以最快做法是直接把 tar.gz 从旧机器带过去。

generation 数据路径这次先不处理。如果之后要跑 generation 训练/评估，再保证
下面两个路径能解析即可：

```bash
cd "$PROJECT_ROOT"
test -d dexgrasp_generation/data/DFCData
test -d dexgrasp_generation/data/mjcf
```

## 3. 关键 H20 环境变量

在 H20 上编译 CUDA 扩展前统一设置：

```bash
export CUDA_HOME="$CUDA128_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
```

`TORCH_CUDA_ARCH_LIST="9.0+PTX"` 会让 nvcc 为 Hopper 架构 (`sm_90`) 生成优化
代码。H20 专用脚本使用支持 `sm_90` 的 PyTorch 2.4.1 CUDA 12.x wheel；如果你在
旧 `torch==1.13.1+cu117` 环境中设置这个变量，只能影响 PointNet2 这类本地扩展，
不能让 PyTorch 自己支持 H20。

## 4. 安装 generation 环境

仓库原版 generation 环境是 legacy proposal generation 栈，会安装
`torch==1.11.0+cu113`。它能作为原版依赖对照，但不能在 H20 上可靠跑 GPU
kernel：

```bash
cd "$PROJECT_ROOT"
unset PYTHONPATH
bash scripts/setup_uv_generation.sh
source scripts/activate_uv_generation.sh
```

验证：

```bash
python - <<'PY'
import torch, pytorch3d, torch_geometric, csdf
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("pytorch3d:", pytorch3d.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("csdf:", csdf.__file__)
PY
```

在当前 H20 服务器上，generation import 已经验证到如下状态：

```text
torch: 1.11.0+cu113
cuda available: True
NVIDIA H20 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
pytorch3d: 0.7.2
torch_geometric: 2.2.0
csdf: .../dexgrasp_generation/thirdparty/CSDF/csdf/__init__.py
```

这表示 Python 包导入成功，但旧 PyTorch 只支持到 `sm_86`，不能把 H20 当作
可用 GPU 计算环境。`torch.cuda.is_available() == True` 在这里只说明驱动和
CUDA runtime 能看到 GPU，不代表当前 `torch==1.11.0+cu113` 能在 H20 上运行
CUDA kernel。

如果现在要先跑 generation，请改用 H20 专用 generation 环境：

```bash
cd "$PROJECT_ROOT"
unset PYTHONPATH
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
bash scripts/setup_uv_generation_h20.sh
source scripts/activate_uv_generation_h20.sh
```

这个环境会新建 `.venvs/unidexgrasp-gen-h20`，不覆盖旧的
`.venvs/unidexgrasp-gen`。它安装：

- Python 3.10
- `torch==2.4.1` / `torchvision==0.19.1` / `torchaudio==2.4.1` from cu124 wheels
- `numpy==1.26.4`，避免 `transforms3d==0.4.1` 在 NumPy 2.x 下报
  `np.maximum_sctype was removed`
- PyG 2.6.1 及匹配 `torch-2.4.0+cu124` 的 `torch-scatter` / `torch-cluster`
  扩展 wheel
- TencentOS glibc 较老，`pyg-lib` / `torch-sparse` 预编译 wheel 可能报
  `GLIBC_2.29 not found`。generation 的 PointNet++ 路径不需要这两个可选包，
  脚本默认会卸载它们；如需保留，设置 `KEEP_PYG_OPTIONAL=1`
- PyTorch3D 从官方 `stable` 分支源码编译
- CSDF 用 `CUDA_HOME=/usr/local/cuda-12.8` 和 `TORCH_CUDA_ARCH_LIST=9.0+PTX`
  重新编译
- CSDF 已对 PyTorch 2.x 做本地兼容补丁：把旧 `CHECK_EQ` CUDA 检查宏改成
  `TORCH_CHECK`，否则会在编译时报 `identifier "CHECK_EQ" is undefined`
- 为兼容 CSDF 的旧 `setup.py`，脚本会固定 `setuptools<70`，否则可能报
  `ModuleNotFoundError: No module named 'pkg_resources'`
- 脚本默认会复用已有 `.venvs/unidexgrasp-gen-h20`，并跳过已经能 import 的大包
  和本地扩展，便于失败后重跑继续配置。如果确实要清空环境并强制重装，用
  `FORCE_REINSTALL=1 bash scripts/setup_uv_generation_h20.sh`

验证 H20 generation 环境：

```bash
source scripts/activate_uv_generation_h20.sh
python - <<'PY'
import torch, pytorch3d, torch_geometric, csdf
from torch_geometric.nn import fps, radius
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
print("cuda matmul:", (x @ x).mean().item())
pos = torch.randn(128, 3, device="cuda")
batch = torch.zeros(128, dtype=torch.long, device="cuda")
idx = fps(pos, batch, ratio=0.5)
row, col = radius(pos, pos[idx], 0.5, batch, batch[idx], max_num_neighbors=16)
print("pyg fps/radius:", idx.numel(), row.numel(), col.numel())
print("pytorch3d:", pytorch3d.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("csdf:", csdf.__file__)
PY
```

跑 generation 训练时不要用旧的 `scripts/run_generation_train.sh`，它会激活
旧环境。请用：

```bash
bash scripts/run_generation_h20_train.sh ipdf_config h20_ipdf_smoke
```

上面这条在没有设置多卡 `GENERATION_GPUS` / `CUDA_VISIBLE_DEVICES` 时是单进程
训练，只用配置里的 `cuda_id: 0`。如果要让同一个 generation 训练任务使用指定
多张 GPU，显式传 GPU 列表即可，脚本会自动改用 `torch.distributed.run` / DDP：

```bash
GENERATION_GPUS=0,1,2,3 \
bash scripts/run_generation_h20_train.sh ipdf_config h20_ipdf_4gpu
```

也可以直接用已有的 `CUDA_VISIBLE_DEVICES`：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/run_generation_h20_train.sh ipdf_config h20_ipdf_4gpu
```

`batch_size` 是每个 GPU 进程的 batch size；例如 4 卡且 `batch_size=16` 时，
全局有效 batch size 是 `64`。如果想保持原来的全局 batch size，可以覆盖：

```bash
GENERATION_GPUS=0,1,2,3 \
bash scripts/run_generation_h20_train.sh ipdf_config h20_ipdf_4gpu batch_size=4
```

如果只想先确认链路，可以临时把配置覆盖小一点：

```bash
source scripts/activate_uv_generation_h20.sh
cd "$PROJECT_ROOT/dexgrasp_generation"
python ./network/train.py --config-name ipdf_config --exp-dir ./h20_ipdf_smoke \
  total_epoch=1 batch_size=2 num_workers=0 freq.plot=1 freq.save=10 freq.test=100000
```

`network/train.py` 和 `network/eval.py` 已支持把额外的 `key=value` 参数透传给
Hydra，所以这些临时覆盖项会生效。

如果多卡命令已经出现 `[rank0]` / `[rank1]` 等日志，说明 DDP 已经拉起来了。
例如下面这种报错不是多卡启动问题，而是 generation 数据缓存不完整：

```text
FileNotFoundError: ... data/DFCData/meshdatav3/.../pcs_table.npy
ValueError: cannot reshape array of size 61536 into shape (100,4000,3)
```

`ipdf_data.yaml` 默认从 `data/DFCData/meshdatav3` 读取每个物体的
`poses.npy` 和 `pcs_table.npy`。`cannot reshape array ... into shape
(100,4000,3)` 表示某个 `pcs_table.npy` 的 header 还在，但文件内容只写了一小段，
通常是生成缓存时进程被中断留下的半文件。`run_generation_h20_train.sh` 现在会在
启动 `torchrun` 前做一次单进程预检，缺文件或坏 `.npy` 都会直接打印具体路径。

如果缺很多 `pcs_table.npy`，先在服务器上生成缓存：

```bash
cd "$PROJECT_ROOT/dexgrasp_generation"
source ../scripts/activate_uv_generation_h20.sh
python scripts/generate_object_table_pc.py \
  --data_root_path data/DFCData/meshdatav3 \
  --gpu_list 4 5 6 7 \
  --n_cpu 8
```

如果只是单个或少量 `pcs_table.npy` 损坏，先从预检报错路径里取出 object code。
例如路径是：

```text
data/DFCData/meshdatav3/core/abc123/pcs_table.npy
```

则 object code 是 `core/abc123`，只重生成这个对象即可：

```bash
python scripts/generate_object_table_pc.py \
  --data_root_path data/DFCData/meshdatav3 \
  --object_code core/abc123 \
  --gpu_list 4 \
  --n_cpu 1
```

如果只是临时确认训练代码、想跳过预检，可以设置
`SKIP_GENERATION_DATA_CHECK=1`，但真正训练仍需要这些文件存在。

## 5. 安装 policy 环境

Policy 是 Isaac Gym 栈，但 H20 不能继续用仓库原版
`torch==1.13.1+cu117` 做 CUDA 计算。这个 wheel 只能支持到 `sm_86`；
在 H20 上会出现：

```text
NVIDIA H20 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities ... sm_86.
```

`TORCH_CUDA_ARCH_LIST=9.0+PTX` 只能影响本地编译的扩展，不能给已经安装好的
PyTorch wheel 补上 `sm_90`。H20 policy 环境使用单独的 Python 3.8 venv、
`torch==2.4.1` CUDA 12.4 wheel、Isaac Gym Preview 4，并在独立的
`Pointnet2_PyTorch_h20` clone 里重编 PointNet2，避免覆盖旧 policy 环境的
PointNet2 `.so`。

```bash
cd "$PROJECT_ROOT"
unset PYTHONPATH

CUDA_HOME="$CUDA128_HOME" \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

这个脚本可以安全重复执行：
- `setup_uv_policy_h20.sh` — 复用 `.venvs/unidexgrasp-policy-h20`
- 安装 `torch==2.4.1` / `torchvision==0.19.1` / `torchaudio==2.4.1`
  from cu124 wheels
- 如果复用已有 `.venvs/unidexgrasp-policy-h20`，日志里可能显示
  `torch==2.4.1+cu121`。脚本会接受任何 `torch==2.4.1`、`torch.version.cuda`
  为 `12.x`、且 `torch.cuda.get_arch_list()` 包含 `sm_90` 的环境；如果想强制
  换成 cu124 wheel，用 `FORCE_REINSTALL=1` 重跑 setup。
- 安装 Isaac Gym Preview 4
- Isaac Gym 的 `gymtorch.cpp` 已加 PyTorch 2.x 兼容分支，用 `torch::from_blob`
  包装 Gym tensor
- 在 `dexgrasp_policy/thirdparty/Pointnet2_PyTorch_h20` 中重编 PointNet2
- PointNet2 的 `setup.py` 和 JIT fallback 已 patch：不再覆盖
  `TORCH_CUDA_ARCH_LIST=9.0+PTX`

如果需要强制重建某个环境，设置 `FORCE_REINSTALL=1` 前缀执行对应脚本。

验证时不要只看 import；必须跑一个 PyTorch CUDA kernel 和一个 PointNet2 CUDA
kernel。Isaac Gym 仍然先 import：

```bash
source scripts/activate_uv_policy_h20.sh
python - <<'PY'
from isaacgym import gymapi
import torch
from isaacgym import gymtorch
from pointnet2_ops import pointnet2_utils

print("gym:", gymapi.acquire_gym())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda")
print("cuda matmul:", (x @ x).mean().item())
points = torch.rand(2, 128, 3, device="cuda").contiguous()
idx = pointnet2_utils.furthest_point_sample(points, 32)
torch.cuda.synchronize()
print("pointnet2 fps:", tuple(idx.shape), idx.device)
print("pointnet2:", pointnet2_utils.__file__)
PY
```

如果 PointNet2 编译失败，先确认：

```bash
echo "$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
echo "$TORCH_CUDA_ARCH_LIST"
```

期望 `CUDA_HOME=/usr/local/cuda-12.8`，`TORCH_CUDA_ARCH_LIST=9.0+PTX`。

## 6. Smoke test

先用 1 张卡跑最小 policy smoke：

```bash
cd "$PROJECT_ROOT"
POLICY_ACTIVATE_SCRIPT="$PWD/scripts/activate_uv_policy_h20.sh" \
GPU_ID=0 \
bash scripts/run_policy_state_smoke.sh

POLICY_ACTIVATE_SCRIPT="$PWD/scripts/activate_uv_policy_h20.sh" \
GPU_ID=0 \
bash scripts/run_policy_vision_smoke.sh
```

这两个 smoke 覆盖的东西不一样：

- `run_policy_state_smoke.sh` 不启用相机；在 `--headless` 下会把
  `graphics_device_id` 设成 `-1`，主要验证 PyTorch、PointNet2、Isaac Gym
  GPU PhysX / GPU pipeline。
- `run_policy_vision_smoke.sh` 会创建 Isaac Gym camera sensors，必须能通过
  Vulkan 创建 NVIDIA graphics device。即使没有窗口、没有 `DISPLAY`，它也需要
  服务器的 NVIDIA Vulkan ICD / graphics driver layer 正常。

先在服务器上做一次 Vulkan 预检：

```bash
cd "$PROJECT_ROOT"
bash scripts/check_policy_vulkan.sh
```

如果 `tmp.log` 里类似下面这样：

```text
No GPU devices found.
[Error] [carb.gym.plugin] Failed to create Nvf device in createNvfGraphics.
*** Failed to create sim
```

说明 H20 policy Python 环境基本已经走到 Isaac Gym 建 sim 这一步了；当前失败点是
系统 Vulkan / graphics 层没有暴露 NVIDIA 设备，不是 `torch==2.4.1`、
PointNet2 或 CUDA arch 的问题。先修 `check_policy_vulkan.sh` 报出的系统项，再
重跑 vision smoke。

如果 smoke test 通过，再考虑多卡训练。当前脚本默认是单进程单卡，
8 张 H20 不是自动并行；需要你自己按实验拆 `GPU_ID` 或改训练脚本。

## 7. 常见失败和最快处理

### `sm_90 is not compatible` 或 `no kernel image is available`

说明当前二进制/扩展没有 H20 可用的 CUDA target。先检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

如果是 `torch==1.13.1+cu117`，并且 `get_arch_list()` 只到 `sm_86`，不要继续在
这个环境上修 PointNet2；PyTorch 自己的 CUDA kernel 也跑不了 H20。切换到：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

如果已经是 `torch==2.4.1` 且 `get_arch_list()` 包含 `sm_90`，但 PointNet2 仍报
`no kernel image is available`，说明加载到的是旧 PointNet2 `.so` 或扩展没按
`9.0+PTX` 重编，直接强制重装 H20 policy 环境：

```bash
FORCE_REINSTALL=1 \
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

### `pointnet2_ops` 编译失败：`Unknown CUDA arch (9.0+PTX) or GPU not supported`

`torch==1.13.1+cu117` 的 `_get_cuda_arch_flags` 只认到 `sm_86`（Ampere），不认
`9.0`（Hopper）。旧的 `setup_pointnet2_policy.sh` 可以 patch 这个检查并让
PointNet2 编译，但它不能解决 `torch==1.13.1+cu117` 本身不支持 H20 的问题。
H20 policy 直接使用 `setup_uv_policy_h20.sh`。

重新编译：

```bash
cd "$PROJECT_ROOT"
CUDA_HOME="$CUDA128_HOME" \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

如果仍然失败，可以临时退到 `8.6+PTX` 生成 PTX fallback，但仍然要在 H20 policy
环境里重编：

```bash
CUDA_HOME="$CUDA128_HOME" \
TORCH_CUDA_ARCH_LIST="8.6+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

### policy vision smoke：`Failed to create Nvf device`

这是 Isaac Gym camera rendering 的 Vulkan graphics device 创建失败。常见原因是：

- 只装了 CUDA / compute driver，没装 NVIDIA Vulkan ICD / graphics driver libs。
- 机器在容器里运行，但容器没有 `graphics` driver capability，例如缺少
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` 或 `all`。
- `/usr/share/vulkan/icd.d/` 或 `/etc/vulkan/icd.d/` 没有 NVIDIA ICD JSON，或
  JSON 指向的 NVIDIA Vulkan library 不存在。
- `vulkaninfo --summary` 看不到 NVIDIA H20。

先跑：

```bash
bash scripts/check_policy_vulkan.sh
```

期望能看到 NVIDIA ICD JSON，并且 `vulkaninfo --summary` 中列出 NVIDIA GPU。
如果 `vulkaninfo` 命令不存在，先安装 `vulkan-tools`。如果 `vulkaninfo` 存在但
看不到 NVIDIA 设备，需要让管理员补齐 NVIDIA driver 的 Vulkan / graphics 组件；
`nvidia-smi` 正常只说明 compute/management 层正常，不等于 Isaac Gym 相机渲染可用。

临时只验证 policy 非视觉链路，可以先跑：

```bash
POLICY_ACTIVATE_SCRIPT="$PWD/scripts/activate_uv_policy_h20.sh" \
GPU_ID=0 \
bash scripts/run_policy_state_smoke.sh
```

### PointNet2 编译失败：`Unsupported gpu architecture 'compute_37'`

这是 PointNet2 旧版 `setup.py` 或 `pointnet2_utils.py` 硬编码了老架构：

```text
TORCH_CUDA_ARCH_LIST=3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5
nvcc fatal   : Unsupported gpu architecture 'compute_37'
```

CUDA 12.8 已经不支持 `compute_37`。`setup_uv_policy_h20.sh` 会自动 patch
`Pointnet2_PyTorch_h20/pointnet2_ops_lib/setup.py` 和
`Pointnet2_PyTorch_h20/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py`，让它们
只在环境变量未设置时才默认使用 `9.0+PTX`。如果你已经有旧的
`Pointnet2_PyTorch_h20` 目录，直接重跑脚本即可：

```bash
CUDA_HOME="$CUDA128_HOME" \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

### Isaac Gym 下载或解压失败

确认这个文件真实存在且不是 HTML：

```bash
file dexgrasp_policy/thirdparty/IsaacGym_Preview_4_Package.tar.gz
tar -tzf dexgrasp_policy/thirdparty/IsaacGym_Preview_4_Package.tar.gz | head
```

如果不存在，从旧机器复制该 tar.gz，不要依赖在线下载。

### generation GPU 路径失败

如果你在旧 `.venvs/unidexgrasp-gen` 里看到 H20 `sm_90` 不兼容 warning，这是
预期现象。切到 H20 generation 环境：

```bash
source scripts/activate_uv_generation_h20.sh
```

如果 H20 generation 环境里的 PyTorch3D / CSDF 编译失败，再按下面方向处理：

- PyTorch3D 改为匹配新 PyTorch 的源码编译或可用 wheel。
- PyG 扩展换成匹配新 PyTorch/CUDA 的 wheel。
- CSDF 用同一个 `CUDA_HOME` 和 `TORCH_CUDA_ARCH_LIST=9.0+PTX` 重新编译。

### `np.maximum_sctype was removed`

这是 `transforms3d==0.4.1` 与 NumPy 2.x 不兼容。H20 generation 脚本已固定
`numpy==1.26.4`。如果手动修：

```bash
source scripts/activate_uv_generation_h20.sh
uv pip install --python .venvs/unidexgrasp-gen-h20/bin/python "numpy==1.26.4"
```

### `cannot import name 'PointConv'`

这是 PyG 2.6 API 变化。老代码从 `torch_geometric.nn` 导入 `PointConv`，新版本
使用 `PointNetConv`。仓库已在
`dexgrasp_generation/network/models/backbones/pointnetpp_encoder.py` 做兼容导入：

```python
try:
    from torch_geometric.nn import PointConv
except ImportError:
    from torch_geometric.nn import PointNetConv as PointConv
```

### CUDA 12.8 编译 PointNet2 失败

当前最快路线是用 CUDA 12.8 编译、跳过版本检查：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
bash scripts/setup_uv_policy_h20.sh
```

如果仍失败，保存完整报错。下一步通常是检查 `gcc/g++`、`nvcc`、`torch.version.cuda`
和 PointNet2 编译日志，而不是退回 `torch==1.13.1+cu117`。

## 8. 最短命令清单

如果现在优先跑 generation，服务器上最快执行：

```bash
export PROJECT_ROOT=/apdcephfs_hzlf/share_304819645/chenmingkang/FF/UniDexGrasp
export CUDA128_HOME=/usr/local/cuda-12.8
if [ -f /opt/rh/gcc-toolset-11/enable ]; then
  set +u
  source /opt/rh/gcc-toolset-11/enable
  set -u
fi
export PATH="$HOME/.local/bin:$CUDA128_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA128_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"
unset PYTHONPATH

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
nvcc --version

CUDA_HOME="$CUDA128_HOME" \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
bash scripts/setup_uv_generation_h20.sh

source scripts/activate_uv_generation_h20.sh
python - <<'PY'
import torch, pytorch3d, torch_geometric, csdf
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
print((x @ x).mean().item())
print("pytorch3d:", pytorch3d.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("csdf:", csdf.__file__)
PY

cd "$PROJECT_ROOT/dexgrasp_generation"
python ./network/train.py --config-name ipdf_config --exp-dir ./h20_ipdf_smoke \
  total_epoch=1 batch_size=2 num_workers=0 freq.plot=1 freq.save=10 freq.test=100000
```

## 9. 仍需人工决策的点

如果目标只是最快跑通已有 policy 代码，先按本文 H20 路线走：

- Python 3.8
- policy: `torch==2.4.1` from cu124 wheels
- Isaac Gym Preview 4
- CUDA 12.8 toolkit 编译 PointNet2
- `TORCH_CUDA_ARCH_LIST=9.0+PTX`

如果目标是长期在 H20 上大规模训练，建议后续专门开一轮环境升级：

- 验证更高版本 PyTorch + CUDA 12.x 是否还能稳定跑 Isaac Gym Preview 4。
- 重编 PointNet2、PyTorch3D、CSDF。
- 固化新的 smoke test 和多卡训练入口，避免每次靠手工环境变量。
