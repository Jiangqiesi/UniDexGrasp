# UniDexGrasp 运行手册

这份文档是当前主机上可直接执行的运行说明。

内容包括：

- `uv` 环境搭建
- 数据要求
- smoke test
- 训练命令
- 测试命令
- 可直接运行的脚本

## 已验证环境

以下状态已在这台机器上验证通过：

- Ubuntu 22.04
- `nvidia-smi` 可用
- GPU：`NVIDIA RTX 6000 Ada Generation`
- 系统 CUDA 工具链：`12.3`
- `uv` 可用
- `dexgrasp_policy` 基础环境可用
- Isaac Gym Preview 4 可用
- `pointnet2_ops` 可用
- 状态任务 PPO smoke test 可用
- 视觉任务 PPO smoke test 可用

## 仓库结构

本仓库有两个核心子项目：

- `dexgrasp_generation`：抓取 proposal 生成
- `dexgrasp_policy`：基于 Isaac Gym 的策略学习

这两个子项目需要不同的 Python 环境，不要合并成一个环境。

## 一次性安装

### 1. 检查 GPU

```bash
nvidia-smi
```

### 2. 安装 Generation 环境

```bash
bash scripts/setup_uv_generation.sh
source scripts/activate_uv_generation.sh
```

快速验证：

```bash
source scripts/activate_uv_generation.sh
python - <<'PY'
import torch, pytorch3d, torch_geometric, csdf
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(pytorch3d.__version__)
print(torch_geometric.__version__)
print(csdf.__file__)
PY
```

### 3. 安装 Policy 环境

```bash
bash scripts/setup_uv_policy.sh
bash scripts/setup_isaacgym.sh
bash scripts/setup_pointnet2_policy.sh
source scripts/activate_uv_policy.sh
```

快速验证：

```bash
source scripts/activate_uv_policy.sh
python - <<'PY'
from isaacgym import gymapi
from pointnet2_ops import pointnet2_utils
print(gymapi.acquire_gym())
print(pointnet2_utils.__file__)
PY
```

## 数据要求

### Generation

`dexgrasp_generation/data` 下需要准备：

- `DFCData`
- `mjcf`

可以直接使用符号链接，不要求数据必须物理放在仓库内。下面两种方式都可以：

```bash
# 方式 1：整个 data 目录直接链接到另一块盘上的数据目录
rm -rf dexgrasp_generation/data
ln -s ~/data dexgrasp_generation/data

# 方式 2：只链接需要的子目录
ln -s ~/data/DFCData dexgrasp_generation/data/DFCData
ln -s ~/data/mjcf dexgrasp_generation/data/mjcf
```

要求是最终路径能正常解析到下面这些位置：

- `dexgrasp_generation/data/DFCData`
- `dexgrasp_generation/data/mjcf`

数据下载说明请参考 upstream README：

```bash
sed -n '1,220p' dexgrasp_generation/README.md
```

### Policy

当前仓库已经包含 smoke test 所需的最小资产：

- `dexgrasp_policy/assets/datasetv4.1`
- `dexgrasp_policy/assets/meshdatav3_scaled`
- `dexgrasp_policy/assets/meshdatav3_pc_feat`
- `dexgrasp_policy/assets/mjcf`
- `dexgrasp_policy/assets/urdf`

## 常用脚本

### 环境脚本

- `scripts/setup_uv_generation.sh`
- `scripts/activate_uv_generation.sh`
- `scripts/setup_uv_policy.sh`
- `scripts/activate_uv_policy.sh`
- `scripts/setup_isaacgym.sh`
- `scripts/setup_pointnet2_policy.sh`

### Generation 脚本

- `scripts/run_generation_train.sh`
- `scripts/run_generation_eval.sh`

### Policy 脚本

- `scripts/run_policy_state_smoke.sh`
- `scripts/run_policy_state_train.sh`
- `scripts/run_policy_vision_smoke.sh`
- `scripts/run_policy_vision_train.sh`
- `scripts/run_policy_dagger_train.sh`

兼容性包装脚本仍保留在 `dexgrasp_policy/dexgrasp/script` 下。

## Generation 用法

### 训练 IPDF

```bash
bash scripts/run_generation_train.sh ipdf_config ipdf_train
```

### 训练 Glow

```bash
bash scripts/run_generation_train.sh glow_config glow_train
```

### 训练 Glow Joint

```bash
bash scripts/run_generation_train.sh glow_joint_config glow_joint_train
```

### 训练 ContactNet

```bash
bash scripts/run_generation_train.sh cm_net_config cm_net_train
```

### 评估

```bash
bash scripts/run_generation_eval.sh eval_config eval
```

## Policy 用法

### 状态 PPO smoke test

这是一个已经验证通过的最小启动检查。

```bash
bash scripts/run_policy_state_smoke.sh
```

### 视觉 PPO smoke test

这是一个已经验证通过的视觉链路最小启动检查。

```bash
bash scripts/run_policy_vision_smoke.sh
```

### 状态 PPO 训练

```bash
bash scripts/run_policy_state_train.sh
```

常见覆盖参数：

```bash
GPU_ID=0 NUM_ENVS=128 MAX_ITERATIONS=100 LOGDIR=logs/state_exp bash scripts/run_policy_state_train.sh
```

### 视觉 PPO 训练

```bash
bash scripts/run_policy_vision_train.sh
```

常见覆盖参数：

```bash
GPU_ID=0 NUM_ENVS=8 MAX_ITERATIONS=100 LOGDIR=logs/vision_exp BACKBONE_TYPE=pn bash scripts/run_policy_vision_train.sh
```

### Vision DAgger 训练

```bash
EXPERT_MODEL=dexgrasp_policy/example_model/model.pt bash scripts/run_policy_dagger_train.sh
```

常见覆盖参数：

```bash
GPU_ID=0 NUM_ENVS=8 MAX_ITERATIONS=100 LOGDIR=logs/dagger_exp BACKBONE_TYPE=pn EXPERT_MODEL=dexgrasp_policy/example_model/model.pt bash scripts/run_policy_dagger_train.sh
```

### 手动测试 / checkpoint 运行

状态 PPO：

```bash
source scripts/activate_uv_policy.sh
cd dexgrasp_policy/dexgrasp
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task=ShadowHandGrasp \
  --algo=ppo \
  --seed=0 \
  --rl_device=cuda:0 \
  --sim_device=cuda:0 \
  --logdir=logs/test_state \
  --headless \
  --test \
  --model_dir=/abs/path/to/checkpoint.pt
```

视觉 PPO：

```bash
source scripts/activate_uv_policy.sh
cd dexgrasp_policy/dexgrasp
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task=ShadowHandRandomLoadVision \
  --algo=ppo \
  --seed=0 \
  --rl_device=cuda:0 \
  --sim_device=cuda:0 \
  --logdir=logs/test_vision \
  --headless \
  --vision \
  --backbone_type=pn \
  --test \
  --model_dir=/abs/path/to/checkpoint.pt
```

## 推荐顺序

新机器上建议按下面顺序执行：

```bash
nvidia-smi
bash scripts/setup_uv_generation.sh
bash scripts/setup_uv_policy.sh
bash scripts/setup_isaacgym.sh
bash scripts/setup_pointnet2_policy.sh
bash scripts/run_policy_state_smoke.sh
bash scripts/run_policy_vision_smoke.sh
```

日常做 policy 开发时：

```bash
source scripts/activate_uv_policy.sh
bash scripts/run_policy_state_train.sh
```

日常做 generation 开发时：

```bash
source scripts/activate_uv_generation.sh
bash scripts/run_generation_train.sh ipdf_config ipdf_train
```

## 已知警告

当前在这台主机上观察到以下 warning，但它们没有阻止 smoke test：

- headless 模式下的 Isaac Gym graphics warning
- Gym 已不再维护的警告
- `JointSpec type free not yet supported!`

## 故障排查

### `nvidia-smi` 失败

先修 GPU / 驱动层，再处理 Python 环境。

### `isaacgym` 能导入但训练一启动就崩

重新执行：

```bash
bash scripts/setup_uv_policy.sh
bash scripts/setup_isaacgym.sh
```

### `pointnet2_ops` 导入失败

重新执行：

```bash
bash scripts/setup_pointnet2_policy.sh
```

### `torch_geometric` 在 generation 环境里导入失败

请通过下面方式激活环境：

```bash
source scripts/activate_uv_generation.sh
```

这个脚本会设置需要的 `LD_LIBRARY_PATH`。
