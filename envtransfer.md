• 下面按“迁移到新服务器可复现运行”的目标来做。核心建议：不要只 git clone，也不要优先复制虚拟环境；先复制代码、资产、数据、你本机的未提交
  改动，然后在新服务器重建环境。只有新旧服务器用户名、路径、OS、Python、CUDA 基本一致时，才考虑直接搬 .venv/.venvs。

  当前情况
  UltraDexGrasp：

  - 路径：/home/ym/fzb/UltraDexGrasp
  - 体积约 36G
  - .venv 约 6.6G
  - asset 约 6.0G
  - outputs 约 19G，如果只是重新跑可以不迁移
  - 当前环境：Python 3.10.12，Torch 2.4.1+cu121，SAPIEN 3.0.0b1，PyTorch3D 0.7.9，cuRobo 0.7.7...
  - 有本地改动：rollout.py 等，不要只拉 upstream

  UniDexGrasp：

  - 路径：/home/ym/fzb/UniDexGrasp
  - 体积约 14G
  - .venvs 约 12G
  - generation 环境：Python 3.8.20，Torch 1.11.0+cu113，PyTorch3D 0.7.2
  - policy 环境：Python 3.8.20，Torch 1.13.1+cu117，Isaac Gym Preview 4，PointNet2
  - dexgrasp_generation/data/DFCData 和 mjcf 是软链接，指向 /home/ym/data/fzb/UniDexGrasp_CVPR2023/...，迁移时必须单独处理
  - 已有迁移/运行脚本：scripts/setup_uv_generation.sh、scripts/setup_uv_policy.sh、scripts/setup_isaacgym.sh、scripts/
    setup_pointnet2_policy.sh

  新服务器要求
  建议新服务器满足：

  Ubuntu 20.04/22.04 x86_64
  NVIDIA GPU + 正常工作的 nvidia-smi
  NVIDIA driver 建议 535+，更高也可以
  至少 80G 可用磁盘；如果迁移完整数据和输出，预留更多

  系统依赖：

  sudo apt update
  sudo apt install -y build-essential git git-lfs curl rsync tmux htop unzip \
    cmake ninja-build pkg-config ffmpeg \
    libgl1 libglib2.0-0 libxrender1 libxext6 libxi6 libxrandr2 libxcursor1 \
    libegl1 libglvnd0 libvulkan1
  git lfs install
  curl -LsSf https://astral.sh/uv/install.sh | sh

  迁移代码和数据
  推荐用 rsync。如果要重建环境，排除虚拟环境和缓存：

  mkdir -p /home/NEWUSER/fzb

  rsync -aH --info=progress2 \
    --exclude '.venv' --exclude '__pycache__' --exclude '.git/logs' \
    /home/ym/fzb/UltraDexGrasp NEWUSER@NEW_HOST:/home/NEWUSER/fzb/

  rsync -aH --info=progress2 \
    --exclude '.venvs' --exclude '__pycache__' --exclude '.git/logs' \
    /home/ym/fzb/UniDexGrasp NEWUSER@NEW_HOST:/home/NEWUSER/fzb/

  UniDexGrasp 的 generation 数据要单独迁移或重新下载。目标机器上保持最终路径成立即可：

  mkdir -p /home/NEWUSER/fzb/UniDexGrasp/dexgrasp_generation/data
  ln -s /data/fzb/UniDexGrasp_CVPR2023/DFCData /home/NEWUSER/fzb/UniDexGrasp/dexgrasp_generation/data/DFCData
  ln -s /data/fzb/UniDexGrasp_CVPR2023/mjcf /home/NEWUSER/fzb/UniDexGrasp/dexgrasp_generation/data/mjcf

  如果 Ultra 的历史生成结果也要搬，保留 UltraDexGrasp/outputs；如果只是重新生成，可以不搬它。

  UltraDexGrasp 环境
  在新服务器：

  cd /home/NEWUSER/fzb/UltraDexGrasp
  uv venv .venv --python 3.10
  source .venv/bin/activate
  unset PYTHONPATH

  pip install -U pip setuptools wheel ninja
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt

  pip install -e third_party/pytorch3d --no-build-isolation
  pip install -e third_party/curobo --no-build-isolation
  pip install -e third_party/BODex_api --no-build-isolation
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
  pip install coal cmeel-eigen setuptools_scm

  cd third_party/BODex_api/src/bodex/geom/cpp
  python setup.py install

  验证：

  cd /home/NEWUSER/fzb/UltraDexGrasp
  source env/activate_uv.sh
  python -c "import torch, sapien, pytorch3d, curobo, coal; print(torch.__version__, torch.cuda.is_available())"
  CUDA_VISIBLE_DEVICES=0 python rollout.py --hand 0 --object_scale_list '[0.08]' \
    --object-root asset/object_mesh --object-names drill --output-root outputs/migrate_test

  UniDexGrasp 环境
  UniDexGrasp 不要合并环境。它分 generation 和 policy 两套：

  cd /home/NEWUSER/fzb/UniDexGrasp

  bash scripts/setup_uv_generation.sh
  bash scripts/setup_uv_policy.sh
  bash scripts/setup_isaacgym.sh
  bash scripts/setup_pointnet2_policy.sh

  如果 PointNet2 编译报 CUDA 版本不匹配，优先在服务器装 CUDA 11.7 toolkit，然后：

  CUDA_HOME=/usr/local/cuda-11.7 TORCH_CUDA_ARCH_LIST="8.6+PTX" bash scripts/setup_pointnet2_policy.sh

  验证 generation：

  source scripts/activate_uv_generation.sh
  python -c "import torch, pytorch3d, torch_geometric, csdf; print(torch.__version__, torch.cuda.is_available(), pytorch3d.__version__)"
  test -d dexgrasp_generation/data/DFCData
  test -d dexgrasp_generation/data/mjcf

  验证 policy，注意 Isaac Gym 必须先 import：

  source scripts/activate_uv_policy.sh
  python -c "from isaacgym import gymapi; import torch, pointnet2_ops; print(torch.__version__, torch.cuda.is_available(),
  gymapi.acquire_gym())"

  bash scripts/run_policy_state_smoke.sh
  bash scripts/run_policy_vision_smoke.sh

  常见坑

  1. 只 git clone 会漏掉本地改动、scripts/、IsaacGym 包、训练 checkpoint、软链接数据。
  2. 复制 .venv/.venvs 很容易因为绝对路径、Python 软链接、CUDA 扩展失效。能重建就重建。
  3. UniDexGrasp 的 dexgrasp_generation/data 当前是软链接，必须在新服务器重新指到真实数据目录。
  4. Isaac Gym import 顺序敏感：先 from isaacgym import gymapi，再 import torch。
  5. 如果服务器上有 ROS/MoveIt 环境，运行前 unset PYTHONPATH，否则会污染这两个项目的 Python 包解析。
  6. 迁移完成的最低标准是：Ultra import 通过、Ultra rollout 能启动；Uni generation import 通过；Uni policy 两个 smoke test 通过。

  H20 / Hopper (sm_90) 专项说明

  如果目标服务器是 H20（或其他 Hopper 架构 GPU），现有脚本无需修改，只需注意以下几点：

  1. PointNet2 编译时将架构改为 9.0：

     TORCH_CUDA_ARCH_LIST="9.0+PTX" \
     CUDA_HOME=/usr/local/cuda \
     POINTNET2_SKIP_CUDA_VERSION_CHECK=1 \
       bash scripts/setup_pointnet2_policy.sh

     原来 Ada 上用的是 8.6+PTX，H20 需要 9.0+PTX。

  2. PyTorch 不需要升级。torch 1.11.0+cu113 和 1.13.1+cu117 的预编译 wheel 包含 PTX 中间码，
     NVIDIA 驱动（535+）会在首次 kernel 调用时 JIT 编译为 sm_90 原生代码。

  3. 首次运行每个环境时会有 10-30 秒额外延迟（PTX JIT），之后驱动缓存在 ~/.nv/ComputeCache/。
     可设置 export CUDA_CACHE_MAXSIZE=1073741824 加大缓存到 1GB。

  4. Isaac Gym Preview 4 的 .so 使用 CUDA Driver API，驱动前向兼容可以处理。
     可能打印 "unsupported GPU architecture" 警告，不影响功能。

  5. 一键脚本：bash scripts/setup_all_h20.sh 会按顺序执行全部 4 个 setup 脚本，
     自动为 PointNet2 设置 TORCH_CUDA_ARCH_LIST=9.0+PTX。