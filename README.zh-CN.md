# MIRRORME 强化学习训练

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个面向 MIRRORME BPX 四足机器人的开源强化学习训练基线。工程采用 Isaac Lab 官方外部项目结构，使用 Manager-Based Environment，并通过官方 RSL-RL 集成训练 PPO 策略。

本仓库只包含仿真训练、策略评估和模型导出，不包含机器人 SDK 接入或实机部署内容。

## 任务与命名

工程目前只注册一个任务：

| 名称 | 值 | 含义 |
|---|---|---|
| 任务 ID | `bpx_flat` | 平地全向速度跟踪任务 |
| 实验名称 | `bpx_flat` | `logs/` 下的 RSL-RL 日志目录名 |
| 运行目录 | `YYYY-MM-DD_HH-MM-SS[_run_name]` | 一次独立训练 |
| 检查点 | `model_<iteration>.pt` | RSL-RL 保存的模型，例如 `model_2999.pt` |
| 导出模型 | `bpx_flat_45.pt` / `bpx_flat_45.onnx` | TorchScript 和 ONNX Actor |

策略在平面上跟踪机身坐标系速度指令 `[vx, vy, wz]`。任务中没有轮关节、高度扫描器、历史编码器、DWAQ/CENet 模块、循环网络或自定义 RL Runner。

## 1. 环境要求

本教程固定使用以下软件栈：

| 组件 | 版本 |
|---|---:|
| 操作系统 | Ubuntu 22.04 LTS，x86-64 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch / torchvision | 2.7.0 / 0.22.0，CUDA 12.8 wheel |
| RSL-RL | 3.0.1，由 Isaac Lab 的 `rsl-rl` extra 安装 |

Isaac Sim 训练需要兼容的 NVIDIA GPU 和驱动。NVIDIA 对 Isaac Lab 2.3.0 软件栈的建议是至少 32 GB 内存和 16 GB 显存。pip 安装还要求 glibc 2.35 或更高版本，Ubuntu 22.04 满足该要求。

## 2. 克隆与安装

克隆仓库时同时拉取 BPX 模型子模块：

```bash
git clone --recurse-submodules <repository-url> mirrorme_rl_train
cd mirrorme_rl_train
```

如果已经克隆但没有拉取子模块：

```bash
git submodule update --init --recursive
```

### 自动创建干净环境

在仓库根目录执行：

```bash
./bpx.sh setup
```

该命令会创建名为 `bpx_train` 的新 Conda 环境，安装固定版本的 Isaac Sim、Isaac Lab、PyTorch、RSL-RL 和本项目，最后运行无界面的接口检查。为避免破坏已有环境，脚本不会复用或修改同名 Conda 环境。

如需指定另一个新环境名称：

```bash
./bpx.sh setup my_bpx_environment
```

如果只安装依赖、暂时不启动 Isaac Sim：

```bash
./bpx.sh setup my_bpx_environment --skip-contract
```

安装完成后激活环境：

```bash
conda activate bpx_train
```

Isaac Sim 第一次启动时可能下载扩展缓存，并要求接受 NVIDIA Omniverse 许可协议。请自行阅读并回答终端提示；安装脚本不会代替用户接受协议。

### 手动安装

以下步骤用于故障排查或需要逐步控制安装过程的情况：

```bash
conda create -n bpx_train python=3.11 -y
conda activate bpx_train
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install "isaaclab[isaacsim,rsl-rl]==2.3.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
./bpx.sh install
```

Isaac Lab 2.3.0 固定依赖旧版源码包 `flatdict==4.0.1`。先使用兼容的构建工具和 `--no-build-isolation` 安装它，可以避免新版 setuptools 引起的构建失败。

检查安装版本和 GPU 可见性：

```bash
python -c "from importlib.metadata import version; import torch; print('Isaac Lab:', version('isaaclab')); print('Isaac Sim:', version('isaacsim')); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

开始 GPU 训练前，`CUDA available` 必须为 `True`。

## 3. 验证工程

不启动仿真器，先检查任务注册：

```bash
python scripts/list_envs.py
```

预期输出：

```text
bpx_flat
```

随后启动一个无界面环境并验证训练接口：

```bash
./bpx.sh contract --headless
```

检查会验证 45 维 Actor 观测、48 维 Critic 观测、12 维动作以及 12 个策略关节名称，成功时输出：

```text
[PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.
```

## 4. 工程结构

```text
mirrorme_rl_train/
├── bpx.sh                         # 统一命令入口
├── resources/
│   ├── mirrorme.py                # BPX Articulation 配置
│   └── BPX/                       # 机器人模型 Git 子模块
├── scripts/
│   ├── setup_environment.sh       # 创建干净 Conda 环境
│   ├── inspect_contract.py        # 运行时维度和关节名称检查
│   ├── train.py                   # PPO 训练
│   ├── play.py                    # 检查点评估
│   └── export_policy.py           # 导出 TorchScript/ONNX
└── source/mirrorme_rl_train/
    └── mirrorme_rl_train/tasks/locomotion/
        ├── bpx_flat_env_cfg.py
        ├── agents/rsl_rl_ppo_cfg.py
        └── mdp/rewards.py
```

训练加载 `resources/BPX/bpx/urdf/bpx.urdf`。不要改变子模块内部目录结构，否则 URDF 的相对网格路径会失效。本地实验可以通过 `MIRRORME_BPX_ASSET_ROOT` 覆盖资源根目录。

## 5. 训练接口

### Actor 观测：45 维

```text
0:3    机身角速度 × 0.25
3:6    投影重力
6:9    [vx, vy, wz] 指令 × [2.0, 2.0, 0.25]
9:21   关节位置 - 默认关节位置
21:33  关节速度 × 0.05
33:45  上一步原始动作
```

训练时会对角速度、重力、关节位置和关节速度加入观测噪声。

### Critic 观测：48 维

Critic 接收 45 维 policy 观测，再加上仅仿真可用的 3 维机身线速度。Actor 不使用观测归一化，Critic 开启观测归一化。

### 动作：12 维

```text
q_target = q_default + 0.25 × clipped_action
```

策略关节顺序：

```text
FL/FR/HL/HR 髋关节 roll，
FL/FR/HL/HR 髋关节 pitch，
FL/FR/HL/HR 膝关节
```

每条腿的默认关节位置分别为 hip roll `0.0`、hip pitch `0.8`、knee `-1.5` 弧度。仿真执行器参数为力矩上限 `30 Nm`、刚度 `30`、阻尼 `1`、armature `0.003`。

## 6. 环境与 PPO 配置

| 设置 | 数值 |
|---|---:|
| 仿真时间步 | `0.005 s` |
| 控制降采样 | `4` |
| 策略频率 | `50 Hz` |
| Episode 长度 | `20 s` |
| 默认环境数量 | `4096` |
| 指令重采样间隔 | `5–9 s` |
| 静止环境比例 | `20%` |
| 前后速度 | `[-1.5, 1.5] m/s` |
| 横向速度 | `[-1.0, 1.0] m/s` |
| 偏航角速度 | `[-2.0, 2.0] rad/s` |

Actor 网络为 `45 → 256 → 256 → 128 → 12`，Critic 网络为 `48 → 512 → 256 → 128 → 1`，两者均使用 ELU 激活函数。

默认 PPO 参数：

| 设置 | 数值 |
|---|---:|
| 每个环境的 rollout 长度 | `24` 步 |
| 每次更新的训练轮数 | `5` |
| Mini-batch 数量 | `4` |
| PPO clip 参数 | `0.2` |
| 折扣因子 / GAE lambda | `0.99 / 0.95` |
| 学习率 | `1e-3`，自适应 KL 调度 |
| 熵系数 | `0.01` |
| 保存间隔 | `200` 次迭代 |
| 默认最大迭代次数 | `10000` |

速度跟踪奖励中，XY 线速度权重为 `+2.0`，偏航角速度权重为 `+1.0`。其他奖励项包括机身稳定性、机身高度、足端腾空和抬脚高度、滑动、接触力、非足端碰撞、关节力矩、机械功率、加速度、动作变化率、关节限位、腿部姿态、静止姿态和终止惩罚。

域随机化包括摩擦和恢复系数、机身质量和质心、初始位姿和速度、关节状态、观测噪声以及水平推力扰动。

## 7. 训练

先运行小规模冒烟测试：

```bash
./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless
```

然后开始较大规模训练：

```bash
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```

训练结果保存在：

```text
logs/bpx_flat/YYYY-MM-DD_HH-MM-SS[_run_name]/
├── model_*.pt
├── params/env.yaml
├── params/agent.yaml
└── events.out.tfevents.*
```

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir logs/bpx_flat
```

从指定运行继续训练：

```bash
./bpx.sh train \
  --task bpx_flat \
  --resume \
  --load_run 2026-07-23_10-00-00_baseline \
  --checkpoint model_2000.pt \
  --num_envs 1024 \
  --max_iterations 1000 \
  --headless
```

也可以通过 `--resume --checkpoint /absolute/path/model_2000.pt` 直接指定绝对路径。

## 8. 评估与导出

打开交互式评估窗口：

```bash
./bpx.sh play --checkpoint /absolute/path/model_2999.pt
```

评估脚本还支持 `--num_envs`、`--video`、`--video_length`、`--real_time`，以及 `--headless`、`--device` 等 Isaac Lab 启动参数。

不进入交互式播放循环，直接导出模型：

```bash
./bpx.sh export --checkpoint /absolute/path/model_2999.pt
```

默认会在检查点旁边创建 `exported/bpx_flat_45.pt` 和 `exported/bpx_flat_45.onnx`。可以通过 `--output_dir PATH` 指定其他输出目录。

## 9. 常见问题

| 现象 | 检查方法 |
|---|---|
| 找不到 BPX 网格或 URDF | 执行 `git submodule update --init --recursive` |
| 环境列表中没有 `bpx_flat` | 激活正确环境后执行 `./bpx.sh install` |
| 缺少 Isaac Lab 或 Isaac Sim 模块 | 确认已经激活 `bpx_train`，然后重新运行版本检查 |
| CUDA 不可用 | 检查 `nvidia-smi`、NVIDIA 驱动、GPU 可见性和 CUDA 12.8 PyTorch wheel |
| `flatdict` 构建失败 | 使用文档中的 `setuptools<81` 和 `--no-build-isolation` 命令 |
| Isaac Sim 第一次启动时暂停 | 完成许可协议提示，并等待扩展缓存下载完成 |
| 长时间训练一启动就失败 | 先运行 `./bpx.sh contract --headless` 和 64 环境冒烟测试 |

## 参考资料与许可证

- [Isaac Lab 2.3.0 安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
- [Isaac Lab pip 包安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)
- [BPX 模型资源](https://github.com/mirrormerobotics/BPX)

本项目采用 [BSD 3-Clause License](LICENSE)。NVIDIA Isaac Sim 及其 pip 包适用 NVIDIA 自身的许可条款。
