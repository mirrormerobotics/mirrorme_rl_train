# MIRRORME RL Train

[English](README.md) | [简体中文](README.zh-CN.md)

An open reinforcement-learning training baseline for the MIRRORME BPX quadruped. It follows the Isaac Lab external-project layout, uses a Manager-Based Environment, and trains PPO policies through the official RSL-RL integration.

This repository covers simulation training, evaluation, and policy export only. Robot SDK integration and physical-robot deployment are intentionally out of scope.

## Task and naming

The project currently registers one task:

| Name | Value | Meaning |
|---|---|---|
| Task ID | `bpx_flat` | Flat-ground omnidirectional velocity tracking |
| Experiment name | `bpx_flat` | RSL-RL log directory under `logs/` |
| Run directory | `YYYY-MM-DD_HH-MM-SS[_run_name]` | One training run |
| Checkpoint | `model_<iteration>.pt` | Saved RSL-RL checkpoint, for example `model_2999.pt` |
| Exported policy | `bpx_flat_45.pt` / `bpx_flat_45.onnx` | TorchScript and ONNX actor exports |

The command target is body-frame velocity `[vx, vy, wz]` on a flat plane. There are no wheel joints, height scanners, history encoders, DWAQ/CENet modules, recurrent networks, or custom RL runners.

## 1. Requirements

The published setup is pinned to the following stack:

| Component | Version |
|---|---:|
| Operating system | Ubuntu 22.04 LTS, x86-64 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch / torchvision | 2.7.0 / 0.22.0, CUDA 12.8 wheels |
| RSL-RL | 3.0.1, installed by the Isaac Lab `rsl-rl` extra |

Isaac Sim training requires a compatible NVIDIA GPU and driver. NVIDIA recommends at least 32 GB of RAM and 16 GB of VRAM for the Isaac Lab 2.3.0 stack. Pip installation also requires glibc 2.35 or newer; Ubuntu 22.04 satisfies this requirement.

## 2. Clone and install

Clone the repository together with the BPX asset submodule:

```bash
git clone --recurse-submodules <repository-url> mirrorme_rl_train
cd mirrorme_rl_train
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### Automatic clean-environment setup

From the repository root, run:

```bash
./bpx.sh setup
```

This creates a new Conda environment named `bpx_train`, installs the pinned Isaac Sim, Isaac Lab, PyTorch, RSL-RL, and project packages, and then launches the headless runtime contract check. It refuses to modify an existing environment.

Use a different new environment name when needed:

```bash
./bpx.sh setup my_bpx_environment
```

Use `--skip-contract` only when installing dependencies without launching Isaac Sim:

```bash
./bpx.sh setup my_bpx_environment --skip-contract
```

After setup finishes:

```bash
conda activate bpx_train
```

The first Isaac Sim launch may download extension caches and ask you to accept the NVIDIA Omniverse license agreement. Read and answer that prompt directly; the setup script does not accept the agreement on your behalf.

### Manual installation

Use these commands only when troubleshooting or when you need control over each installation step:

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

Isaac Lab 2.3.0 pins the legacy `flatdict==4.0.1` source package. Preinstalling it with a compatible build toolchain avoids failures caused by newer setuptools releases.

Verify the installed versions and GPU visibility:

```bash
python -c "from importlib.metadata import version; import torch; print('Isaac Lab:', version('isaaclab')); print('Isaac Sim:', version('isaacsim')); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True` before GPU training.

## 3. Verify the project

Check task registration without starting the simulator:

```bash
python scripts/list_envs.py
```

Expected output:

```text
bpx_flat
```

Then launch one headless environment and validate the training interface:

```bash
./bpx.sh contract --headless
```

A successful check verifies 45 Actor observations, 48 Critic observations, 12 actions, and all 12 policy joint names, then prints:

```text
[PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.
```

## 4. Repository layout

```text
mirrorme_rl_train/
├── bpx.sh                         # command entry point
├── resources/
│   ├── mirrorme.py                # BPX articulation configuration
│   └── BPX/                       # robot asset Git submodule
├── scripts/
│   ├── setup_environment.sh       # clean Conda setup
│   ├── inspect_contract.py        # runtime dimension/name check
│   ├── train.py                   # PPO training
│   ├── play.py                    # checkpoint evaluation
│   └── export_policy.py           # TorchScript/ONNX export
└── source/mirrorme_rl_train/
    └── mirrorme_rl_train/tasks/locomotion/
        ├── bpx_flat_env_cfg.py
        ├── agents/rsl_rl_ppo_cfg.py
        └── mdp/rewards.py
```

Training loads `resources/BPX/bpx/urdf/bpx.urdf`. Keep the submodule directory structure intact so its relative mesh paths remain valid. `MIRRORME_BPX_ASSET_ROOT` can override the asset root for local experiments.

## 5. Training interface

### Actor observations: 45 dimensions

```text
0:3    base angular velocity × 0.25
3:6    projected gravity
6:9    [vx, vy, wz] command × [2.0, 2.0, 0.25]
9:21   joint position - default joint position
21:33  joint velocity × 0.05
33:45  previous raw action
```

Observation noise is enabled during training for angular velocity, gravity, joint position, and joint velocity.

### Critic observations: 48 dimensions

The Critic receives the 45-dimensional policy group plus the simulator-only 3D base linear velocity. Actor normalization is disabled; Critic observation normalization is enabled.

### Actions: 12 dimensions

```text
q_target = q_default + 0.25 × clipped_action
```

Policy joint order:

```text
FL/FR/HL/HR hip roll,
FL/FR/HL/HR hip pitch,
FL/FR/HL/HR knee
```

The default joint pose is hip roll `0.0`, hip pitch `0.8`, and knee `-1.5` radians for each leg. The simulated actuator uses effort limit `30 Nm`, stiffness `30`, damping `1`, and armature `0.003`.

## 6. Environment and PPO configuration

| Setting | Value |
|---|---:|
| Simulation timestep | `0.005 s` |
| Control decimation | `4` |
| Policy frequency | `50 Hz` |
| Episode length | `20 s` |
| Default environments | `4096` |
| Command resampling | `5–9 s` |
| Standing environments | `20%` |
| Forward velocity | `[-1.5, 1.5] m/s` |
| Lateral velocity | `[-1.0, 1.0] m/s` |
| Yaw rate | `[-2.0, 2.0] rad/s` |

The Actor is `45 → 256 → 256 → 128 → 12`; the Critic is `48 → 512 → 256 → 128 → 1`. Both use ELU activations.

Default PPO settings:

| Setting | Value |
|---|---:|
| Rollout length | `24` steps per environment |
| Learning epochs | `5` |
| Mini-batches | `4` |
| Clip parameter | `0.2` |
| Discount / GAE lambda | `0.99 / 0.95` |
| Learning rate | `1e-3`, adaptive KL schedule |
| Entropy coefficient | `0.01` |
| Save interval | `200` iterations |
| Default maximum iterations | `10000` |

Velocity tracking uses weights `+2.0` for XY velocity and `+1.0` for yaw rate. Additional terms cover body stability, base height, foot air time and clearance, sliding, contact force, non-foot contacts, torque, mechanical power, acceleration, action rate, joint limits, leg posture, standing posture, and termination.

Domain randomization covers friction, restitution, base mass and center of mass, initial pose and velocity, joint state, observation noise, and horizontal pushes.

## 7. Train

Run a small smoke test first:

```bash
./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless
```

Then start a larger run:

```bash
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```

Training outputs are written to:

```text
logs/bpx_flat/YYYY-MM-DD_HH-MM-SS[_run_name]/
├── model_*.pt
├── params/env.yaml
├── params/agent.yaml
└── events.out.tfevents.*
```

Monitor a run with TensorBoard:

```bash
tensorboard --logdir logs/bpx_flat
```

Resume from a named run:

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

An absolute checkpoint path is also accepted with `--resume --checkpoint /absolute/path/model_2000.pt`.

## 8. Evaluate and export

Open an interactive evaluation window:

```bash
./bpx.sh play --checkpoint /absolute/path/model_2999.pt
```

Optional evaluation flags include `--num_envs`, `--video`, `--video_length`, `--real_time`, and normal Isaac Lab launcher flags such as `--headless` and `--device`.

Export without entering the interactive play loop:

```bash
./bpx.sh export --checkpoint /absolute/path/model_2999.pt
```

By default, the exporter creates `exported/bpx_flat_45.pt` and `exported/bpx_flat_45.onnx` beside the checkpoint. Use `--output_dir PATH` to select another directory.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| BPX meshes or URDF are missing | Run `git submodule update --init --recursive` |
| `bpx_flat` is not listed | Activate the intended environment and run `./bpx.sh install` |
| `ModuleNotFoundError` for Isaac Lab or Isaac Sim | Confirm the `bpx_train` environment is active and repeat the version check |
| CUDA is unavailable | Check `nvidia-smi`, the NVIDIA driver, GPU visibility, and the CUDA 12.8 PyTorch wheel |
| `flatdict` fails to build | Use the documented `setuptools<81` and `--no-build-isolation` commands |
| Isaac Sim pauses on first launch | Complete the license prompt and allow extension caches to download |
| A long run fails immediately | Run `./bpx.sh contract --headless` and the 64-environment smoke test first |

## References and license

- [Isaac Lab 2.3.0 installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
- [Isaac Lab pip-package installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)
- [BPX model assets](https://github.com/mirrormerobotics/BPX)

The project is released under the [BSD 3-Clause License](LICENSE). NVIDIA Isaac Sim and its pip packages are governed by NVIDIA's own license terms.
