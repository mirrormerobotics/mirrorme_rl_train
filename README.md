# MIRRORME RL Train

An open reinforcement-learning training baseline for the MIRRORME BPX robot. The project uses the official Isaac Lab external-project layout, a Manager-Based Environment, and the RSL-RL PPO interface. It trains one task only:

> BPX tracks body-frame omnidirectional velocity commands `[vx, vy, wz]` on a flat plane.

## Task and Naming

This project exposes exactly one Isaac Lab task:

| Name | Value | Meaning |
|---|---|---|
| Task ID | `bpx_flat` | The only Gym/Isaac Lab task. Training, playback, export, and contract checks all use it. |
| Experiment name | `mirrorme_rl_train_flat` | The RSL-RL experiment and log-directory name; it is not a second task. |
| Run directory | `YYYY-MM-DD_HH-MM-SS[_run_name]` | One individual training run, created under `logs/bpx_flat/`. |
| Checkpoint | `model_<iteration>.pt` | A saved policy from one run, for example `model_29999.pt`. |
| Exported policy | `bpx_flat_45.pt` / `bpx_flat_45.onnx` | TorchScript/ONNX policies with a 45-dimensional input and 12-dimensional output. |

`bpx_flat` is the currently registered task. Training accepts an explicit `--task` argument so that future registered tasks can be selected without changing the command structure.

## 1. Tested Versions

The project is currently tested with:

| Component | Version |
|---|---:|
| Ubuntu | 22.04 LTS |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.0 |
| PyTorch | 2.7.x |
| RL backend | Isaac Lab bundled RSL-RL integration |

Install Isaac Sim and Isaac Lab according to the official Isaac Lab documentation first. This repository does not duplicate step-by-step GPU driver, CUDA, or Conda installation guides.

## 2. Installation

Clone the project with the BPX asset submodule:

```bash
git clone --recurse-submodules <this-repository-url> mirrorme_rl_train
cd mirrorme_rl_train
```

If the repository was cloned without submodules, initialize them afterwards:

```bash
git submodule update --init --recursive
```

Activate the Python environment where Isaac Sim and Isaac Lab are installed:

```bash
conda activate bpx_open
```

Install this project in editable mode:

```bash
./bpx.sh install
```

Verify that the task is registered:

```bash
python scripts/list_envs.py
```

## 3. Robot Assets

Robot assets are tracked through the `resources/BPX` Git submodule. Training loads the URDF entry at
`resources/BPX/bpx/urdf/bpx.urdf`; generated USD assets are also kept in the upstream asset repository.

```text
resources/BPX/
├── bpx/
│   ├── urdf/bpx.urdf
│   └── meshes/*
├── mujoco/
└── usd/
    ├── bpx.usd
    └── configuration/*
```

## 4. Training Interface

### Actor: Fixed 45 Dimensions

```text
0:3    base angular velocity × 0.25
3:6    projected gravity
6:9    [vx, vy, wz] command × [2.0, 2.0, 0.25]
9:21   joint position - default joint position
21:33  joint velocity × 0.05
33:45  previous raw action
```

### Critic: 48 Dimensions, Training Only

```text
critic = actor_obs[45] + simulator base linear velocity[3]
```

This is asymmetric actor-critic training: the Actor uses only the 45 observations available on hardware, while the Critic uses simulator privileged information for value learning. The exported policy remains strictly:

```text
input  [batch, 45]
output [batch, 12]
```

### Actions

```text
q_target = q_default + 0.25 × action
```

Policy joint order:

```text
FL/FR/HL/HR hip roll,
FL/FR/HL/HR hip pitch,
FL/FR/HL/HR knee
```

## 5. Training Design

This is a flat-ground quadruped locomotion task, not a wheel-legged task. There are no wheel joints, wheel-speed references, terrain-height scanners, DWAQ/CENet encoders, reconstruction losses, or recovery/self-righting objectives.

The policy is trained with Isaac Lab's official RSL-RL PPO integration:

```text
Actor:  45 -> 256 -> 256 -> 128 -> 12
Critic: 48 -> 512 -> 256 -> 128 -> 1
```

The Critic receives the Actor observation plus simulator-only base linear velocity. This asymmetric setup improves value estimation during simulation while keeping the exported Actor strictly deployable as a `45 -> 12` policy.

Default PPO settings:

| Setting | Value |
|---|---:|
| Rollout length | 24 |
| Learning epochs | 5 |
| Mini-batches | 4 |
| Clip parameter | 0.2 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | `1e-3` with adaptive KL |

Commands are sampled as body-frame omnidirectional velocity targets. Thirty percent of environments receive exact zero commands to reinforce stable standing, and yaw commands count as movement for swing-foot and stand-still gating.

## 6. Reward Overview

Velocity tracking dominates the objective: XY velocity tracking in the yaw-aligned frame has weight `+4.0`, and yaw-rate tracking has weight `+2.0`.

The remaining rewards shape a hardware-friendly quadruped gait:

| Group | Terms |
|---|---|
| Base stability | vertical velocity, roll/pitch angular velocity, flat orientation, nominal base height |
| Feet and contacts | air time, sliding, support symmetry, swing clearance, scuffing, stumbling, excessive contact force, non-foot leg contacts |
| Joint regularization | torque, mechanical power, acceleration, action rate, joint limits, hip-roll posture, leg posture |
| Standing and failure | default-pose standing under low command, termination penalty |

Domain randomization covers ground friction, base mass, center of mass, reset pose/velocity, reset joint state, observation noise, and conservative horizontal pushes.

Recommended training flow:

1. Run 64 environments for 20 iterations to validate names and dimensions.
2. Verify reward terms in TensorBoard and check they remain finite.
3. Train the default run with 1024 environments.
4. Evaluate zero command, forward/backward, lateral motion, and pure yaw separately.
5. Export TorchScript/ONNX and confirm the `45 -> 12` interface.
6. Run low-limit real-machine safety validation before deployment.

## 7. Quick Start

All commands below run the sole task, `bpx_flat`.

Isaac Lab runtime contract check:

```bash
./bpx.sh contract --headless
```

Small smoke-test training run:

```bash
./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless
```

Full training run:

```bash
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```

Resume training:

```bash
./bpx.sh train \
  --task bpx_flat \
  --resume \
  --load_run 2026-07-23_10-00-00_baseline \
  --checkpoint model_2000.pt \
  --num_envs 1024 \
  --max_iterations 3000 \
  --headless
```

Play a checkpoint:

```bash
./bpx.sh play --checkpoint /absolute/path/model_3000.pt
```

Export a checkpoint:

```bash
./bpx.sh export --checkpoint /absolute/path/model_3000.pt
```
