#!/usr/bin/env python3
"""Start one Isaac Lab environment and assert the public 45-D/12-D contract."""

import argparse
import sys

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import mirrorme_rl_train.tasks  # noqa: F401
from mirrorme_rl_train.tasks.locomotion.bpx_flat_env_cfg import BPXFlatEnvCfg

ACTION_DIM = 12
OBSERVATION_DIM = 45
JOINT_NAMES = (
    "fl_hip_roll_joint",
    "fr_hip_roll_joint",
    "hl_hip_roll_joint",
    "hr_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fr_hip_pitch_joint",
    "hl_hip_pitch_joint",
    "hr_hip_pitch_joint",
    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint",
)


def main() -> None:
    cfg = BPXFlatEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("bpx_flat", cfg=cfg)
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    privileged_obs = obs["privileged"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    actual_joint_names = tuple(env.unwrapped.scene["robot"].joint_names)
    missing = [name for name in JOINT_NAMES if name not in actual_joint_names]

    print(f"policy observation shape: {tuple(policy_obs.shape)}")
    print(f"privileged observation shape: {tuple(privileged_obs.shape)}")
    print(f"critic observation dimension: {policy_obs.shape[-1] + privileged_obs.shape[-1]}")
    print(f"action dimension: {action_dim}")
    print("policy joint order:")
    for index, name in enumerate(JOINT_NAMES):
        print(
            f"  {index:02d}: {name} "
            f"(robot index={actual_joint_names.index(name) if name in actual_joint_names else 'MISSING'})"
        )

    if policy_obs.shape[-1] != OBSERVATION_DIM:
        raise RuntimeError(f"expected {OBSERVATION_DIM} observations, got {policy_obs.shape[-1]}")
    if privileged_obs.shape[-1] != 3:
        raise RuntimeError(f"expected 3 privileged observations, got {privileged_obs.shape[-1]}")
    if action_dim != ACTION_DIM:
        raise RuntimeError(f"expected {ACTION_DIM} actions, got {action_dim}")
    if missing:
        raise RuntimeError(f"robot is missing policy joints: {missing}")
    print("[PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
