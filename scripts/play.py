#!/usr/bin/env python3
"""Play a BPX checkpoint and automatically export TorchScript and ONNX."""

import argparse
import os
import sys
import time

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real_time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
default_hydra_args = ["hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"]
sys.argv = [sys.argv[0]] + default_hydra_args + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import mirrorme_rl_train.tasks  # noqa: F401


def export_policy(runner: OnPolicyRunner, export_dir: str) -> None:
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="bpx_flat_45.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="bpx_flat_45.onnx")


@hydra_task_config("bpx_flat", args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or 16
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(checkpoint)
    env_cfg.log_dir = log_dir

    env = gym.make("bpx_flat", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    export_dir = os.path.join(log_dir, "exported")
    export_policy(runner, export_dir)
    print(f"[INFO] Exported to: {export_dir}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    steps = 0
    while simulation_app.is_running():
        begin = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)
        steps += 1
        if args_cli.video and steps >= args_cli.video_length:
            break
        delay = dt - (time.time() - begin)
        if args_cli.real_time and delay > 0:
            time.sleep(delay)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
