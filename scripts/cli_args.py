"""Small subset of the official Isaac Lab RSL-RL CLI helper."""

from __future__ import annotations

import argparse
import random


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl", description="RSL-RL options")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)
    group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    group.add_argument("--log_project_name", type=str, default=None)


def update_rsl_rl_cfg(agent_cfg, args_cli):
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg
