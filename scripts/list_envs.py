#!/usr/bin/env python3
"""List MIRRORME RL Train Gym environments without starting Isaac Sim."""

from project_paths import add_project_source

add_project_source()

import gymnasium as gym
import mirrorme_rl_train.tasks  # noqa: F401

for spec in sorted(gym.registry.values(), key=lambda item: item.id):
    if spec.id.startswith("bpx_"):
        print(spec.id)
