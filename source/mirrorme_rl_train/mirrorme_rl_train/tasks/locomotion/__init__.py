"""Gym registration for the sole public BPX flat-ground task."""

import gymnasium as gym

from . import agents


gym.register(
    id="bpx_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bpx_flat_env_cfg:BPXFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BPXFlatPPORunnerCfg",
    },
)
