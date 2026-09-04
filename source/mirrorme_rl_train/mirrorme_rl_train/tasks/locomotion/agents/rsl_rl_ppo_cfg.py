"""Official RSL-RL PPO configuration with an asymmetric actor-critic."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BPXFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 2026
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "bpx_flat"
    clip_actions = 100.0

    # Actor: policy group only = 45-D. Critic: policy + privileged = 48-D.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
