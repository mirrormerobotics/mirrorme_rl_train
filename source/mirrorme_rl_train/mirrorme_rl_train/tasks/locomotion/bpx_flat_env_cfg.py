"""MIRRORME RL Train Isaac Lab manager-based flat-ground locomotion task for BPX.

Deployment contract
-------------------
* Actor observation group: exactly 45 dimensions.
* Action: 12 joint-position residuals.
* Critic: actor observations plus privileged base linear velocity (48 dimensions).

The task intentionally contains no DWAQ, history encoder, estimator, terrain scanner,
or custom RL runner. Training is handled by Isaac Lab's official RSL-RL integration.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from . import mdp as bpx_mdp
from resources.mirrorme import BPX_CFG

BASE_BODY_NAME = "torso"
FOOT_BODY_NAMES = ("fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link")
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


@configclass
class BPXSceneCfg(InteractiveSceneCfg):
    """Flat plane, BPX articulation, contact sensing and lighting."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = BPX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/"
                "PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class CommandsCfg:
    """Body-frame omnidirectional velocity commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 9.0),
        # Give the policy substantial experience of holding a stable stance.
        rel_standing_envs=0.20,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Full-speed omnidirectional locomotion.
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Twelve residual joint-position commands in deployment order."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(JOINT_NAMES),
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Deployable actor observations and training-only privileged observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Contractual order: 3 + 3 + 3 + 12 + 12 + 12 = 45.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
            clip=(-100.0, 100.0),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            scale=0.05,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # Missing from the actor by design. Available only in simulation to train V(s).
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Conservative domain randomization for flat-ground sim-to-real robustness."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Broader, especially dynamic, friction variation makes the gait
            # robust to noticeably different ground grip across environments.
            "static_friction_range": (0.30, 1.50),
            "dynamic_friction_range": (0.20, 1.25),
            "restitution_range": (0.0, 0.12),
            "num_buckets": 128,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.10, 0.10),
                "roll": (-0.18, 0.18),
                "pitch": (-0.18, 0.18),
                "yaw": (-0.25, 0.25),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.90, 1.10), "velocity_range": (-0.2, 0.2)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 15.0),
        params={"velocity_range": {"x": (-0.60, 0.60), "y": (-0.60, 0.60)}},
    )


@configclass
class RewardsCfg:
    """Complete BPX quadruped reward suite inherited from the original DWAQ task.

    DWAQ-specific representation/estimator losses are intentionally absent.  All
    terms below describe physical four-legged locomotion and remain valid for the
    public single-frame 45-D Actor.
    """

    # Command tracking rewards.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.40},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.40},
    )

    # Body stability rewards.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-3.0, params={"target_height": 0.48})

    # Foot contact and gait rewards.
    feet_air_time = RewTerm(
        func=bpx_mdp.quadruped_feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "command_name": "base_velocity",
            "threshold": 0.35,
            "command_threshold": 0.15,
            "yaw_scale": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
        },
    )
    # Reward swing feet for clearing the ground during commanded motion.
    swing_foot_clearance = RewTerm(
        func=bpx_mdp.swing_foot_clearance,
        weight=0.04,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "target_height": 0.05,
            "contact_force_threshold": 1.0,
            "command_name": "base_velocity",
            "command_threshold": 0.15,
            "yaw_scale": 0.5,
        },
    )
    feet_contact_force = RewTerm(
        func=bpx_mdp.feet_contact_force_l1,
        weight=-2.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "max_contact_force": 120.0,
        },
    )
    undesired_leg_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(hip|thigh|calf).*"),
        },
    )

    # Joint motion rewards.
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    # Penalize rapid, high-power corrections.  These terms are deliberately
    # stronger than the original gait-shaping values so a resumed high-speed
    # policy learns to retain its tracking performance with smoother commands.
    joint_power_l1 = RewTerm(
        func=bpx_mdp.joint_power_l1,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    # Keep the entire leg close to its nominal walking posture.  Penalizing only
    # thigh folding made it easy for the policy to compensate by locking the
    # lower leg straight; including the knees avoids both deep crouching and
    # overly extended lower legs while retaining enough freedom for a gait.
    leg_posture_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "fl_hip_pitch_joint",
                    "fr_hip_pitch_joint",
                    "hl_hip_pitch_joint",
                    "hr_hip_pitch_joint",
                    "fl_knee_joint",
                    "fr_knee_joint",
                    "hl_knee_joint",
                    "hr_knee_joint",
                ],
                preserve_order=True,
            )
        },
    )

    # Standing and termination rewards.
    stand_still = RewTerm(
        func=bpx_mdp.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.15,
            "yaw_scale": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME),
            "threshold": 5.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(65.0)})


@configclass
class BPXFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: BPXSceneCfg = BPXSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
