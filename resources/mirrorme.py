"""BPX articulation configuration loaded from the BPX URDF submodule."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _asset_root() -> Path:
    override = os.environ.get("MIRRORME_BPX_ASSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "BPX"


ASSET_ROOT = _asset_root()
BPX_URDF_PATH = ASSET_ROOT / "bpx" / "urdf" / "bpx.urdf"


BPX_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(BPX_URDF_PATH),
        fix_base=False,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=30.0,
                damping=1.0,
            ),
            target_type="position",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "fl_hip_roll_joint": 0.0,
            "fr_hip_roll_joint": 0.0,
            "hl_hip_roll_joint": 0.0,
            "hr_hip_roll_joint": 0.0,
            "fl_hip_pitch_joint": 0.8,
            "fr_hip_pitch_joint": 0.8,
            "hl_hip_pitch_joint": 0.8,
            "hr_hip_pitch_joint": 0.8,
            "fl_knee_joint": -1.5,
            "fr_knee_joint": -1.5,
            "hl_knee_joint": -1.5,
            "hr_knee_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
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
            ],
            effort_limit_sim=30.0,
            stiffness=30.0,
            damping=1.0,
            armature=0.003,
        )
    },
)
