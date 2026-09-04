"""BPX-specific quadruped reward terms.

These terms complement Isaac Lab's official locomotion rewards.  They contain no
DWAQ network logic: DWAQ changes representation learning, while these functions
shape the physical quadruped behavior and can be reused by a 45-D feed-forward
Actor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _motion_command_norm(command: torch.Tensor, yaw_scale: float = 0.5) -> torch.Tensor:
    """Return a common locomotion-command magnitude including pure yaw commands."""

    return torch.linalg.vector_norm(
        torch.stack((command[:, 0], command[:, 1], yaw_scale * command[:, 2]), dim=1), dim=1
    )


def quadruped_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_threshold: float = 0.1,
    yaw_scale: float = 0.5,
) -> torch.Tensor:
    """Reward useful swing duration for all four feet.

    Isaac Lab's generic quadruped term gates the reward using only commanded XY
    speed.  BPX is trained for omnidirectional motion, including pure yaw turns,
    so this variant includes yaw in the motion gate.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    command = env.command_manager.get_command(command_name)
    moving = _motion_command_norm(command, yaw_scale=yaw_scale) > command_threshold
    return reward * moving


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.08,
    yaw_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize departure from the default pose only for a true zero command.

    Unlike an XY-only gate, pure yaw commands are treated as movement and are
    therefore not incorrectly pulled back toward the standing pose.
    """

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)

    command = env.command_manager.get_command(command_name)
    standing = _motion_command_norm(command, yaw_scale=yaw_scale) < command_threshold
    return deviation * standing


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute mechanical joint power ``|tau * qdot|``."""

    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def feet_contact_force_l1(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 120.0,
) -> torch.Tensor:
    """Penalize peak foot-contact force above a hardware-friendly threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak_force = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1)
    return torch.sum(torch.relu(peak_force - max_contact_force), dim=1)


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    contact_force_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    yaw_scale: float = 0.5,
) -> torch.Tensor:
    """Reward airborne feet that clear the local flat ground plane.

    The term is evaluated only while a foot is not supporting the robot and
    while the commanded motion is outside the standing deadband.  This avoids
    rewarding repeated foot lifts when the robot should stand still.  The
    reward saturates at ``target_height`` to avoid needlessly high stepping.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    vertical_force = torch.relu(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    airborne = vertical_force < contact_force_threshold
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = torch.clamp(foot_height / target_height, min=0.0, max=1.0)
    command = env.command_manager.get_command(command_name)
    moving = _motion_command_norm(command, yaw_scale=yaw_scale) > command_threshold
    return torch.sum(clearance * airborne, dim=1) * moving
