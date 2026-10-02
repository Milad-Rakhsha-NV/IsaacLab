# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow locomotion environment shared by the Ant and Humanoid tasks."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_apply,
    sample_uniform,
    scale_transform,
    wrap_to_pi,
)
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from .ant.ant_direct_env_cfg import AntEnvCfg
    from .humanoid.humanoid_direct_env_cfg import HumanoidEnvCfg


class LocomotionDirectEnv(DirectRLEnv):
    """Base direct-workflow environment shared by the ant and humanoid locomotion tasks.

    The robot is driven by joint efforts and rewarded for walking towards a distant target while
    staying upright. The MDP mirrors the manager-based ant and humanoid tasks term for term, so both
    workflows train against the same problem and converge to the same reward.
    """

    cfg: AntEnvCfg | HumanoidEnvCfg

    def __init__(self, cfg: AntEnvCfg | HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot, self.terrain, self.joint_wrench = [
            self.scene[name] for name in ("robot", "terrain", "joint_wrench")
        ]
        self.action_scale = self.cfg.action_scale
        # Resolve the gears by joint name, since the joint ordering differs across physics backends.
        # Keep the resolved all-joint Warp index: Newton's indexed command path preserves the
        # backend's articulation order, unlike the later cached all-joint fast path.
        self.joint_gears = torch.ones(self.robot.num_joints, device=self.sim.device)
        joint_ids, _, gears = resolve_matching_names_values(self.cfg.joint_gears, self.robot.joint_names)
        self.joint_gears[joint_ids] = torch.tensor(gears, device=self.sim.device)
        joint_dof_idx, _ = self.robot.find_joints(".*", as_proxy=True)
        self._joint_dof_idx = joint_dof_idx.warp
        # the energy and joint-limit penalties weigh each joint by its gear relative to the largest one
        self.gear_ratio_scaled = self.joint_gears / torch.max(self.joint_gears)
        # resolve against the sensor's own body list: its ordering is backend-specific and does not
        # necessarily match the articulation's body ordering
        self._feet_body_idx, _ = self.joint_wrench.find_bodies(self.cfg.feet_body_names)

        # walk target, placed far enough away that the robot never reaches it
        self.targets = self.scene.env_origins + torch.tensor(
            self.cfg.target_pos, dtype=torch.float32, device=self.sim.device
        )
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Use the archived stable DVI command lifecycle: stage the indexed effort target for
        # every decimation substep, rather than the release-3 all-joint persistent-buffer path.
        forces = self.action_scale * self.joint_gears * torch.clamp(self.actions, -1.0, 1.0)
        self.robot.set_joint_effort_target_index(target=forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position = self.robot.data.root_pos_w.torch
        self.dof_vel = self.robot.data.joint_vel.torch - self.robot.data.default_joint_vel.torch
        torso_rotation = self.robot.data.root_quat_w.torch

        # linear and angular velocity in the torso frame
        self.vel_loc = self.robot.data.root_lin_vel_b.torch
        self.angvel_loc = self.robot.data.root_ang_vel_b.torch

        # planar vector from the torso to the walk target
        to_target = self.targets - self.torso_position
        to_target[:, 2].zero_()

        # alignment of the torso with the world up axis and with the direction to the target
        self.up_proj = -self.robot.data.projected_gravity_b.torch[:, 2]
        heading_vec = quat_apply(torso_rotation, self.robot.data.FORWARD_VEC_B.torch)
        self.heading_proj = torch.sum(heading_vec * normalize(to_target), dim=-1)

        # torso orientation and its misalignment with the direction to the target
        self.roll, _, self.yaw = euler_xyz_from_quat(torso_rotation)
        self.angle_to_target = torch.atan2(to_target[:, 1], to_target[:, 0]) - self.yaw

        self.dof_pos_scaled = scale_transform(
            self.robot.data.joint_pos.torch,
            self.robot.data.soft_joint_pos_limits.torch[..., 0],
            self.robot.data.soft_joint_pos_limits.torch[..., 1],
        )

        # the potential grows as the target gets closer, so its increase measures progress
        self.prev_potentials = self.potentials
        self.potentials = -torch.linalg.norm(to_target, ord=2, dim=-1) / self.step_dt

    def _get_observations(self) -> dict[str, torch.Tensor]:
        feet_wrench = torch.cat(
            (
                self.joint_wrench.data.force.torch[:, self._feet_body_idx],
                self.joint_wrench.data.torque.torch[:, self._feet_body_idx],
            ),
            dim=-1,
        ).view(self.num_envs, -1)
        terms = {
            "torso_height": self.torso_position[:, 2].view(-1, 1),
            "linear_velocity": self.vel_loc,
            "angular_velocity": self.angvel_loc * self.cfg.angular_velocity_scale,
            "yaw": wrap_to_pi(self.yaw).unsqueeze(-1),
            "roll": wrap_to_pi(self.roll).unsqueeze(-1),
            "angle_to_target": wrap_to_pi(self.angle_to_target).unsqueeze(-1),
            "up_projection": self.up_proj.unsqueeze(-1),
            "heading_projection": self.heading_proj.unsqueeze(-1),
            "joint_position": self.dof_pos_scaled,
            "joint_velocity": self.dof_vel * self.cfg.dof_vel_scale,
            "feet_wrench": feet_wrench * self.cfg.contact_force_scale,
            "actions": self.actions,
        }
        if os.environ.get("ISAACLAB_FINITE_DIAGNOSTICS"):
            for name, value in terms.items():
                invalid = ~torch.isfinite(value)
                if torch.any(invalid):
                    env_id, component = invalid.nonzero(as_tuple=False)[0].tolist()
                    raise RuntimeError(
                        f"Non-finite Humanoid observation term {name!r}: env={env_id}, component={component}, "
                        f"value={value[env_id, component].item()}"
                    )
        return {"policy": torch.cat(tuple(terms.values()), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.gear_ratio_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.joint_pos_limits_cost_scale,
            self.cfg.joint_pos_limits_threshold,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.step_dt,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # log survival success rate before resetting (survived = timed out without falling)
        survived = self.reset_time_outs[env_ids].float()
        self.extras.setdefault("log", {})["Metrics/success_rate"] = survived.mean().item()

        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)

        # root state is reset to the default pose, offset into the environment
        # Clone every default-state slice before applying environment offsets or reset noise.
        # Advanced indexing is not guaranteed to allocate for a full contiguous env-id slice;
        # mutating such a view would corrupt the canonical defaults used by later resets.
        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )

        # Joint reset noise and clamping must likewise never modify the canonical defaults.
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.initial_joint_pos_range, joint_pos.shape, joint_pos.device)
        joint_vel += sample_uniform(*self.cfg.initial_joint_vel_range, joint_vel.shape, joint_vel.device)
        joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.robot.data.soft_joint_vel_limits.torch[env_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)

        to_target = self.targets[env_ids] - default_root_pose[:, :3]
        to_target[:, 2].zero_()
        self.potentials[env_ids] = -torch.linalg.norm(to_target, ord=2, dim=-1) / self.step_dt

        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    gear_ratio_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joint_pos_limits_cost_scale: float,
    joint_pos_limits_threshold: float,
    death_cost: float,
    alive_reward_scale: float,
    step_dt: float,
) -> torch.Tensor:
    alive = ~reset_terminated

    # reward for making progress towards the target
    progress_reward = potentials - prev_potentials

    # reward for duration of staying alive
    alive_reward = alive_reward_scale * alive.float()

    # aligning up axis of robot and environment
    up_reward = up_weight * (up_proj > 0.93).float()

    # reward for facing the target, saturating once the alignment is good enough
    heading_reward = heading_weight * torch.where(heading_proj > 0.8, torch.ones_like(heading_proj), heading_proj / 0.8)

    # energy penalty for movement, weighted by how strong each joint's gear is
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * dof_vel * gear_ratio_scaled), dim=-1)

    # penalty for driving joints into their limits, scaled by how far past the threshold they are
    violation = (torch.abs(dof_pos_scaled) - joint_pos_limits_threshold) / (1.0 - joint_pos_limits_threshold)
    dof_at_limit_cost = torch.sum(
        (torch.abs(dof_pos_scaled) > joint_pos_limits_threshold) * violation * gear_ratio_scaled, dim=-1
    )

    # the continuous terms accrue per second, matching the manager's ``step_dt`` scaling
    total_reward = step_dt * (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - joint_pos_limits_cost_scale * dof_at_limit_cost
    )
    # the death cost is a one-off penalty on the step the robot falls over
    return total_reward + death_cost * (~alive).float()
