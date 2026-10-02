# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose independent locomotion trajectories into one physics world."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Sequence
from itertools import combinations, product

import torch
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonCfg

from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils import replace
from isaaclab.utils.math import euler_xyz_from_quat, normalize, quat_apply, sample_uniform, scale_transform, wrap_to_pi
from isaaclab.utils.string import resolve_matching_names_values

from isaaclab_tasks.core.locomotion.locomotion_direct_env import compute_rewards

from .multi_robot_env_cfg import MultiAntEnvCfg


class MultiRobotLocomotionEnv(DirectMARLEnv):
    """Train one independent policy per manifest entry across cloned worlds.

    Uses the stock Ant/Humanoid effort, observation and reward contracts. Source task configurations
    serve as robot recipes; they do not create nested environments or simulation contexts.
    """

    cfg: MultiAntEnvCfg

    def __init__(self, cfg: MultiAntEnvCfg, **kwargs):
        cfg = copy.deepcopy(cfg)
        if not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("Multi-robot locomotion currently supports Newton physics only.")
        if not cfg.independent_resets or not cfg.robots:
            raise ValueError("Multi-robot locomotion requires independent resets and a nonempty robot manifest.")
        scene_names = [name for agent in cfg.robots for name in (agent, f"{agent}_wrench")]
        if len(scene_names) != len(set(scene_names)):
            raise ValueError("Robot identifiers conflict with generated wrench sensor names.")
        cfg.possible_agents = list(cfg.robots)
        cfg.observation_spaces = {}
        cfg.action_spaces = {}
        for agent, spec in cfg.robots.items():
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", agent)
                or hasattr(cfg.scene, agent)
                or hasattr(cfg.scene, f"{agent}_wrench")
            ):
                raise ValueError(f"Invalid or reserved robot identifier: {agent!r}")
            task = spec.task
            if not math.isclose(task.sim.dt * task.decimation, cfg.sim.dt * cfg.decimation):
                raise ValueError(f"Robot {agent!r} must use the composed environment's control timestep.")
            path = f"{{ENV_REGEX_NS}}/{agent}"
            robot_cfg = replace(task.scene.robot, prim_path=path)
            robot_cfg.init_state.pos = tuple(a + b for a, b in zip(robot_cfg.init_state.pos, spec.offset, strict=True))
            setattr(cfg.scene, agent, robot_cfg)
            setattr(cfg.scene, f"{agent}_wrench", JointWrenchSensorCfg(prim_path=path))
            cfg.action_spaces[agent] = task.action_space
            cfg.observation_spaces[agent] = task.observation_space
        # The public hook operates on assembled worlds, so filters span separate asset prototypes.
        with newton_builder_world_hook(self._exclude_robot_contacts):
            super().__init__(cfg, **kwargs)
        self.robots = {agent: self.scene[agent] for agent in self.possible_agents}
        self.wrenches = {agent: self.scene[f"{agent}_wrench"] for agent in self.possible_agents}
        self.joint_gears, self.feet_indices, self.targets = {}, {}, {}
        self.actions, self.potentials, self.previous_potentials = {}, {}, {}
        self.episode_returns, self.episode_progress = {}, {}
        self.robot_max_episode_length = {}
        for agent, robot in self.robots.items():
            task = cfg.robots[agent].task
            if robot.num_joints != task.action_space:
                raise ValueError(
                    f"{agent}: asset has {robot.num_joints} joints, recipe declares {task.action_space} actions."
                )
            ids, _, gears = resolve_matching_names_values(task.joint_gears, robot.joint_names)
            self.joint_gears[agent] = torch.ones(robot.num_joints, device=self.device)
            self.joint_gears[agent][ids] = torch.tensor(gears, device=self.device)
            feet, _ = self.wrenches[agent].find_bodies(task.feet_body_names)
            self.feet_indices[agent] = feet
            expected_obs = 12 + 3 * robot.num_joints + 6 * len(feet)
            if task.observation_space != expected_obs:
                raise ValueError(
                    f"{agent}: recipe declares {task.observation_space} observations; asset needs {expected_obs}."
                )
            self.targets[agent] = self.scene.env_origins + torch.tensor(task.target_pos, device=self.device)
            self.targets[agent] += torch.tensor(cfg.robots[agent].offset, device=self.device)
            self.actions[agent] = torch.zeros(self.num_envs, robot.num_joints, device=self.device)
            self.potentials[agent] = torch.zeros(self.num_envs, device=self.device)
            self.previous_potentials[agent] = torch.zeros(self.num_envs, device=self.device)
            self.episode_returns[agent] = torch.zeros(self.num_envs, device=self.device)
            self.episode_progress[agent] = torch.zeros(self.num_envs, device=self.device)
            self.robot_max_episode_length[agent] = math.ceil(task.episode_length_s / self.step_dt)

    def _exclude_robot_contacts(self, builder, world, position, orientation):
        """Exclude cross-robot pairs in this world, retaining each robot's ground contacts."""
        groups = {agent: [] for agent in self.cfg.possible_agents}
        for shape in range(builder.shape_count - 1, -1, -1):
            if builder.shape_world[shape] != world:
                break
            label = builder.shape_label[shape]
            for agent in groups:
                if f"/{agent}/" in label:
                    groups[agent].append(shape)
                    break
        for left, right in combinations(groups.values(), 2):
            for shape_a, shape_b in product(left, right):
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

    def _pre_physics_step(self, actions):
        if actions.keys() != self.actions.keys():
            raise ValueError("Actions must contain exactly the configured robot identifiers.")
        for agent, action in actions.items():
            if action.shape != self.actions[agent].shape:
                raise ValueError(
                    f"Wrong action shape for {agent}: {action.shape}; expected {self.actions[agent].shape}."
                )
            self.actions[agent].copy_(action)
            effort = self.cfg.robots[agent].task.action_scale * self.joint_gears[agent] * action.clamp(-1, 1)
            self.robots[agent].actuators.target_command.set_effort_index(value=effort)

    def _apply_action(self):
        pass

    def _get_dones(self):
        terminated, truncated = {}, {}
        for agent, robot in self.robots.items():
            task = self.cfg.robots[agent].task
            height = robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
            terminated[agent] = height < task.termination_height
            truncated[agent] = self.agent_episode_length_buf[agent] >= self.robot_max_episode_length[agent]
            self.previous_potentials[agent].copy_(self.potentials[agent])
            to_target = self.targets[agent] - robot.data.root_pos_w.torch
            self.potentials[agent].copy_(-torch.linalg.vector_norm(to_target[:, :2], dim=-1) / self.step_dt)
        return terminated, truncated

    def _joint_state(self, agent):
        data = self.robots[agent].data
        limits = data.soft_joint_pos_limits.torch
        return (
            scale_transform(data.joint_pos.torch, limits[..., 0], limits[..., 1]),
            data.joint_vel.torch - data.default_joint_vel.torch,
        )

    def _heading(self, agent):
        data = self.robots[agent].data
        to_target = self.targets[agent] - data.root_pos_w.torch
        to_target[:, 2] = 0
        forward = quat_apply(data.root_quat_w.torch, data.FORWARD_VEC_B.torch)
        return to_target, torch.sum(forward * normalize(to_target), dim=-1)

    def _get_observations(self):
        observations = {}
        for agent, robot in self.robots.items():
            cfg = self.cfg.robots[agent].task
            data = robot.data
            q, dq = self._joint_state(agent)
            roll, _, yaw = euler_xyz_from_quat(data.root_quat_w.torch)
            to_target, heading = self._heading(agent)
            wrench = self.wrenches[agent].data
            feet = self.feet_indices[agent]
            observations[agent] = torch.cat(
                (
                    (data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1),
                    data.root_lin_vel_b.torch,
                    data.root_ang_vel_b.torch * cfg.angular_velocity_scale,
                    wrap_to_pi(yaw).unsqueeze(-1),
                    wrap_to_pi(roll).unsqueeze(-1),
                    wrap_to_pi(torch.atan2(to_target[:, 1], to_target[:, 0]) - yaw).unsqueeze(-1),
                    -data.projected_gravity_b.torch[:, 2:3],
                    heading.unsqueeze(-1),
                    q,
                    dq * cfg.dof_vel_scale,
                    torch.cat((wrench.force.torch[:, feet], wrench.torque.torch[:, feet]), dim=-1).flatten(1)
                    * cfg.contact_force_scale,
                    self.actions[agent],
                ),
                dim=-1,
            )
        return observations

    def _get_rewards(self):
        rewards = {}
        for agent, robot in self.robots.items():
            cfg = self.cfg.robots[agent].task
            q, dq = self._joint_state(agent)
            _, heading = self._heading(agent)
            rewards[agent] = compute_rewards(
                self.actions[agent],
                self.terminated_dict[agent],
                cfg.up_weight,
                cfg.heading_weight,
                heading,
                -robot.data.projected_gravity_b.torch[:, 2],
                dq,
                q,
                self.joint_gears[agent] / self.joint_gears[agent].max(),
                self.potentials[agent],
                self.previous_potentials[agent],
                cfg.actions_cost_scale,
                cfg.energy_cost_scale,
                cfg.joint_pos_limits_cost_scale,
                cfg.joint_pos_limits_threshold,
                cfg.death_cost,
                cfg.alive_reward_scale,
                self.step_dt,
            )
            self.episode_returns[agent] += rewards[agent]
            self.episode_progress[agent] += (self.potentials[agent] - self.previous_potentials[agent]) * self.step_dt
            self.extras[agent]["log"] = {}
        return rewards

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        for agent in self.possible_agents:
            self._reset_agent_idx(agent, env_ids)

    def _reset_agent_idx(self, agent, env_ids):
        robot = self.robots[agent]
        cfg = self.cfg.robots[agent].task
        lengths = self.agent_episode_length_buf[agent][env_ids]
        completed = lengths > 0
        if completed.any():
            ids = env_ids[completed]
            self.extras[agent]["log"] = {
                "Episode/return": self.episode_returns[agent][ids].mean(),
                "Episode/progress_m": self.episode_progress[agent][ids].mean(),
                "Episode/length_s": lengths[completed].float().mean() * self.step_dt,
                "Episode/survival": (self.time_out_dict[agent][ids] & ~self.terminated_dict[agent][ids]).float().mean(),
            }
        robot.reset(env_ids)
        self.wrenches[agent].reset(env_ids)
        self.actions[agent][env_ids] = 0
        self.episode_returns[agent][env_ids] = 0
        self.episode_progress[agent][env_ids] = 0
        root_pose = robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=robot.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        q = robot.data.default_joint_pos.torch[env_ids].clone()
        dq = robot.data.default_joint_vel.torch[env_ids].clone()
        q += sample_uniform(*cfg.initial_joint_pos_range, q.shape, self.device)
        dq += sample_uniform(*cfg.initial_joint_vel_range, dq.shape, self.device)
        limits = robot.data.soft_joint_pos_limits.torch[env_ids]
        robot.write_joint_state_to_sim_index(
            position=q.clamp(limits[..., 0], limits[..., 1]), velocity=dq, env_ids=env_ids
        )
        distance = torch.linalg.vector_norm((self.targets[agent][env_ids] - root_pose[:, :3])[:, :2], dim=-1)
        self.potentials[agent][env_ids] = -distance / self.step_dt
        self.previous_potentials[agent][env_ids] = self.potentials[agent][env_ids]
