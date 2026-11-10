# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit Go2 task: particle-free RFT training and a fixed policy contract."""

from __future__ import annotations

import math
import re

import torch
import warp as wp

from isaaclab.envs import DirectRLEnv

from .rft_env_cfg import Go2RFTEnvCfg


class Go2RFTEnv(DirectRLEnv):
    """Train on surface loads; actor observes 45 proprioceptive/command values."""

    cfg: Go2RFTEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        from isaaclab_newton.physics import NewtonCfg

        if not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("The Go2 granular task requires the Newton backend")
        from isaaclab_newton.physics.rft_solver_cfg import RFTSolverCfg

        if not isinstance(cfg.sim.physics.solver_cfg, RFTSolverCfg):
            from isaaclab_contrib.coupling import CouplerProxyCfg

            coupled = cfg.sim.physics.solver_cfg
            if not isinstance(coupled, CouplerProxyCfg) or not any(e.name == "mpm" for e in coupled.entries):
                raise ValueError("Select RFT for training or an MPM proxy configuration for inference")
            if cfg.randomize_material:
                raise ValueError("Disable RFT material randomization for explicitly configured MPM inference")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.error_sum = torch.zeros(self.num_envs, device=self.device)
        self.steps_sum = torch.zeros_like(self.error_sum)
        self.rft = self.sim.backend_manager.backend.solver
        from newton.solvers import SolverRFT

        if isinstance(self.rft, SolverRFT):
            self.material = wp.to_torch(self.rft.material)
        else:
            from newton.solvers.experimental.coupled import SolverCoupledProxy

            if not isinstance(self.rft, SolverCoupledProxy) or "mpm" not in self.rft.entry_names():
                raise ValueError("Use the RFT solver for training or the MPM proxy coupler for inference")
            if cfg.randomize_material:
                raise ValueError("MPM inference uses explicit material settings; disable RFT material randomization")
            self.material = None
        worlds = self.rft.model.body_world.numpy()
        mapping = {}
        for body, label in enumerate(self.rft.model.body_label):
            match = re.search(r"/env_(\d+)/Robot/", label)
            if match:
                mapping[int(match.group(1))] = int(worlds[body])
        if set(mapping) != set(range(self.num_envs)):
            raise ValueError("RFT worlds do not match the cloned Go2 environments")
        self.world_ids = torch.tensor([mapping[i] for i in range(self.num_envs)], device=self.device)

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-5.0, 5.0))

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(
            value=self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions
        )

    def _get_observations(self):
        data = self.robot.data
        proprio = torch.cat(
            (
                data.root_ang_vel_b.torch,
                data.projected_gravity_b.torch,
                self.commands,
                data.joint_pos.torch - data.default_joint_pos.torch,
                data.joint_vel.torch,
                self.actions,
            ),
            dim=-1,
        )
        critic = torch.cat((proprio, data.root_lin_vel_b.torch), dim=-1)
        if self.cfg.observation_noise:
            proprio = proprio + 0.01 * torch.randn_like(proprio)
        return {"policy": proprio, "critic": critic}

    def _get_rewards(self):
        d = self.robot.data
        xy = ((d.root_lin_vel_b.torch[:, :2] - self.commands[:, :2]) ** 2).sum(-1)
        yaw = (d.root_ang_vel_b.torch[:, 2] - self.commands[:, 2]) ** 2
        tilt = (d.projected_gravity_b.torch[:, :2] ** 2).sum(-1)
        height = d.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        effort = (self.robot.actuators.applied_effort.torch**2).sum(-1)
        smooth = ((self.actions - self.previous_actions) ** 2).sum(-1)
        reward = (
            2.0 * torch.exp(-xy / 0.16)
            + 0.5 * torch.exp(-yaw / 0.25)
            + 0.2
            - 2.0 * tilt
            - 30.0 * (height - self.cfg.target_height) ** 2
            - 0.0002 * effort
            - 0.005 * smooth
        ) * self.step_dt
        reward -= self.reset_terminated.float()
        self.error_sum += xy.sqrt()
        self.steps_sum += 1
        return reward

    def _get_dones(self):
        d = self.robot.data
        height = d.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        failed = (height < 0.16) | (d.projected_gravity_b.torch[:, 2] > -0.3)
        failed |= ~torch.isfinite(d.root_pos_w.torch).all(-1)
        return failed, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.extras.setdefault("log", {})["Metrics/velocity_error"] = (
            self.error_sum[env_ids] / self.steps_sum[env_ids].clamp_min(1)
        ).mean()
        self.extras["log"]["Metrics/timeout_fraction"] = self.reset_time_outs[env_ids].float().mean()
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.steps_sum[env_ids] = 0
        count = len(env_ids)
        self.commands[env_ids, 0] = self.cfg.command_speed
        self.commands[env_ids, 1:] = 0
        if self.cfg.randomize_commands:
            self.commands[env_ids, 0] = torch.empty(count, device=self.device).uniform_(0.1, 0.5)
            self.commands[env_ids, 1] = torch.empty(count, device=self.device).uniform_(-0.1, 0.1)
            self.commands[env_ids, 2] = torch.empty(count, device=self.device).uniform_(-0.3, 0.3)
            stand = torch.rand(count, device=self.device) < 0.1
            self.commands[env_ids[stand]] = 0
        world = self.world_ids[env_ids]
        if self.cfg.randomize_material:
            density = torch.empty(count, device=self.device).uniform_(*self.cfg.density_range)
            friction = torch.empty(count, device=self.device).uniform_(*self.cfg.friction_range)
            rough = torch.empty(count, device=self.device).uniform_(*self.cfg.roughness_range)
            factor = (
                torch.empty(count, device=self.device)
                .uniform_(
                    math.log(self.cfg.strength_multiplier_range[0]), math.log(self.cfg.strength_multiplier_range[1])
                )
                .exp()
            )
            self.material[world, 0] = density
            self.material[world, 1] = friction
            self.material[world, 2] = rough * friction
            self.material[world, 3] = density * 9.81 * (894 * friction**3 - 386 * friction**2 + 89 * friction) * factor
            wp.to_torch(self.rft.surface_height)[world] = torch.empty(count, device=self.device).uniform_(
                *self.cfg.bed_height_range
            )
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        qd = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
