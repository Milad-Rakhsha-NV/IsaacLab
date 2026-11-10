# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit Go2 task: particle-free RFT training and a fixed policy contract."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import warp as wp

from isaaclab.envs import DirectRLEnv

from .rft_env_cfg import Go2RFTEnvCfg, rigid_substrate_cfg


class Go2RFTEnv(DirectRLEnv):
    """Train on surface loads; actor observes 45 proprioceptive/command values."""

    cfg: Go2RFTEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        from isaaclab_newton.physics import NewtonCfg

        if not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("The Go2 granular task requires the Newton backend")
        from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.rft_solver_cfg import RFTSolverCfg

        for name in ("material_difficulty", "curriculum_initial_difficulty"):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if cfg.curriculum_enabled and (
            not cfg.randomize_material
            or cfg.curriculum_min_episodes < 1
            or cfg.curriculum_patience < 1
            or not 0 < cfg.curriculum_step <= 1
            or not 0 <= cfg.curriculum_anchor_fraction < 1
            or not 0 <= cfg.curriculum_retreat_threshold < cfg.curriculum_success_threshold <= 1
            or not cfg.curriculum_error_threshold > 0
        ):
            raise ValueError("Invalid reset curriculum configuration")

        if cfg.bed_depth is not None:
            if (
                not isinstance(cfg.sim.physics.solver_cfg, RFTSolverCfg)
                or not math.isfinite(cfg.bed_depth)
                or cfg.bed_depth < 0
            ):
                raise ValueError("A nonnegative finite layer requires the RFT solver")
            bottom = cfg.sim.physics.solver_cfg.surface_height - cfg.bed_depth
            cfg.sim.physics.solver_cfg.surface_bottom = bottom
            cfg.scene.substrate = rigid_substrate_cfg(bottom)

        if not isinstance(cfg.sim.physics.solver_cfg, RFTSolverCfg):
            from isaaclab_contrib.coupling import CouplerProxyCfg

            coupled = cfg.sim.physics.solver_cfg
            if not isinstance(coupled, MJWarpSolverCfg) and (
                not isinstance(coupled, CouplerProxyCfg) or not any(e.name == "mpm" for e in coupled.entries)
            ):
                raise ValueError("Select RFT, the rigid-flat control, or the MPM inference coupler")
            if cfg.randomize_material:
                raise ValueError("Disable RFT material randomization for explicitly configured MPM inference")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.error_sum = torch.zeros(self.num_envs, device=self.device)
        self.steps_sum = torch.zeros_like(self.error_sum)
        self.curriculum_difficulty = torch.tensor(cfg.curriculum_initial_difficulty, device=self.device)
        self.curriculum_count = torch.zeros((), device=self.device)
        self.curriculum_successes = torch.zeros_like(self.curriculum_count)
        self.curriculum_errors = torch.zeros_like(self.curriculum_count)
        self.curriculum_streak = torch.zeros_like(self.curriculum_count)
        self.curriculum_complete = self.curriculum_difficulty >= 1
        self.episode_difficulty = torch.full_like(self.error_sum, -1)
        self.episode_anchor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_saved_difficulty = -1.0
        self.rft = self.sim.backend_manager.backend.solver
        from newton.solvers import SolverRFT

        if isinstance(self.rft, SolverRFT):
            self.material = wp.to_torch(self.rft.material)
        else:
            from newton.solvers import SolverMuJoCo
            from newton.solvers.experimental.coupled import SolverCoupledProxy

            if not isinstance(self.rft, SolverMuJoCo) and (
                not isinstance(self.rft, SolverCoupledProxy) or "mpm" not in self.rft.entry_names()
            ):
                raise ValueError("Unsupported granular/control solver")
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
        if self.cfg.curriculum_enabled and self.common_step_counter % 240 == 0 and self.cfg.log_dir:
            difficulty = float(self.curriculum_difficulty)
            if difficulty != self._last_saved_difficulty:
                path = Path(self.cfg.log_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / f"curriculum_{self.common_step_counter}.json").write_text(
                    json.dumps(dict(control_step=self.common_step_counter, difficulty=difficulty), indent=2)
                )
                self._last_saved_difficulty = difficulty

    def surface_elevation(self) -> torch.Tensor:
        """Initial bed/free-surface reference; this is not an evolved MPM height estimate."""
        if self.material is not None:
            return wp.to_torch(self.rft.surface_height)[self.world_ids]
        return self.scene.env_origins[:, 2]

    def relative_base_height(self) -> torch.Tensor:
        reference = self.surface_elevation() if self.cfg.surface_relative_height else self.scene.env_origins[:, 2]
        return self.robot.data.root_pos_w.torch[:, 2] - reference

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
        height = self.relative_base_height()
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
        height = self.relative_base_height()
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
        if self.cfg.curriculum_enabled:
            self._update_curriculum(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.steps_sum[env_ids] = 0
        count = len(env_ids)
        if self.cfg.curriculum_enabled:
            difficulty = self.curriculum_difficulty.expand(count)
            anchors = (torch.rand(count, device=self.device) < self.cfg.curriculum_anchor_fraction) & (difficulty < 1)
            self.episode_difficulty[env_ids] = difficulty
            self.episode_anchor[env_ids] = anchors
            level = torch.where(anchors, 0.0, difficulty)
            material_level = level
        else:
            material_level = self.cfg.material_difficulty
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
            factor = torch.empty(count, device=self.device).uniform_(
                math.log(self.cfg.strength_multiplier_range[0]), math.log(self.cfg.strength_multiplier_range[1])
            )
            density = 1600 + material_level * (density - 1600)
            friction = 0.4 + material_level * (friction - 0.4)
            rough = 1 + material_level * (rough - 1)
            factor = (material_level * factor).exp()
            self.material[world, 0] = density
            self.material[world, 1] = friction
            self.material[world, 2] = rough * friction
            self.material[world, 3] = density * 9.81 * (894 * friction**3 - 386 * friction**2 + 89 * friction) * factor
            wp.to_torch(self.rft.surface_height)[world] = material_level * torch.empty(
                count, device=self.device
            ).uniform_(*self.cfg.bed_height_range)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.surface_relative_height:
            pose[:, 2] += self.surface_elevation()[env_ids] - self.scene.env_origins[env_ids, 2]
        velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        qd = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        """Use only completed, non-anchor episodes sampled at the current difficulty."""
        active = (self.steps_sum[env_ids] > 0) & ~self.episode_anchor[env_ids]
        active &= (self.episode_difficulty[env_ids] - self.curriculum_difficulty).abs() < 1e-5
        completed = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        # Exclude artificially shortened initial episodes when a runner randomizes episode ages.
        active &= self.reset_terminated[env_ids] | (self.steps_sum[env_ids] >= self.max_episode_length - 1)
        self.curriculum_count += active.sum()
        self.curriculum_successes += (active & completed).sum()
        error = self.error_sum[env_ids] / self.steps_sum[env_ids].clamp_min(1)
        self.curriculum_errors += (error * active).sum()
        count = self.curriculum_count.clamp_min(1)
        rate = self.curriculum_successes / count
        mean_error = self.curriculum_errors / count
        ready = self.curriculum_count >= self.cfg.curriculum_min_episodes
        good = (rate >= self.cfg.curriculum_success_threshold) & (mean_error <= self.cfg.curriculum_error_threshold)
        streak = torch.where(good, self.curriculum_streak + 1, 0.0)
        self.curriculum_streak.copy_(torch.where(ready, streak, self.curriculum_streak))
        advance = ready & (self.curriculum_streak >= self.cfg.curriculum_patience)
        retreat = ready & (rate < self.cfg.curriculum_retreat_threshold)
        delta = self.cfg.curriculum_step * (advance.float() - retreat.float()) * ~self.curriculum_complete
        self.curriculum_difficulty.copy_((self.curriculum_difficulty + delta).clamp(0, 1))
        self.curriculum_complete |= self.curriculum_difficulty >= 1
        self.curriculum_streak.copy_(torch.where(advance | retreat, 0.0, self.curriculum_streak))
        self.extras["log"].update(
            {
                "Curriculum/difficulty": self.curriculum_difficulty.clone(),
                "Curriculum/full_range": self.curriculum_complete.float(),
                "Curriculum/window_success": rate,
                "Curriculum/window_error": mean_error,
            }
        )
        self.curriculum_count.copy_(torch.where(ready, 0.0, self.curriculum_count))
        self.curriculum_successes.copy_(torch.where(ready, 0.0, self.curriculum_successes))
        self.curriculum_errors.copy_(torch.where(ready, 0.0, self.curriculum_errors))
