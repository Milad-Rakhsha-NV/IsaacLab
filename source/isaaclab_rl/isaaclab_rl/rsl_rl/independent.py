# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate independent RSL-RL PPO learners in a single multi-robot simulation."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv


class RslRlIndependentVecEnvWrapper:
    """Expose per-robot tensors without merging rewards, spaces or episode boundaries."""

    def __init__(self, env, clip_actions: float | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        if not isinstance(self.unwrapped, DirectMARLEnv) or not self.unwrapped.cfg.independent_resets:
            raise ValueError("Independent RSL-RL requires a DirectMARLEnv with independent_resets=True.")
        if not self.unwrapped.cfg.compute_final_obs or self.unwrapped.cfg.state_space:
            raise ValueError("Independent RSL-RL requires final observations and no centralized state.")
        self.cfg = self.unwrapped.cfg
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.possible_agents = self.unwrapped.possible_agents
        self.clip_actions = clip_actions
        self.reset()

    def observation(self, value: torch.Tensor) -> TensorDict:
        """Wrap one robot's flat observation in RSL-RL's observation-group interface."""
        return TensorDict({"policy": value}, batch_size=[self.num_envs])

    def get_observations(self) -> dict[str, TensorDict]:
        """Return observations from the last physical step or reset."""
        return self.observations

    def reset(self):
        """Explicitly reset all robots; normal episode resets belong to the environment."""
        observations, extras = self.env.reset()
        self.observations = {agent: self.observation(obs) for agent, obs in observations.items()}
        return self.observations, extras

    def step(self, actions):
        """Advance the shared simulation exactly once with every robot's action."""
        if self.clip_actions is not None:
            actions = {agent: action.clamp(-self.clip_actions, self.clip_actions) for agent, action in actions.items()}
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.observations = {agent: self.observation(obs) for agent, obs in observations.items()}
        dones = {agent: terminated[agent] | truncated[agent] for agent in self.possible_agents}
        extras = {}
        for agent in self.possible_agents:
            extras[agent] = dict(infos[agent])
            if not self.cfg.is_finite_horizon:
                extras[agent]["time_outs"] = truncated[agent] & ~terminated[agent]
        return self.observations, rewards, dones, extras

    def close(self):
        """Close the owning environment."""
        self.env.close()


class _RobotView:
    """Read-only construction metadata for one PPO; only the coordinator steps physics."""

    def __init__(self, env: RslRlIndependentVecEnvWrapper, agent: str):
        self.env, self.agent = env, agent
        self.cfg, self.num_envs, self.device = env.cfg, env.num_envs, env.device
        self.num_actions = gym.spaces.flatdim(env.unwrapped.action_spaces[agent])

    def get_observations(self):
        return self.env.get_observations()[self.agent]


class IndependentPolicy:
    """Dictionary policy used by the standard playback loop."""

    def __init__(self, policies):
        self.policies = policies

    def __call__(self, observations):
        return {agent: policy(observations[agent]) for agent, policy in self.policies.items()}

    def reset(self, dones):
        """Reset each policy's episode state using its own done mask."""
        for agent, policy in self.policies.items():
            policy.reset(dones[agent])


class IndependentOnPolicyRunner:
    """Collect one shared physical rollout and optimize separate ordinary PPO instances.

    Supports feedforward PPO with TensorBoard on one device. Checkpoints store all robots'
    model/normalizer/optimizer states together; simulator and rollout state are not restored.
    """

    def __init__(self, env: RslRlIndependentVecEnvWrapper, train_cfg: dict, log_dir=None, device="cpu"):
        if not isinstance(env, RslRlIndependentVecEnvWrapper):
            raise ValueError("IndependentOnPolicyRunner requires RslRlIndependentVecEnvWrapper.")
        if int(os.getenv("WORLD_SIZE", "1")) != 1 or train_cfg.get("logger", "tensorboard") != "tensorboard":
            raise ValueError("Independent RSL-RL currently supports one process and TensorBoard logging.")
        if train_cfg["algorithm"]["class_name"] != "PPO":
            raise ValueError("Independent RSL-RL currently supports PPO only.")
        if any(train_cfg["algorithm"].get(key) for key in ("rnd_cfg", "symmetry_cfg")):
            raise ValueError("Independent RSL-RL does not yet support RND or symmetry extensions.")
        if any(train_cfg[key]["class_name"] != "MLPModel" for key in ("actor", "critic")):
            raise ValueError("Independent RSL-RL currently supports feedforward MLPModel actors and critics.")
        self.env, self.cfg, self.device, self.log_dir = env, copy.deepcopy(train_cfg), device, log_dir
        self.current_learning_iteration = 0  # completed updates; resume starts with the next update
        self.runners = {
            agent: OnPolicyRunner(
                _RobotView(env, agent),
                copy.deepcopy(train_cfg),
                log_dir=str(Path(log_dir) / agent) if log_dir else None,
                device=device,
            )
            for agent in env.possible_agents
        }

    def record_transition(self, observations, rewards, dones, infos):
        """Bootstrap pure timeouts from final observations, then let PPO store/reset normally."""
        for agent, runner in self.runners.items():
            alg = runner.alg
            reward = rewards[agent].to(self.device)
            extras = dict(infos[agent])
            timeout = extras.pop("time_outs", None)
            if timeout is not None and timeout.any():
                final_obs = self.env.observation(extras["final_obs"]).to(self.device)
                reward = reward + alg.gamma * alg.critic(final_obs).squeeze(-1) * timeout.to(self.device)
            # Remove time_outs to prevent upstream PPO applying its stored-state bootstrap a second time.
            alg.process_env_step(observations[agent], reward, dones[agent].to(self.device), extras)
            runner.logger.process_env_step(rewards[agent].to(self.device), dones[agent].to(self.device), extras)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Train all policies while advancing physics once per rollout step."""
        if init_at_random_ep_len:
            raise ValueError("Independent RSL-RL requires init_at_random_ep_len=False for complete episode metrics.")
        observations = {agent: obs.to(self.device) for agent, obs in self.env.get_observations().items()}
        for runner in self.runners.values():
            runner.alg.train_mode()
            runner.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        try:
            for it in range(start_it, total_it):
                start = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = {
                            agent: runner.alg.act(observations[agent]).to(self.env.device)
                            for agent, runner in self.runners.items()
                        }
                        observations, rewards, dones, infos = self.env.step(actions)
                        observations = {agent: obs.to(self.device) for agent, obs in observations.items()}
                        if self.cfg.get("check_for_nan", True):
                            for agent in self.runners:
                                check_nan(observations[agent], rewards[agent], dones[agent])
                        self.record_transition(observations, rewards, dones, infos)
                    for agent, runner in self.runners.items():
                        runner.alg.compute_returns(observations[agent])
                collect_time = time.perf_counter() - start
                start = time.perf_counter()
                losses = {agent: runner.alg.update() for agent, runner in self.runners.items()}
                learn_time = time.perf_counter() - start
                self.current_learning_iteration = it + 1
                for agent, runner in self.runners.items():
                    runner.logger.log(
                        it=it,
                        start_it=start_it,
                        total_it=total_it,
                        collect_time=collect_time,
                        learn_time=learn_time,
                        loss_dict=losses[agent],
                        learning_rate=runner.alg.learning_rate,
                        action_std=runner.alg.get_policy().output_std,
                        rnd_weight=None,
                        print_minimal=True,
                    )
                if self.log_dir and self.current_learning_iteration % self.cfg["save_interval"] == 0:
                    self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
            if self.log_dir:
                self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
        finally:
            for runner in self.runners.values():
                runner.logger.stop_logging_writer()

    def save(self, path: str, infos=None):
        """Save all policies and their optimizers at the same completed iteration."""
        torch.save(
            {
                "format": "isaaclab_independent_rsl_rl_v1",
                "agents": {agent: runner.alg.save() for agent, runner in self.runners.items()},
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        """Restore a checkpoint with exactly the configured robot IDs."""
        saved = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if saved.get("format") != "isaaclab_independent_rsl_rl_v1" or set(saved["agents"]) != set(self.runners):
            raise ValueError("Checkpoint format or robot IDs do not match the independent environment.")
        load_iteration = []
        for agent, runner in self.runners.items():
            load_iteration.append(runner.alg.load(saved["agents"][agent], load_cfg, strict))
        if all(load_iteration):
            self.current_learning_iteration = saved["iter"]
        return saved["infos"]

    def get_inference_policy(self, device=None):
        """Return a deterministic dictionary policy with saved per-robot normalization."""
        return IndependentPolicy({agent: runner.get_inference_policy(device) for agent, runner in self.runners.items()})

    def add_git_repo_to_log(self, repo_file_path):
        """Attach source metadata to each robot's training log."""
        for runner in self.runners.values():
            runner.add_git_repo_to_log(repo_file_path)

    def export_policy_to_jit(self, path: str, filename="policy.pt"):
        """Export a separate policy artifact per robot ID."""
        for agent, runner in self.runners.items():
            runner.export_policy_to_jit(str(Path(path) / agent), filename)

    def export_policy_to_onnx(self, path: str, filename="policy.onnx"):
        """Export a separate policy artifact per robot ID."""
        for agent, runner in self.runners.items():
            runner.export_policy_to_onnx(str(Path(path) / agent), filename)
