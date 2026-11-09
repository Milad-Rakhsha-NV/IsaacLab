# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online residual world-model learning and short-horizon analytic actor--critic."""

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .math_utils import features, state_delta
from .networks import Actor, Critic, WorldModel, goal
from .replay import Replay


@dataclass
class TrainConfig:
    num_envs: int = 256
    policy_batch: int = 512
    model_batch: int = 256
    model_batches: int = 10
    replay_steps: int = 256
    model_horizon: int = 8
    horizon: int = 32
    learning_rate: float = 0.0002
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise: float = 0.1
    sensor_noise: bool = True
    utd: float = 0.4
    critic_epochs: int = 5
    seed: int = 42
    compile: bool = False
    small_networks: bool = False

    def validate(self):
        if min(self.num_envs, self.policy_batch, self.model_batch, self.model_batches, self.critic_epochs) < 1:
            raise ValueError("Batch sizes, environment count and optimizer counts must be positive")
        if self.replay_steps < max(self.model_horizon, 10) or self.model_horizon < 1 or self.horizon < 1:
            raise ValueError("Invalid replay or rollout horizons")
        if not 0 < self.utd <= 1 or not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid update ratio or discount parameters")


@contextmanager
def frozen(*modules):
    params = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags, strict=True):
            p.requires_grad_(flag)


def optimizer_step(loss, module, optimizer, element_clip=False):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
    if element_clip:
        for p in module.parameters():
            if p.grad is not None:
                p.grad.clamp_(-0.1, 0.1)
    optimizer.step()
    return float(norm.detach())


class Learner:
    def __init__(self, env, cfg):
        cfg.validate()
        self.env, self.cfg = env, cfg
        self.task, self.library, self.device = env.task, env.library, env.device
        wm_hidden = (256, 256, 256) if cfg.small_networks else (2048, 2048, 2048)
        pi_hidden = (256, 128, 64) if cfg.small_networks else (2048, 1024, 512, 256)
        self.model = WorldModel(wm_hidden).to(self.device)
        self.actor = Actor(pi_hidden).to(self.device)
        self.critic = Critic().to(self.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.replay = Replay(cfg.replay_steps, cfg.num_envs, self.device)
        self.iteration, self.updates = 0, 0
        self.window_loss = self.model_loss
        self.imagined_step = self._imagined_step
        if cfg.compile:
            self.window_loss = torch.compile(self.model_loss)
            self.imagined_step = torch.compile(self._imagined_step)

    @torch.no_grad()
    def collect(self):
        env = self.env
        state = env.read()
        future = self.library.future(env.clip, env.frame)
        target = goal(state, future)
        self.actor.obs_norm.update(env.history)
        self.actor.goal_norm.update(target)
        action = self.actor(env.history, target)
        action = (action + torch.randn_like(action) * self.cfg.action_noise).clamp(-10, 10)
        result = env.step(action, noise=self.cfg.sensor_noise)
        self.replay.add(*(result[k] for k in ("state", "successor", "action", "previous", "episode", "clip", "frame")))
        env.reset((result["failure"] | result["timeout"]).nonzero().flatten())
        return {
            "sim_reward": float(result["reward"].mean()),
            "sim_failure_fraction": float(result["failure"].float().mean()),
        }

    def model_loss(self, s, actions, targets):
        loss = s.new_zeros(())
        for i in range(actions.shape[1]):
            s = self.model(s, actions[:, i])
            error = state_delta(targets[:, i], s) / self.model.outputs.std
            loss = loss + error.square().mean()
        return loss / actions.shape[1]

    def fit_model(self):
        losses = []
        norm = 0
        for _ in range(self.cfg.model_batches):
            s, a, target = self.replay.windows(self.cfg.model_batch, self.cfg.model_horizon)
            with torch.no_grad():
                self.model.inputs.update(torch.cat((features(s), a[:, 0]), -1))
                self.model.outputs.update(state_delta(s, target[:, 0]))
            loss = self.window_loss(s, a, target)
            norm = optimizer_step(loss, self.model, self.model_optimizer)
            losses.append(float(loss.detach()))
        return {"model_loss": sum(losses) / len(losses), "model_gradient_norm": norm}

    def _imagined_step(self, state, history, previous, future, reference, ref_kin):
        target = goal(state, future)
        inputs = self.critic.inputs(state, target)
        action = self.actor(history, target)
        successor = self.model(state, action)
        reward = self.task.reward(successor, reference, action, previous, ref_kin)
        done = self.task.termination(successor, reference, soft=True)
        obs = self.task.observation(successor, action, noise=self.cfg.sensor_noise)
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        return successor, history, action, reward, done, inputs

    def improve_policy(self):
        cfg = self.cfg
        sample = self.replay.starts(cfg.policy_batch, 10)
        s, previous = sample["state"], sample["previous"]
        history = self.task.observation(sample["history_states"], sample["history_actions"], noise=cfg.sensor_noise)
        clip, frame = sample["clip"], sample["frame"]
        times = frame[:, None] + torch.arange(cfg.horizon + 1, device=self.device)
        refs = self.library.reference(clip[:, None], times)
        future = self.library.future(clip[:, None], times)
        with torch.no_grad():
            ref_kin = self.task.kinematics(refs)
        survival = torch.ones(cfg.policy_batch, device=self.device)
        objective = torch.zeros_like(survival)
        records = {"inputs": [], "rewards": [], "discounts": []}
        # Freeze weights, not operations: state and action Jacobians remain live.
        with frozen(self.model, self.critic):
            for t in range(cfg.horizon):
                s, history, action, reward, done, inputs = self.imagined_step(
                    s, history, previous, future[:, t], refs[:, t + 1], tuple(x[:, t + 1] for x in ref_kin)
                )
                alive = (times[:, t] < self.library.lengths[clip] - 1).float()
                objective = objective + cfg.gamma**t * survival * reward * alive
                discount = (1 - done) * alive
                survival = survival * discount
                records["inputs"].append(inputs.detach())
                records["rewards"].append((reward * alive).detach())
                records["discounts"].append(discount.detach())
                previous = action
            final_inputs = self.critic.inputs(s, goal(s, future[:, -1]))
            tail = self.critic(final_inputs)
            objective = objective + cfg.gamma**cfg.horizon * survival * tail
            # Divide by the finite-horizon discount sum to keep update magnitudes stable.
            loss = -objective.mean() / ((1 - cfg.gamma**cfg.horizon) / (1 - cfg.gamma))
            norm = optimizer_step(loss, self.actor, self.actor_optimizer, element_clip=True)
        inputs = torch.stack(records["inputs"])
        rewards = torch.stack(records["rewards"])
        discounts = torch.stack(records["discounts"])
        with torch.no_grad():
            values = self.critic(inputs)
            target = tail.detach()
            next_value = tail.detach()
            returns = []
            for t in reversed(range(cfg.horizon)):
                target = rewards[t] + cfg.gamma * discounts[t] * (
                    (1 - cfg.gae_lambda) * next_value + cfg.gae_lambda * target
                )
                returns.append(target)
                next_value = values[t]
            targets = torch.stack(list(reversed(returns))).flatten()
            flat_inputs = inputs.flatten(0, 1)
            self.critic.norm.update(flat_inputs)
        value_loss = 0
        for _ in range(cfg.critic_epochs):
            indices = torch.randperm(len(targets), device=self.device)
            for batch in indices.tensor_split(4):
                if len(batch) == 0:
                    continue
                vl = (self.critic(flat_inputs[batch]) - targets[batch]).square().mean()
                optimizer_step(vl, self.critic, self.critic_optimizer)
                value_loss += float(vl.detach()) / (4 * cfg.critic_epochs)
        self.updates += 1
        return {
            "actor_loss": float(loss.detach()),
            "actor_gradient_norm": norm,
            "critic_loss": value_loss,
            "imagined_reward": float(rewards.mean()),
            "imagined_survival": float(survival.detach().mean()),
        }

    @torch.no_grad()
    def prediction_errors(self, windows):
        s, actions, targets = windows
        persistence = s
        prediction = s
        for i in range(actions.shape[1]):
            prediction = self.model(prediction, actions[:, i])
        error = state_delta(targets[:, -1], prediction)
        baseline = state_delta(targets[:, -1], persistence)
        return {
            "position_rmse_m": float(error[:, :3].square().mean().sqrt()),
            "orientation_rmse_rad": float(error[:, 3:6].square().mean().sqrt()),
            "joint_rmse_rad": float(error[:, 12:41].square().mean().sqrt()),
            "normalized_error": float((error / self.model.outputs.std).square().mean()),
            "persistence_normalized_error": float((baseline / self.model.outputs.std).square().mean()),
        }

    def save(self, path):
        checkpoint = {
            "format": 1,
            "config": asdict(self.cfg),
            "motion_sha256": self.library.sha256,
            "clips": self.library.names,
            "provenance": self.library.provenance,
            "iteration": self.iteration,
            "updates": self.updates,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        for key in ("model", "actor", "critic", "model_optimizer", "actor_optimizer", "critic_optimizer"):
            checkpoint[key] = getattr(self, key).state_dict()
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["format"] != 1 or checkpoint["config"] != asdict(self.cfg):
            raise ValueError("Checkpoint configuration differs from the requested experiment")
        if checkpoint["motion_sha256"] != self.library.sha256 or checkpoint["clips"] != self.library.names:
            raise ValueError("Checkpoint motion library differs from the requested experiment")
        for key in ("model", "actor", "critic", "model_optimizer", "actor_optimizer", "critic_optimizer"):
            getattr(self, key).load_state_dict(checkpoint[key])
        self.iteration, self.updates = checkpoint["iteration"], checkpoint["updates"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
