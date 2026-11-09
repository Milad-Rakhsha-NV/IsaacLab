# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-competence and explicit inertial-information controls for the mechanical pilot."""

import torch
import torch.nn.functional as F
from rsl_rl.modules import EmpiricalNormalization
from torch import nn
from torch.distributions import Normal, kl_divergence

from .mechanical_observations import STATIC_SLICES
from .mechanical_policy import FeedForwardModel, MechanicalActor
from .mechanical_training import mechanical_train_cfg


class MechanicalSpecialists(FeedForwardModel):
    """Separate [256,128,64] mean networks and input statistics for each fixed prototype.

    All dynamic information in the token actor is present in the flat observation. Fixed
    mechanics are implicit in each specialist's identity/weights. Scalar exploration, pooled
    PPO normalization, adaptive learning rate and clipping match the shared-policy experiment;
    this is network separation, not fully independent optimizers. There is no unseen-body head.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, observation_widths, action_counts):
        super().__init__()
        if len(observation_widths) != len(action_counts) or max(action_counts) > output_dim:
            raise ValueError("Specialist dimensions must match the source manifest and action padding.")
        self.widths, self.action_counts = observation_widths, action_counts
        self.norms = nn.ModuleList([EmpiricalNormalization(width) for width in observation_widths])
        self.policies = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, 64),
                    nn.ELU(),
                    nn.Linear(64, actions),
                )
                for width, actions in zip(observation_widths, action_counts, strict=True)
            ]
        )
        self.log_std = nn.Parameter(torch.zeros(()))
        self.distribution, self.mask = None, None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = obs["critic"].new_zeros(obs["action_mask"].shape)
        for index, (norm, policy, width, actions) in enumerate(
            zip(self.norms, self.policies, self.widths, self.action_counts, strict=True)
        ):
            selected = obs["embodiment"] == index
            # Empty forwards create zero gradients, allowing Adam momentum to move an absent head.
            if not selected.any():
                continue
            mean[selected] = F.pad(policy(norm(obs["critic"][selected, :width])), (0, mean.shape[1] - actions))
        mean = mean * obs["action_mask"]
        if stochastic_output:
            self.mask = obs["action_mask"]
            self.distribution = Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean))
            return self.distribution.sample() * self.mask
        return mean

    def update_normalization(self, obs):
        for index, (norm, width) in enumerate(zip(self.norms, self.widths, strict=True)):
            selected = obs["embodiment"] == index
            if selected.any():
                norm.update(obs["critic"][selected, :width])

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.stddev

    @property
    def output_entropy(self):
        return (self.distribution.entropy() * self.mask).sum(-1)

    @property
    def output_distribution_params(self):
        return self.output_mean, self.output_std

    def get_output_log_prob(self, outputs):
        return (self.distribution.log_prob(outputs) * self.mask).sum(-1)

    def get_kl_divergence(self, old_params, new_params):
        return (kl_divergence(Normal(*old_params), Normal(*new_params)) * self.mask).sum(-1)


class MechanicalWithoutInertia(MechanicalActor):
    """Retrain without explicit mass, COM, full inertia, armature or total-mass inputs.

    Physical dynamics, geometry, body state, actuator settings, rewards and critic inputs stay
    available. Inertial effects may still be inferred from state and wrenches. This tests the
    benefit of explicit inertial descriptors, not whether the policy learns physical dynamics.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        static_keep = torch.ones(obs["static"].shape[-1])
        for name in ("body_mass", "body_com", "body_inertia", "joint_armature"):
            static_keep[STATIC_SLICES[name]] = 0
        global_keep = torch.ones(obs["global"].shape[-1])
        # Schema v1: 15 dynamic globals followed by body/joint/action counts, then total mass.
        global_keep[18] = 0
        self.register_buffer("static_keep", static_keep)
        self.register_buffer("global_keep", global_keep)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        actor_obs = obs.clone(recurse=False)
        actor_obs["static"] = obs["static"] * self.static_keep
        actor_obs["global"] = obs["global"] * self.global_keep
        return super().forward(actor_obs, masks, hidden_state, stochastic_output)


def mechanical_control_cfg(env, condition, routing="attention", seed=42):
    """Change only the actor in the frozen mechanical pilot's PPO/critic contract."""
    cfg = mechanical_train_cfg(env, routing, seed)
    prefix = "isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls"
    if condition == "specialists":
        cfg["actor"] = {
            "class_name": f"{prefix}:MechanicalSpecialists",
            "observation_widths": [env.graphs[name].flat_width for name in env.agents],
            "action_counts": [env.graphs[name].num_joints for name in env.agents],
        }
        cfg["obs_groups"]["actor"] = ["critic", "embodiment", "action_mask"]
    elif condition == "no_inertia":
        cfg["actor"]["class_name"] = f"{prefix}:MechanicalWithoutInertia"
    else:
        raise ValueError("Choose specialists or no_inertia as the control condition.")
    return cfg
