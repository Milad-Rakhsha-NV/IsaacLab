# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental shared graph encoder, fixed latent decision core and per-joint action decoder.

Parameter shapes do not depend on joint count. Joint identities enter through physical features
and connectivity, never node indices. Padding is excluded from messages, pooling and PPO densities.
"""

import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn
from torch.distributions import Normal, kl_divergence


class GraphModel(nn.Module):
    """Feedforward graph actor or critic implementing the RSL-RL model protocol."""

    is_recurrent = False

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dim=64, latent_dim=32, adapter_dim=0):
        super().__init__()
        self.is_actor = obs_set == "actor"
        self.global_norm = EmpiricalNormalization(obs["global"].shape[-1])
        self.node_norm = EmpiricalNormalization(obs["nodes"].shape[-1])
        self.encoder = nn.Sequential(nn.Linear(obs["nodes"].shape[-1], hidden_dim), nn.ELU())
        self.messages = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU()) for _ in range(2)]
        )
        self.decider = nn.Sequential(
            nn.Linear(hidden_dim + obs["global"].shape[-1], hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + (hidden_dim if self.is_actor else 0), hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )
        if self.is_actor:
            # One shared exploration scale, not a fixed vector indexed by robot joints.
            self.log_std = nn.Parameter(torch.zeros(()))
        self.adapters = None
        self.latent_ablation = "none"
        if adapter_dim:
            if not self.is_actor or adapter_dim < 1:
                raise ValueError("Positive-width target adapters are supported only on the actor.")
            # Freeze the entire source actor, including exploration. Normalization is frozen below.
            self.requires_grad_(False)
            self.adapters = nn.ModuleDict()
            for name, inputs, outputs in (
                ("nodes", obs["nodes"].shape[-1], obs["nodes"].shape[-1]),
                ("global", obs["global"].shape[-1], obs["global"].shape[-1]),
                ("actions", hidden_dim + latent_dim, 1),
            ):
                self.adapters[name] = nn.Sequential(
                    nn.Linear(inputs, adapter_dim), nn.ELU(), nn.Linear(adapter_dim, outputs)
                )
                nn.init.zeros_(self.adapters[name][-1].weight)
                nn.init.zeros_(self.adapters[name][-1].bias)
        self.distribution = None
        self.mask = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mask = obs["mask"].bool()
        nodes = self.node_norm(obs["nodes"])
        global_obs = self.global_norm(obs["global"])
        if self.adapters is not None:
            nodes = nodes + self.adapters["nodes"](nodes)
            global_obs = global_obs + self.adapters["global"](global_obs)
        h = self.encoder(nodes) * mask.unsqueeze(-1)
        adjacency = obs["adjacency"] * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
        for layer in self.messages:
            h = layer(torch.cat((h, torch.bmm(adjacency, h)), dim=-1)) * mask.unsqueeze(-1)
        pooled = h.sum(1) / mask.sum(-1, keepdim=True).clamp_min(1)
        latent = self.decider(torch.cat((pooled, global_obs), dim=-1))
        if not self.is_actor:
            return self.decoder(latent)
        if self.latent_ablation == "zero":
            latent = torch.zeros_like(latent)
        elif self.latent_ablation == "shuffle":
            latent = latent[torch.randperm(latent.shape[0], device=latent.device)]
        context = torch.cat((h, latent.unsqueeze(1).expand(-1, h.shape[1], -1)), dim=-1)
        mean = self.decoder(context)
        if self.adapters is not None:
            mean = mean + self.adapters["actions"](context)
        mean = mean.squeeze(-1)
        mean = mean * mask
        if stochastic_output:
            self.mask = mask
            self.distribution = Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean))
            return self.distribution.sample() * mask
        return mean

    def update_normalization(self, obs):
        if self.adapters is not None:
            return
        self.global_norm.update(obs["global"])
        self.node_norm.update(obs["nodes"][obs["mask"].bool()])

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

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
