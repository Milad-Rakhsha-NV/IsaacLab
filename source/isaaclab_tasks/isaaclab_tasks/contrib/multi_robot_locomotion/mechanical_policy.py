# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanics-conditioned actors and a common set of independent morphology critics."""

import torch
import torch.nn.functional as F
from rsl_rl.modules import EmpiricalNormalization
from torch import nn
from torch.distributions import Normal, kl_divergence

from .mechanical_observations import RELATION_COUNT, STATIC_SLICES


class FeedForwardModel(nn.Module):
    """RSL-RL stateless model lifecycle shared by the experimental actor and critic."""

    is_recurrent = False

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass


class MechanicalBlock(nn.Module):
    """Residual relation-aware attention or learned pooling, with identical feedforward widths."""

    def __init__(self, width, heads, routing="attention", local=False):
        super().__init__()
        self.heads, self.routing, self.local = heads, routing, local
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width) if routing == "attention" else nn.Linear(width, width + 1)
        self.relations = nn.Embedding(RELATION_COUNT, heads) if routing == "attention" else None
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, tokens, mask, edges):
        h = self.norm(tokens)
        if self.routing == "attention":
            batch, nodes, width = h.shape
            q, k, v = self.qkv(h).reshape(batch, nodes, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
            bias = self.relations(edges.long()).permute(0, 3, 1, 2)
            valid = mask[:, None, None, :]
            if self.local:
                allowed = (edges != 0) | torch.eye(nodes, dtype=torch.bool, device=h.device)
                # The task token is available everywhere; mechanical entities keep their typed edges.
                allowed[:, 0, :] = True
                allowed[:, :, 0] = True
                valid = valid & allowed[:, None]
            bias = bias.masked_fill(~valid, -torch.inf)
            context = (
                F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(batch, nodes, width)
            )
        else:
            score, value = self.qkv(h).split((1, h.shape[-1]), dim=-1)
            weights = score.squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
            context = torch.sum(weights.unsqueeze(-1) * value, dim=1, keepdim=True)
        tokens = tokens + self.output(context)
        return (tokens + self.ff(tokens)) * mask.unsqueeze(-1)


class MechanicalMessageBlock(nn.Module):
    """Typed neighbor messages plus a uniform global mean, without attention.

    Each relation has a learned scalar for each feature group. These weights depend only
    on the mechanical relation, never on a receiver/source query-key comparison. Neighbor
    sums are divided by the number of valid neighbors; global pooling uses a masked mean.
    Width and parameter count match MechanicalBlock's full-attention path.
    """

    def __init__(self, width, groups):
        super().__init__()
        self.groups = groups
        self.norm = nn.LayerNorm(width)
        self.message = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, width))
        self.relations = nn.Embedding(RELATION_COUNT, groups)
        self.global_message = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, tokens, mask, edges):
        batch, nodes, width = tokens.shape
        h = self.norm(tokens)
        # Match the local attention block's graph, including self and task-token links.
        neighbors = (edges != 0) | torch.eye(nodes, dtype=torch.bool, device=h.device)
        neighbors[:, 0, :] = True
        neighbors[:, :, 0] = True
        neighbors = neighbors & mask[:, None, :] & mask[:, :, None]
        weights = self.relations(edges.long()).permute(0, 3, 1, 2)
        weights = weights * neighbors[:, None] / neighbors.sum(-1).clamp_min(1)[:, None, :, None]
        values = self.message(h).reshape(batch, nodes, self.groups, width // self.groups).transpose(1, 2)
        local = (weights @ values).transpose(1, 2).reshape(batch, nodes, width)
        global_mean = (h * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
        tokens = tokens + self.output(local + self.global_message(global_mean))
        return (tokens + self.ff(tokens)) * mask.unsqueeze(-1)


class MechanicalActor(FeedForwardModel):
    """Variable-size actor; only explicitly mapped actuator DOF tokens produce actions.

    Attention/pooled routing shares a local mechanical-attention encoder followed by two
    full-attention or learned-pooling blocks. Message-passing routing replaces all three with
    typed graph messages and uniform global pooling, without attention. No robot ID or
    token-position embedding enters the actor. Pooling is not a URMA replica.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dim=64, heads=4, routing="attention"):
        super().__init__()
        if routing not in ("attention", "pooled", "message_passing") or hidden_dim % heads:
            raise ValueError("Choose attention/pooled/message_passing routing and a width divisible by head count.")
        self.global_norm = EmpiricalNormalization(15)
        self.dynamic_norm = EmpiricalNormalization(obs["dynamic"].shape[-1])
        scale = torch.ones(obs["static"].shape[-1])
        scale[STATIC_SLICES["body_inertia"]] = 1000.0
        scale[STATIC_SLICES["joint_armature"]] = 100.0
        self.register_buffer("static_scale", scale)
        self.encoder = nn.Sequential(
            nn.Linear(obs["static"].shape[-1] + obs["dynamic"].shape[-1], hidden_dim), nn.ELU()
        )
        self.global_encoder = nn.Linear(obs["global"].shape[-1], hidden_dim)
        if routing == "message_passing":
            self.local = MechanicalMessageBlock(hidden_dim, heads)
            self.blocks = nn.ModuleList([MechanicalMessageBlock(hidden_dim, heads) for _ in range(2)])
        else:
            self.local = MechanicalBlock(hidden_dim, heads, local=True)
            self.blocks = nn.ModuleList([MechanicalBlock(hidden_dim, heads, routing) for _ in range(2)])
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(()))
        self.distribution, self.mask = None, None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mask = obs["node_mask"].bool()
        static = obs["static"].masked_fill(~mask.unsqueeze(-1), 0) * self.static_scale
        # Fixed invertible compression preserves variation and finite Newton limit sentinels.
        static = static.sign() * torch.log1p(static.abs())
        dynamic = self.dynamic_norm(obs["dynamic"].masked_fill(~mask.unsqueeze(-1), 0))
        h = self.encoder(torch.cat((static, dynamic), dim=-1)) * mask.unsqueeze(-1)
        context = obs["global"][:, 15:]
        global_obs = torch.cat(
            (self.global_norm(obs["global"][:, :15]), context.sign() * torch.log1p(context.abs())), dim=-1
        )
        h = torch.cat((self.global_encoder(global_obs).unsqueeze(1), h), dim=1)
        mask = F.pad(mask, (1, 0), value=True)
        edges = F.pad(obs["edges"], (1, 0, 1, 0))
        h = self.local(h, mask, edges)
        for block in self.blocks:
            h = block(h, mask, edges)
        selected = h.gather(1, (obs["action_nodes"] + 1).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        mean = self.decoder(selected).squeeze(-1) * obs["action_mask"]
        if stochastic_output:
            self.mask = obs["action_mask"]
            self.distribution = Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean))
            return self.distribution.sample() * self.mask
        return mean

    def update_normalization(self, obs):
        self.global_norm.update(obs["global"][:, :15])
        self.dynamic_norm.update(obs["dynamic"][obs["node_mask"].bool()])

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


class MechanicalCritic(FeedForwardModel):
    """Independent [256,128,64] value functions, identical for both actor experiments.

    Only this training-time critic receives prototype indices. It consumes complete flat dynamic
    observations; fixed morphology need not be inferred by a shared value-function bottleneck.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, num_critics, initialization_seed=42):
        super().__init__()
        width = obs["critic"].shape[-1]
        self.norms = nn.ModuleList([EmpiricalNormalization(width) for _ in range(num_critics)])
        # Actor architectures consume different amounts of RNG during initialization.
        with torch.random.fork_rng():
            torch.manual_seed(initialization_seed)
            self.values = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(width, 256),
                        nn.ELU(),
                        nn.Linear(256, 128),
                        nn.ELU(),
                        nn.Linear(128, 64),
                        nn.ELU(),
                        nn.Linear(64, 1),
                    )
                    for _ in range(num_critics)
                ]
            )

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        output = obs["critic"].new_zeros((obs["critic"].shape[0], 1))
        for index, (norm, value) in enumerate(zip(self.norms, self.values, strict=True)):
            selected = obs["embodiment"] == index
            output[selected] = value(norm(obs["critic"][selected]))
        return output

    def update_normalization(self, obs):
        for index, norm in enumerate(self.norms):
            selected = obs["embodiment"] == index
            if selected.any():
                norm.update(obs["critic"][selected])
