# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Original normalized residual dynamics, deterministic actor, and critic."""

import torch
from torch import nn

from .math_utils import QUAT, conjugate, features, integrate, multiply, rotate


def mlp(inputs, hidden, outputs):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, width, floor=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.zeros(()))
        self.floor = floor

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        mean, var = x.mean(0), x.var(0, unbiased=False)
        n = x.shape[0]
        weight = n / (self.count + n)
        delta = mean - self.mean
        self.variance.copy_((1 - weight) * self.variance + weight * var + weight * (1 - weight) * delta.square())
        self.mean.add_(weight * delta)
        self.count.add_(n)

    @property
    def std(self):
        return self.variance.clamp_min(self.floor**2).sqrt()

    def forward(self, x):
        return ((x - self.mean) / self.std).clamp(-20, 20)


def goal(state, future):
    """Proprioception-compatible targets: future joints/velocities and root orientations."""
    inv = conjugate(state[..., QUAT])[:, None].expand(-1, future.shape[1], -1)
    relative = multiply(inv, future[..., QUAT])
    x = torch.zeros_like(relative[..., :3])
    x[..., 0] = 1
    y = torch.zeros_like(x)
    y[..., 1] = 1
    return torch.cat((future[..., 13:], rotate(relative, x), rotate(relative, y)), -1).flatten(1)


class WorldModel(nn.Module):
    def __init__(self, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.inputs = Normalizer(98)
        self.outputs = Normalizer(70)
        self.net = mlp(98, hidden, 70)
        nn.init.normal_(self.net[-1].weight, std=0.0001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, action):
        raw = torch.cat((features(state), action), -1)
        delta = self.net(self.inputs(raw)) * self.outputs.std + self.outputs.mean
        return integrate(state, delta)


class Actor(nn.Module):
    def __init__(self, hidden=(2048, 1024, 512, 256), latent=32):
        super().__init__()
        self.encoder = mlp(640, hidden, latent)
        self.decoder = mlp(930 + latent, hidden, 29)
        self.obs_norm = Normalizer(93)
        self.goal_norm = Normalizer(640)
        nn.init.normal_(self.decoder[-1].weight, std=0.0001)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, history, target):
        z = torch.tanh(self.encoder(self.goal_norm(target)))
        return self.decoder(torch.cat((self.obs_norm(history).flatten(1), z), -1)).clamp(-10, 10)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalizer(709)
        self.net = mlp(709, (256, 128), 1)

    def inputs(self, state, target):
        return torch.cat((features(state), target), -1)

    def forward(self, inputs):
        # A symmetric bootstrap clamp limits critic exploitation of the model.
        # The negative bound is a stabilization choice, not a reward bound.
        return self.net(self.norm(inputs)).squeeze(-1).clamp(-12, 12)
