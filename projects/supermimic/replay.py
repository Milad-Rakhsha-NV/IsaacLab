# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""GPU ring replay storing terminal successors and episode-local history."""

import torch


class Replay:
    def __init__(self, capacity, num_envs, device):
        self.capacity, self.num_envs = capacity, num_envs
        self.device, self.cursor, self.size = device, 0, 0
        self.state = torch.zeros(capacity, num_envs, 71, device=device)
        self.successor = torch.zeros_like(self.state)
        self.action = torch.zeros(capacity, num_envs, 29, device=device)
        self.previous = torch.zeros_like(self.action)
        self.episode = torch.full((capacity, num_envs), -1, dtype=torch.long, device=device)
        self.clip = torch.zeros_like(self.episode)
        self.frame = torch.zeros_like(self.episode)

    @torch.no_grad()
    def add(self, state, successor, action, previous, episode, clip, frame):
        i = self.cursor
        for target, value in (
            (self.state, state),
            (self.successor, successor),
            (self.action, action),
            (self.previous, previous),
            (self.episode, episode),
            (self.clip, clip),
            (self.frame, frame),
        ):
            target[i].copy_(value)
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def windows(self, batch, horizon):
        if self.size < horizon:
            raise ValueError("Replay is shorter than the model fitting horizon")
        times, envs = [], []
        remaining = batch
        for _ in range(20):
            env = torch.randint(self.num_envs, (remaining * 4,), device=self.device)
            age = torch.randint(horizon - 1, self.size, env.shape, device=self.device)
            indices = (self.cursor - 1 - age[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
            episodes = self.episode[indices, env[:, None]]
            valid = (episodes == episodes[:, :1]).all(-1) & (episodes[:, 0] >= 0)
            selected = valid.nonzero().flatten()[:remaining]
            times.append(indices[selected])
            envs.append(env[selected])
            remaining -= len(selected)
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("Not enough episode-local replay windows; inspect early terminations")
        t, e = torch.cat(times), torch.cat(envs)
        return self.state[t[:, 0], e], self.action[t, e[:, None]], self.successor[t, e[:, None]]

    def starts(self, batch, history):
        age = torch.randint(self.size, (batch,), device=self.device)
        env = torch.randint(self.num_envs, (batch,), device=self.device)
        t = (self.cursor - 1 - age) % self.capacity
        offsets = torch.arange(history - 1, -1, -1, device=self.device)
        h = (t[:, None] - offsets) % self.capacity
        valid = (self.episode[h, env[:, None]] == self.episode[t, env][:, None]) & (age[:, None] + offsets < self.size)
        first = valid.long().argmax(-1)
        h = torch.where(valid, h, h.gather(1, first[:, None]))
        return {
            "state": self.state[t, env],
            "history_states": self.state[h, env[:, None]],
            "history_actions": self.previous[h, env[:, None]],
            "previous": self.previous[t, env],
            "clip": self.clip[t, env],
            "frame": self.frame[t, env],
        }
