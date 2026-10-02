# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IPPO integration for independent Same-Step autoresetting robot trajectories."""

from __future__ import annotations

import torch
from skrl.multi_agents.torch.ippo import IPPO
from skrl.utils.runner.torch import Runner


class IndependentIPPO(IPPO):
    """Bootstrap time limits from terminal observations, keeping rollout episodes separate.

    Upstream IPPO uses ``next_observations`` for timeout values. Isaac Lab's Same-Step autoreset
    instead returns the new episode there. Substitute the captured final observations only for
    transition recording; the trainer continues acting on the reset observations.
    """

    def record_transition(self, *, next_observations, terminated, truncated, infos, **kwargs):
        next_observations = dict(next_observations)
        truncated = {agent: value & ~terminated[agent] for agent, value in truncated.items()}
        for agent, timeout in truncated.items():
            if timeout.any():
                final_obs = infos[agent].get("final_obs")
                if final_obs is None:
                    raise ValueError("Independent IPPO timeouts require compute_final_obs=True.")
                next_observations[agent] = torch.where(timeout, final_obs, next_observations[agent])
        super().record_transition(
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            **kwargs,
        )


class IndependentRunner(Runner):
    """Use ordinary IPPO models/optimization with correct independent timeout handling."""

    def __init__(self, env, cfg):
        if cfg["agent"]["class"].lower() != "ippo":
            raise ValueError("Independent robot training currently requires IPPO.")
        if cfg["agent"].get("time_limit_bootstrap") is not True:
            raise ValueError("Independent IPPO requires time_limit_bootstrap=True to cut GAE at resets.")
        if any(space is not None for space in env.state_spaces.values()):
            raise ValueError("Independent IPPO requires per-robot observations and no centralized state.")
        super().__init__(env, cfg)

    def _component(self, name):
        if name.lower() == "ippo":
            return IndependentIPPO
        return super()._component(name)
