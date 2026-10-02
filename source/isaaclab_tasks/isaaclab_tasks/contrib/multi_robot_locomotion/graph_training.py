# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental shared RSL-RL PPO adapter for independent Ant trajectories."""

import torch
from rsl_rl.algorithms import PPO

from .graph_observations import AntJointGraph


class GraphVecEnv:
    """Stack independent robot trajectories; each shared-simulation step still occurs once.

    Each source contributes exactly N trajectories per rollout. Graph padding is local to this
    environment batch; a target-only environment can use a different width with the same weights.
    """

    def __init__(self, env):
        self.env = env
        self.cfg, self.device = env.cfg, env.device
        if not self.cfg.independent_resets or not self.cfg.compute_final_obs or self.cfg.is_finite_horizon:
            raise ValueError("Graph PPO requires independent, infinite-horizon episodes with final observations.")
        self.agents = env.possible_agents
        self.graphs = {agent: AntJointGraph(env, agent) for agent in self.agents}
        self.num_actions = max(graph.num_joints for graph in self.graphs.values())
        self.num_envs = env.num_envs * len(self.agents)
        self.observations = self.encode(env.reset()[0])
        self.last_infos = {}

    def encode(self, observations):
        return torch.cat(
            [self.graphs[agent].encode(observations[agent], self.num_actions) for agent in self.agents], dim=0
        )

    def get_observations(self):
        return self.observations

    def step(self, actions):
        action_dict = {
            agent: action[:, : self.graphs[agent].num_joints]
            for agent, action in zip(self.agents, actions.split(self.env.num_envs), strict=True)
        }
        obs, rewards, terminated, truncated, infos = self.env.step(action_dict)
        self.last_infos = infos
        self.observations = self.encode(obs)
        dones = torch.cat([terminated[a] | truncated[a] for a in self.agents])
        timeouts = torch.cat([truncated[a] & ~terminated[a] for a in self.agents])
        final = {a: infos[a].get("final_obs", obs[a]) for a in self.agents}
        extras = {"terminal_graph": self.encode(final), "terminal_timeout": timeouts}
        return self.observations, torch.cat([rewards[a] for a in self.agents]), dones, extras


class GraphPPO(PPO):
    """Use the final graph's critic value once for timeouts, including per-robot resets."""

    def process_env_step(self, obs, rewards, dones, extras):
        timeout = extras["terminal_timeout"]
        if timeout.any():
            rewards = rewards + self.gamma * self.critic(extras["terminal_graph"]).squeeze(-1) * timeout
        super().process_env_step(obs, rewards, dones, {})
