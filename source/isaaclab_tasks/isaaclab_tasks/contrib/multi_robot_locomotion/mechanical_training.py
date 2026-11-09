# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanics observations using the existing independent-reset PPO trajectory contract."""

import torch
import torch.nn.functional as F

from isaaclab.utils import to_dict

from .agents.rsl_rl_ppo_cfg import IndependentPPORunnerCfg
from .graph_training import GraphVecEnv
from .mechanical_observations import MechanicalAntGraph


class MechanicalVecEnv(GraphVecEnv):
    """Batch unequal token and actuator counts; retain pre-reset terminal body state."""

    def __init__(self, env):
        self.env, self.cfg, self.device = env, env.cfg, env.device
        if not self.cfg.independent_resets or not self.cfg.compute_final_obs or self.cfg.is_finite_horizon:
            raise ValueError("Mechanical PPO requires independent resets and infinite-horizon final observations.")
        self.agents = env.possible_agents
        self.graphs = {agent: MechanicalAntGraph(env, agent) for agent in self.agents}
        self.num_actions = max(graph.num_joints for graph in self.graphs.values())
        self.node_width = max(graph.num_nodes for graph in self.graphs.values())
        self.critic_width = max(graph.flat_width for graph in self.graphs.values())
        self.num_envs = env.num_envs * len(self.agents)
        self.observations = self.encode(env.reset()[0])
        self.last_infos = {}

    def encode(self, observations):
        batches = []
        for index, agent in enumerate(self.agents):
            flat = observations[agent]
            graph = self.graphs[agent].encode(flat, self.node_width, self.num_actions)
            graph["critic"] = F.pad(flat, (0, self.critic_width - flat.shape[-1]))
            graph["embodiment"] = torch.full((flat.shape[0],), index, device=flat.device, dtype=torch.long)
            batches.append(graph)
        return torch.cat(batches, dim=0)

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        extras["log"] = {
            f"{agent}/{key}": value
            for agent in self.agents
            for key, value in self.last_infos[agent].get("log", {}).items()
        }
        return obs, rewards, dones, extras


def mechanical_train_cfg(env, routing="attention", seed=42):
    """Keep rollout/PPO settings and critic class constant across the actor candidates."""
    cfg = to_dict(IndependentPPORunnerCfg())
    prefix = "isaaclab_tasks.contrib.multi_robot_locomotion"
    cfg.update(class_name="OnPolicyRunner", seed=seed)
    cfg["actor"] = {"class_name": f"{prefix}.mechanical_policy:MechanicalActor", "routing": routing}
    cfg["critic"] = {
        "class_name": f"{prefix}.mechanical_policy:MechanicalCritic",
        "num_critics": len(env.agents),
        "initialization_seed": seed,
    }
    cfg["algorithm"]["class_name"] = f"{prefix}.graph_training:GraphPPO"
    cfg["obs_groups"] = {
        "actor": ["global", "static", "dynamic", "edges", "node_mask", "action_nodes", "action_mask"],
        "critic": ["critic", "embodiment"],
    }
    return cfg
