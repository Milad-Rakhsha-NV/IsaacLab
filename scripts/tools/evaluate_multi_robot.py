# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate every independent robot policy and write per-robot learning evidence.

Example:
    uv run --extra skrl python scripts/tools/evaluate_multi_robot.py \
        --task IsaacContrib-Multi-Ant-Direct --checkpoint PATH --report logs/multi_robot_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_rl.entrypoints.common import set_hydra_args

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


def main():
    """Evaluate a saved checkpoint or the same architecture before training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-Multi-Ant-Direct")
    parser.add_argument("--rl_library", choices=("skrl", "rsl_rl"), default="skrl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive")

    set_hydra_args([])
    env_cfg, agent_cfg = resolve_task_config(args.task, f"{args.rl_library}_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.rl_library == "skrl":
        agent_cfg["seed"] = args.seed
        agent_cfg["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
    else:
        agent_cfg.seed = args.seed

    import gymnasium as gym

    with launch_simulation(env_cfg):
        raw = gym.make(args.task, cfg=env_cfg)
        try:
            if args.rl_library == "skrl":
                from isaaclab_rl.skrl import SkrlVecEnvWrapper, import_skrl_runner

                env = SkrlVecEnvWrapper(raw)
                runner = import_skrl_runner("torch", independent_agents=True)(env, agent_cfg)
                if args.checkpoint:
                    runner.agent.load(str(args.checkpoint))
                runner.agent.enable_training_mode(False, apply_to_models=True)
            else:
                from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
                from isaaclab_rl.rsl_rl.independent import RslRlIndependentVecEnvWrapper

                env = RslRlIndependentVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
                runner = create_rsl_rl_runner(env, agent_cfg)
                if args.checkpoint:
                    runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(env.device)
            observations, _ = env.reset()
            totals = {
                agent: {"episodes": 0, "return": 0.0, "duration_s": 0.0, "progress_m": 0.0, "survival": 0.0}
                for agent in env.possible_agents
            }
            with torch.inference_mode():
                for step in range(args.steps):
                    if args.rl_library == "skrl":
                        sampled, outputs = runner.agent.act(
                            observations, env.state(), timestep=step, timesteps=args.steps
                        )
                        actions = {
                            agent: outputs[agent].get("mean_actions", sampled[agent]) for agent in env.possible_agents
                        }
                        observations, _, terminated, truncated, infos = env.step(actions)
                        dones = {agent: terminated[agent] | truncated[agent] for agent in env.possible_agents}
                    else:
                        observations, _, dones, infos = env.step(policy(observations))
                        policy.reset(dones)
                    for agent, result in totals.items():
                        count = int(dones[agent].sum())
                        if count:
                            metrics = infos[agent]["log"]
                            result["episodes"] += count
                            for key, log_key in (
                                ("return", "Episode/return"),
                                ("duration_s", "Episode/length_s"),
                                ("progress_m", "Episode/progress_m"),
                                ("survival", "Episode/survival"),
                            ):
                                result[key] += count * float(metrics[log_key])
            for agent, result in totals.items():
                count = result["episodes"]
                if not count:
                    raise RuntimeError(f"No completed episodes for {agent}; increase --steps.")
                for key in ("return", "duration_s", "progress_m", "survival"):
                    result[key] /= count
                result["progress_speed_m_s"] = result["progress_m"] / result["duration_s"]
            report = {
                "task": args.task,
                "rl_library": args.rl_library,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "seed": args.seed,
                "worlds": args.num_envs,
                "steps": args.steps,
                "deterministic_actions": True,
                "metrics": totals,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            raw.close()


if __name__ == "__main__":
    main()
