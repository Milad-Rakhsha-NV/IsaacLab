# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare mechanical-token routing and source diversity at a fixed total PPO batch size."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import (
    ANT_SPLITS,
    MechanicalAntEnv,
    mechanical_ant_cfg,
)
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION, STATIC_FIELDS
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_policy import MechanicalActor
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv, mechanical_train_cfg


def main():
    """Train source prototypes or evaluate a checkpoint without updating weights/statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=ANT_SPLITS)
    parser.add_argument("--routing", choices=("attention", "pooled", "message_passing"))
    parser.add_argument(
        "--total_envs", type=int, default=1024, help="Total trajectories, divided equally among prototypes."
    )
    parser.add_argument("--updates", type=int, default=500, help="Additional updates when resuming.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, weights_only=False, map_location="cpu") if args.checkpoint else None
    metadata = saved.get("infos") if saved else None
    if saved and not metadata:
        metadata = json.loads((args.checkpoint.parent / "experiment.json").read_text())
        metadata["completed_updates"] = saved["iter"] + 1
    metadata = metadata or {}
    args.split = args.split or metadata.get("split", "canonical")
    args.routing = args.routing or metadata.get("routing", "attention")
    args.seed = args.seed if args.seed is not None else metadata.get("seed", 42)
    if saved and (metadata.get("schema") != SCHEMA_VERSION or args.routing != metadata["routing"]):
        parser.error("Checkpoint descriptor schema or actor routing is incompatible.")
    if args.total_envs < 1 or args.total_envs % len(ANT_SPLITS[args.split]) or args.updates < 1 or args.steps < 1:
        parser.error("Positive total_envs must be divisible by the prototype count; updates/steps must be positive.")
    if args.evaluate:
        if not saved or not args.report:
            parser.error("Evaluation requires --checkpoint and --report.")
    else:
        if args.split not in ("canonical", "diverse"):
            parser.error("Development/test prototypes cannot enter source training.")
        if saved and (args.split != metadata["split"] or args.total_envs != metadata["total_envs"]):
            parser.error("Resume must preserve the source split and total batch size.")
        if saved and args.seed != metadata["seed"]:
            parser.error("Resume must preserve the training seed; use a new run for independent training seeds.")
        if not args.run_dir or (args.run_dir.exists() and any(args.run_dir.iterdir())):
            parser.error("Training requires an empty/new --run_dir to preserve previous artifacts.")
    worlds = args.total_envs // len(ANT_SPLITS[args.split])
    cfg = mechanical_ant_cfg(args.split, worlds, args.seed)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            if args.evaluate:
                actor = MechanicalActor(env.get_observations(), {}, "actor", env.num_actions, routing=args.routing).to(
                    env.device
                )
                actor.load_state_dict(saved["actor_state_dict"])
                actor.eval()
                report = evaluate(env, actor, args.steps)
                report.update(
                    checkpoint=str(args.checkpoint),
                    seed=args.seed,
                    split=args.split,
                    source_split=metadata["split"],
                    completed_updates=metadata["completed_updates"],
                    total_envs=args.total_envs,
                    schema=SCHEMA_VERSION,
                )
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return
            train_cfg = mechanical_train_cfg(env, args.routing, args.seed)
            runner = OnPolicyRunner(env, train_cfg, log_dir=str(args.run_dir), device=env.device)
            if saved:
                runner.load(str(args.checkpoint))
                runner.current_learning_iteration = metadata["completed_updates"]
            descriptors = {
                name: {
                    "static": graph.static.cpu(),
                    "edges": graph.edges.cpu(),
                    "action_nodes": graph.action_nodes.cpu(),
                    "context": graph.context.cpu(),
                }
                for name, graph in env.graphs.items()
            }
            fingerprint = hashlib.sha256()
            for tensors in descriptors.values():
                for tensor in tensors.values():
                    fingerprint.update(tensor.numpy().tobytes())
            descriptor_hash = fingerprint.hexdigest()
            if saved and descriptor_hash != metadata["descriptor_sha256"]:
                raise ValueError("Finalized Newton mechanics changed since the checkpoint; start a new experiment.")
            args.run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(descriptors, args.run_dir / "descriptors.pt")
            source = args.run_dir / "source_snapshot"
            source.mkdir()
            package = (
                Path(__file__).resolve().parents[2]
                / "source/isaaclab_tasks/isaaclab_tasks/contrib/multi_robot_locomotion"
            )
            for path in (Path(__file__), *package.glob("*.py")):
                shutil.copy2(path, source / path.name)
            manifest = {
                "schema": SCHEMA_VERSION,
                "static_fields": STATIC_FIELDS,
                "split": args.split,
                "routing": args.routing,
                "seed": args.seed,
                "total_envs": args.total_envs,
                "worlds": worlds,
                "training_agents": env.agents,
                "target_training": False,
                "mechanics": {name: asdict(value) for name, value in cfg.mechanics.items()},
                "all_splits": ANT_SPLITS,
                "training_config": train_cfg,
                "transitions_per_update": args.total_envs * train_cfg["num_steps_per_env"],
                "transitions_per_agent_per_update": worlds * train_cfg["num_steps_per_env"],
                "actor_parameters": sum(p.numel() for p in runner.alg.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in runner.alg.critic.parameters()),
                "tokens": {name: graph.num_nodes for name, graph in env.graphs.items()},
                "descriptor_sha256": descriptor_hash,
                "resume_checkpoint": str(args.checkpoint) if saved else None,
                "completed_updates": metadata.get("completed_updates", 0),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "newton", "mujoco-warp", "warp-lang", "rsl-rl-lib")
                },
            }
            manifest_path = args.run_dir / "experiment.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            runner.save(str(args.run_dir / "initial.pt"), infos=manifest)
            runner.learn(args.updates)
            manifest["completed_updates"] += args.updates
            runner.save(str(args.run_dir / "final.pt"), infos=manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            raw.close()


def evaluate(env, actor, steps):
    """Aggregate completed episodes without confusing reset seeds with independent training runs."""
    totals = {
        name: {"episodes": 0, "return": 0.0, "length_s": 0.0, "progress_m": 0.0, "survival": 0.0} for name in env.agents
    }
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, _ = env.step(actor(obs))
            for name, done in zip(env.agents, dones.split(env.env.num_envs), strict=True):
                count = int(done.sum())
                if count:
                    totals[name]["episodes"] += count
                    for key in ("return", "length_s", "progress_m", "survival"):
                        totals[name][key] += count * float(env.last_infos[name]["log"][f"Episode/{key}"])
    for result in totals.values():
        if not result["episodes"]:
            raise RuntimeError("No completed episodes; increase --steps.")
        for key in ("return", "length_s", "progress_m", "survival"):
            result[key] /= result["episodes"]
        result["progress_speed_m_s"] = result["progress_m"] / result["length_s"]
    return {"metrics": totals}


if __name__ == "__main__":
    main()
