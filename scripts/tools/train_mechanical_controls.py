# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh specialist or inertial-descriptor controls under the mechanical pilot's PPO contract."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable
from train_mechanical_ant import evaluate

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls import mechanical_control_cfg
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import (
    ANT_SPLITS,
    MechanicalAntEnv,
    mechanical_ant_cfg,
)
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv


def main():
    """Train a new control, or evaluate it with frozen weights and normalization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("specialists", "no_inertia"))
    parser.add_argument("--routing", choices=("attention", "pooled"))
    parser.add_argument("--split", choices=ANT_SPLITS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total_envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument(
        "--initialization_checkpoint", type=Path, help="Untrained full-descriptor actor for a paired ablation."
    )
    parser.add_argument("--checkpoint", type=Path, help="Trained control checkpoint for evaluation only.")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    metadata = saved.get("infos") if saved else None
    if saved and not metadata:
        metadata = json.loads((args.checkpoint.parent / "experiment.json").read_text())
        metadata["completed_updates"] = saved["iter"] + 1
    metadata = metadata or {}
    condition = args.condition or metadata.get("condition", "specialists")
    routing = args.routing or metadata.get("routing", "attention")
    split = args.split or metadata.get("split", "canonical")
    seed = args.seed if args.seed is not None else metadata.get("seed", 42)
    if saved and (
        metadata.get("schema") != SCHEMA_VERSION
        or condition != metadata.get("condition")
        or routing != metadata.get("routing")
    ):
        parser.error("Checkpoint schema or control condition does not match.")
    if min(args.total_envs, args.updates, args.steps) < 1 or args.total_envs % len(ANT_SPLITS[split]):
        parser.error("Positive total_envs must divide equally across prototypes; updates/steps must be positive.")
    if args.evaluate:
        if not saved or not args.report or args.initialization_checkpoint:
            parser.error("Evaluation requires --checkpoint and --report, without initialization.")
        if condition == "specialists" and not set(ANT_SPLITS[split]) <= set(metadata["training_agents"]):
            parser.error("A specialist has no head for an unseen prototype; evaluate only its training bodies.")
    elif saved or split not in ("canonical", "diverse") or not args.run_dir:
        parser.error("Controls train fresh on canonical/diverse sources; supply --run_dir and no --checkpoint.")
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Training requires an empty/new run directory.")
    initialization = None
    if args.initialization_checkpoint:
        initialization = torch.load(args.initialization_checkpoint, map_location="cpu", weights_only=False)
        initial_info = initialization.get("infos") or {}
        if (
            condition != "no_inertia"
            or initial_info.get("completed_updates") != 0
            or initial_info.get("schema") != SCHEMA_VERSION
            or initial_info.get("routing") != routing
            or initial_info.get("split") != split
            or initial_info.get("seed") != seed
            or initial_info.get("total_envs") != args.total_envs
        ):
            parser.error(
                "Paired ablations require the full actor's untrained checkpoint with matching split, seed and batch."
            )
    torch.manual_seed(seed)
    cfg = mechanical_ant_cfg(split, args.total_envs // len(ANT_SPLITS[split]), seed)
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            if args.evaluate:
                actor_cfg = dict(metadata["training_config"]["actor"])
                actor_type = resolve_callable(actor_cfg.pop("class_name"))
                actor = actor_type(env.get_observations(), {}, "actor", env.num_actions, **actor_cfg).to(env.device)
                actor.load_state_dict(saved["actor_state_dict"])
                actor.eval()
                mapping = (
                    torch.tensor([metadata["training_agents"].index(name) for name in env.agents], device=env.device)
                    if condition == "specialists"
                    else None
                )

                def policy(obs):
                    if mapping is not None:
                        obs = obs.clone(recurse=False)
                        obs["embodiment"] = mapping[obs["embodiment"]]
                    return actor(obs)

                report = evaluate(env, policy, args.steps)
                report.update(
                    condition=condition,
                    routing=routing,
                    split=split,
                    seed=seed,
                    checkpoint=str(args.checkpoint),
                    training_agents=metadata["training_agents"],
                    completed_updates=metadata["completed_updates"],
                )
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return
            train = mechanical_control_cfg(env, condition, routing, seed)
            runner = OnPolicyRunner(env, train, log_dir=str(args.run_dir), device=env.device)
            if initialization:
                missing, unexpected = runner.alg.actor.load_state_dict(initialization["actor_state_dict"], strict=False)
                if set(missing) != {"static_keep", "global_keep"} or unexpected:
                    raise ValueError("Paired actor initialization is incompatible.")
                runner.alg.critic.load_state_dict(initialization["critic_state_dict"])
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
            for fields in descriptors.values():
                for tensor in fields.values():
                    fingerprint.update(tensor.numpy().tobytes())
            if initialization and fingerprint.hexdigest() != initial_info["descriptor_sha256"]:
                raise ValueError("Physical model differs from the paired full-descriptor run.")
            args.run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(descriptors, args.run_dir / "descriptors.pt")
            snapshot = args.run_dir / "source_snapshot"
            snapshot.mkdir()
            root = Path(__file__).resolve().parents[2]
            package = root / "source/isaaclab_tasks/isaaclab_tasks/contrib/multi_robot_locomotion"
            for path in (Path(__file__), Path(__file__).with_name("train_mechanical_ant.py"), *package.glob("*.py")):
                shutil.copy2(path, snapshot / path.name)
            manifest = {
                "schema": SCHEMA_VERSION,
                "condition": condition,
                "routing": routing,
                "split": split,
                "seed": seed,
                "training_agents": env.agents,
                "target_training": False,
                "total_envs": args.total_envs,
                "worlds": raw.num_envs,
                "training_config": train,
                "completed_updates": 0,
                "mechanics": {name: asdict(value) for name, value in cfg.mechanics.items()},
                "transitions_per_update": env.num_envs * train["num_steps_per_env"],
                "transitions_per_agent_per_update": raw.num_envs * train["num_steps_per_env"],
                "actor_parameters": sum(p.numel() for p in runner.alg.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in runner.alg.critic.parameters()),
                "descriptor_sha256": fingerprint.hexdigest(),
                "initialization_checkpoint": str(args.initialization_checkpoint) if initialization else None,
                "initialization_sha256": hashlib.sha256(args.initialization_checkpoint.read_bytes()).hexdigest()
                if initialization
                else None,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "newton", "mujoco-warp", "warp-lang", "rsl-rl-lib")
                },
            }
            manifest_path = args.run_dir / "experiment.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            runner.save(str(args.run_dir / "initial.pt"), infos=manifest)
            runner.learn(args.updates)
            manifest["completed_updates"] = args.updates
            runner.save(str(args.run_dir / "final.pt"), infos=manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            raw.close()


if __name__ == "__main__":
    main()
