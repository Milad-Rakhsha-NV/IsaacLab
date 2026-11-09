# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train or evaluate a Go2 MLP specialist under the frozen transfer experiment's effort task."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch
from evaluate_mechanical_transfer import rollout, transfer_cfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls import (
    MechanicalSpecialists,
    mechanical_control_cfg,
)
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import MechanicalAntEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_imitation import descriptor_roundoff
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv


def main() -> None:
    """Preserve the task, record target-training provenance, and support checkpoint-safe continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    if min(args.worlds, args.updates, args.steps) < 1:
        parser.error("Counts must be positive.")
    if args.evaluate:
        if not args.checkpoint or not args.report or args.report.exists():
            parser.error("Evaluation needs a checkpoint and a new report path.")
    elif not args.run_dir or (args.run_dir.exists() and any(args.run_dir.iterdir())):
        parser.error("Training needs a new or empty run directory.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "newton", "mujoco-warp", "warp-lang", "rsl-rl-lib")
    }
    if saved:
        info = saved["infos"]
        if (
            info["schema"] != SCHEMA_VERSION
            or info["versions"] != versions
            or info["training_agents"] != ["go2"]
            or info["condition"] != "go2_specialist"
        ):
            parser.error("Checkpoint schema, dependencies or specialist identity differ.")
        if not args.evaluate and (info["seed"] != args.seed or info["worlds"] != args.worlds):
            parser.error("Continuation must retain training seed and world count.")
    cfg = transfer_cfg("go2", args.worlds, args.seed)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            descriptors = {
                name: {key: getattr(graph, key).cpu() for key in ("static", "edges", "action_nodes", "context")}
                for name, graph in env.graphs.items()
            }
            fingerprint = hashlib.sha256()
            for fields in descriptors.values():
                for tensor in fields.values():
                    fingerprint.update(tensor.numpy().tobytes())
            if saved:
                reference = torch.load(args.checkpoint.parent / "descriptors.pt", weights_only=True)
                descriptor_roundoff(descriptors, reference)
            if args.evaluate:
                actor = MechanicalSpecialists(
                    env.get_observations(),
                    {},
                    "actor",
                    env.num_actions,
                    [env.graphs["go2"].flat_width],
                    [env.num_actions],
                ).to(env.device)
                actor.load_state_dict(saved["actor_state_dict"], strict=True)
                actor.eval()
                before = {key: value.clone() for key, value in actor.state_dict().items()}
                result = rollout(env, actor, None, args.steps)
                assert all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())
                report = {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "seed": args.seed,
                    "worlds": args.worlds,
                    "steps": args.steps,
                    "training_updates": info["completed_updates"],
                    "actor_and_normalization_unchanged": True,
                    "target_training": True,
                    "metrics": {"go2": result},
                    "descriptor_sha256": fingerprint.hexdigest(),
                    "versions": versions,
                }
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2), flush=True)
                return
            train = mechanical_control_cfg(env, "specialists", "attention", args.seed)
            train["save_interval"] = 100
            runner = OnPolicyRunner(env, train, log_dir=str(args.run_dir), device=env.device)
            completed = 0
            if saved:
                runner.load(str(args.checkpoint))
                completed = info["completed_updates"]
                runner.current_learning_iteration = completed
                if train != info["training_config"]:
                    raise ValueError("Continuation changes the PPO configuration.")
            args.run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(descriptors, args.run_dir / "descriptors.pt")
            root = Path(__file__).resolve().parents[2]
            package = root / "source/isaaclab_tasks/isaaclab_tasks/contrib/multi_robot_locomotion"
            sources = [
                Path(__file__),
                Path(__file__).with_name("evaluate_mechanical_transfer.py"),
                *package.glob("*.py"),
            ]
            snapshot = args.run_dir / "source_snapshot"
            snapshot.mkdir()
            for path in sources:
                shutil.copy2(path, snapshot / path.name)
            manifest = {
                "condition": "go2_specialist",
                "schema": SCHEMA_VERSION,
                "versions": versions,
                "training_agents": ["go2"],
                "target_training": True,
                "seed": args.seed,
                "worlds": args.worlds,
                "completed_updates": completed,
                "requested_additional_updates": args.updates,
                "training_config": train,
                "environment_config": to_dict(cfg),
                "descriptor_sha256": fingerprint.hexdigest(),
                "transitions_per_update": env.num_envs * train["num_steps_per_env"],
                "actor_parameters": sum(p.numel() for p in runner.alg.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in runner.alg.critic.parameters()),
                "resume_checkpoint": str(args.checkpoint.resolve()) if saved else None,
                "source_hashes": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
                "task_note": "Same ideal-effort Go2 task as frozen transfer; 480 Hz physics, 60 Hz actions.",
                "status": "running",
            }
            manifest_path = args.run_dir / "experiment.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            runner.save(str(args.run_dir / "initial.pt"), infos=manifest)
            try:
                runner.learn(args.updates)
                manifest.update(completed_updates=completed + args.updates, status="complete")
                runner.current_learning_iteration = manifest["completed_updates"]
                if not all(torch.isfinite(v).all() for v in runner.alg.actor.state_dict().values()):
                    raise RuntimeError("Non-finite specialist state after training.")
                runner.save(str(args.run_dir / "final.pt"), infos=manifest)
            except Exception as error:
                manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        finally:
            raw.close()


if __name__ == "__main__":
    main()
