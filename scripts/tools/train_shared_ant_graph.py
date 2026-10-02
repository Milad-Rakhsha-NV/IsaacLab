# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train/evaluate the experimental shared Ant graph PPO, separately from specialist baselines."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.entrypoints.common import set_hydra_args

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.multi_robot_locomotion.graph_training import GraphVecEnv
from isaaclab_tasks.utils import resolve_task_config


def main():
    """Run shared source PPO or evaluate frozen weights on a source/target morphology."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-Ant-Three-Four-Direct")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=Path, default=Path("logs/shared_ant_graph/source_seed42"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, help="Source actor initialization for target adaptation.")
    parser.add_argument("--mode", choices=("source", "adapters", "random-core", "scratch", "finetune"))
    parser.add_argument("--adapter-dim", type=int, default=8)
    parser.add_argument("--latent-ablation", choices=("none", "zero", "shuffle"), default="none")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    metadata = (checkpoint.get("infos") or {}) if checkpoint else {}
    if checkpoint and not metadata and (args.checkpoint.parent / "experiment.json").exists():
        metadata = json.loads((args.checkpoint.parent / "experiment.json").read_text())
        # Upstream periodic saves omit infos and store the zero-based update index.
        metadata["completed_updates"] = checkpoint["iter"] + 1
    saved_adapters = any(k.startswith("adapters.") for k in checkpoint["actor_state_dict"]) if checkpoint else False
    mode = args.mode or metadata.get("mode", "adapters" if saved_adapters else "source")
    adapter_dim = metadata.get("adapter_dim", args.adapter_dim) if saved_adapters else args.adapter_dim
    if args.num_envs < 1 or args.updates < 1 or args.steps < 1:
        parser.error("Environment count, updates and evaluation steps must be positive.")
    if args.evaluate and args.report is None:
        parser.error("Evaluation requires --report; omit --checkpoint for an untrained baseline.")
    if args.adapter_dim < 1:
        parser.error("Adapter width must be positive.")
    if args.latent_ablation != "none" and not args.evaluate:
        parser.error("Latent ablations are evaluation-only.")
    if not args.evaluate and mode == "source" and args.task != "IsaacContrib-Ant-Three-Four-Direct":
        parser.error("Source training is restricted to three-/four-legged Ants; target data must stay held out.")
    if not args.evaluate and mode != "source" and args.task != "IsaacContrib-Ant-Five-Specialist-Direct":
        parser.error("Target experiments require the five-legged task.")
    if args.source_checkpoint and (args.evaluate or args.checkpoint or mode in ("source", "scratch")):
        parser.error("Use --source-checkpoint only to initialize a new target transfer run.")
    if (
        not args.evaluate
        and not args.checkpoint
        and mode in ("adapters", "random-core", "finetune")
        and not args.source_checkpoint
    ):
        parser.error("Transfer initialization requires --source-checkpoint.")
    if args.checkpoint and args.mode and metadata.get("mode", "adapters" if saved_adapters else "source") != mode:
        parser.error("Checkpoint mode differs; use --source-checkpoint to initialize target learning.")
    if args.source_checkpoint:
        source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        source_metadata = source.get("infos") or json.loads(
            (args.source_checkpoint.parent / "experiment.json").read_text()
        )
        source_agents = source_metadata.get("training_agents", source_metadata.get("source_agents"))
        if source_metadata.get("target_training") is not False or set(source_agents or []) != {"ant_3", "ant_4"}:
            parser.error("Source initialization must be a documented three-/four-leg-only run.")
    set_hydra_args([])
    cfg, agent = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    torch.manual_seed(args.seed)
    train_cfg = to_dict(agent)
    module = "isaaclab_tasks.contrib.multi_robot_locomotion"
    train_cfg.update(class_name="OnPolicyRunner", seed=args.seed)
    train_cfg["actor"] = {"class_name": f"{module}.graph_policy:GraphModel"}
    if mode in ("adapters", "random-core"):
        train_cfg["actor"]["adapter_dim"] = adapter_dim
    train_cfg["critic"] = {"class_name": f"{module}.graph_policy:GraphModel"}
    train_cfg["algorithm"]["class_name"] = f"{module}.graph_training:GraphPPO"
    train_cfg["obs_groups"] = {name: ["global", "nodes", "adjacency", "mask"] for name in ("actor", "critic")}
    with launch_simulation(cfg):
        raw = gym.make(args.task, cfg=cfg)
        try:
            env = GraphVecEnv(raw.unwrapped)
            runner = OnPolicyRunner(
                env, train_cfg, log_dir=None if args.evaluate else str(args.run_dir), device=env.device
            )
            if args.checkpoint:
                runner.load(
                    str(args.checkpoint),
                    load_cfg={
                        "actor": True,
                        "critic": True,
                        "optimizer": not args.evaluate,
                        "iteration": not args.evaluate,
                    },
                )
                if not args.evaluate and "completed_updates" in metadata:
                    runner.current_learning_iteration = metadata["completed_updates"]
            actor = runner.alg.actor
            if args.source_checkpoint:
                source = torch.load(args.source_checkpoint, map_location=env.device, weights_only=False)
                state = source["actor_state_dict"]
                if any(k.startswith("adapters.") for k in state):
                    raise ValueError("Target adapters cannot be used as source initialization.")
                if mode == "random-core":
                    # Match source input scaling and exploration; randomize only the learned actor maps.
                    state = {k: v for k, v in state.items() if k in dict(actor.named_buffers()) or k == "log_std"}
                    expected_missing = set(actor.state_dict()) - set(state)
                else:
                    expected_missing = {k for k in actor.state_dict() if k.startswith("adapters.")}
                missing, unexpected = actor.load_state_dict(state, strict=False)
                if set(missing) != expected_missing or unexpected:
                    raise ValueError(f"Incompatible source checkpoint: missing={missing}, unexpected={unexpected}")
            actor.latent_ablation = args.latent_ablation
            if not args.evaluate:
                args.run_dir.mkdir(parents=True, exist_ok=True)
                frozen = (
                    {
                        k: v.detach().cpu().clone()
                        for k, v in actor.state_dict().items()
                        if not k.startswith("adapters.")
                    }
                    if actor.adapters is not None
                    else {}
                )
                manifest = {
                    "task": args.task,
                    "seed": args.seed,
                    "worlds": args.num_envs,
                    "updates": args.updates,
                    "mode": mode,
                    "adapter_dim": adapter_dim if actor.adapters is not None else 0,
                    "architecture": "Shared graph encoder + 32-D decider latent + per-joint decoder",
                    "training_agents": env.agents,
                    "target_training": mode != "source",
                    "source_checkpoint": str(args.source_checkpoint)
                    if args.source_checkpoint
                    else metadata.get("source_checkpoint"),
                    "source_sha256": hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()
                    if args.source_checkpoint
                    else metadata.get("source_sha256"),
                    "resume_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "actor_parameters": sum(p.numel() for p in actor.parameters()),
                    "trainable_actor_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
                    "trainable_critic_parameters": sum(
                        p.numel() for p in runner.alg.critic.parameters() if p.requires_grad
                    ),
                    "critic_initialization": "resumed" if args.checkpoint else "fresh",
                    "transitions_per_update": env.num_envs * train_cfg["num_steps_per_env"],
                    "completed_updates": metadata.get("completed_updates", 0),
                }
                (args.run_dir / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if not args.checkpoint:
                    runner.save(str(args.run_dir / "initial.pt"), infos=manifest)
                runner.learn(args.updates)
                changed = [k for k, v in frozen.items() if not torch.equal(v, actor.state_dict()[k].cpu())]
                if changed:
                    raise RuntimeError(f"Frozen source actor changed during adaptation: {changed}")
                manifest["frozen_actor_verified"] = bool(frozen)
                manifest["completed_updates"] += args.updates
                runner.save(str(args.run_dir / "final.pt"), infos=manifest)
                (args.run_dir / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
                return
            policy = runner.get_inference_policy(env.device)
            obs = env.get_observations()
            totals = {
                a: {"episodes": 0, "return": 0.0, "duration_s": 0.0, "progress_m": 0.0, "survival": 0.0}
                for a in env.agents
            }
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    for agent, done in zip(env.agents, dones.split(raw.unwrapped.num_envs), strict=True):
                        count = int(done.sum())
                        if not count:
                            continue
                        totals[agent]["episodes"] += count
                        for key, name in (
                            ("return", "return"),
                            ("duration_s", "length_s"),
                            ("progress_m", "progress_m"),
                            ("survival", "survival"),
                        ):
                            totals[agent][key] += count * float(env.last_infos[agent]["log"][f"Episode/{name}"])
            for result in totals.values():
                if not result["episodes"]:
                    raise RuntimeError("No completed evaluation episodes; increase --steps.")
                for key in ("return", "duration_s", "progress_m", "survival"):
                    result[key] /= result["episodes"]
                result["progress_speed_m_s"] = result["progress_m"] / result["duration_s"]
            report = {
                "task": args.task,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "seed": args.seed,
                "worlds": args.num_envs,
                "metrics": totals,
                "mode": mode,
                "latent_ablation": args.latent_ablation,
                "target_updates": metadata.get("completed_updates") if mode != "source" else 0,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            raw.close()


if __name__ == "__main__":
    main()
