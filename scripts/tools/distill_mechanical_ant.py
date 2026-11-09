# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect source-specialist demonstrations, fit mechanical actors, and test their closed-loop control."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from train_mechanical_ant import evaluate

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls import MechanicalSpecialists
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import MechanicalAntEnv, mechanical_ant_cfg
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_imitation import balanced_action_mse, descriptor_roundoff
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_policy import MechanicalActor
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv

DATA_SCHEMA = "mechanical-ant-demonstrations-v1"


def descriptor_hash(env: MechanicalVecEnv) -> str:
    """Fingerprint finalized physics in the same order as the teacher's training manifest."""
    fingerprint = hashlib.sha256()
    for graph in env.graphs.values():
        for value in (graph.static, graph.edges, graph.action_nodes, graph.context):
            fingerprint.update(value.cpu().numpy().tobytes())
    return fingerprint.hexdigest()


def make_actor(env: MechanicalVecEnv, student: str) -> torch.nn.Module:
    """Construct the unchanged policy families, with no teacher weights transferred."""
    if student == "mlp":
        actor = MechanicalSpecialists(
            env.get_observations(),
            {},
            "actor",
            env.num_actions,
            [env.graphs[name].flat_width for name in env.agents],
            [env.graphs[name].num_joints for name in env.agents],
        )
    else:
        actor = MechanicalActor(env.get_observations(), {}, "actor", env.num_actions, routing=student)
    return actor.to(env.device)


def encode_rows(env: MechanicalVecEnv, name: str, flat: torch.Tensor):
    """Use the live actor's encoding for stored, unnormalized student-available state."""
    obs = env.graphs[name].encode(flat, env.node_width, env.num_actions)
    obs["critic"] = F.pad(flat, (0, env.critic_width - flat.shape[-1]))
    obs["embodiment"] = torch.full((len(flat),), env.agents.index(name), device=flat.device, dtype=torch.long)
    return obs


def collect(env: MechanicalVecEnv, teacher: torch.nn.Module, args, metadata: dict) -> None:
    """Capture pre-action state/means, commanded effort, and episode identities without teacher updates."""
    records = {
        name: {key: [] for key in ("flat", "action", "effort_command", "trajectory", "step")} for name in env.agents
    }
    episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    worlds = torch.arange(env.env.num_envs, device=env.device)
    totals = {
        name: {key: 0.0 for key in ("episodes", "return", "length_s", "progress_m", "survival")} for name in env.agents
    }
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args.steps):
            action = teacher(obs)
            if step % args.stride == 0:
                for index, name in enumerate(env.agents):
                    rows = slice(index * env.env.num_envs, (index + 1) * env.env.num_envs)
                    graph = env.graphs[name]
                    mean = action[rows, : graph.num_joints]
                    gain = env.env.joint_gears[name] * env.cfg.robots[name].task.action_scale
                    values = {
                        "flat": obs["critic"][rows, : graph.flat_width],
                        "action": mean,
                        "effort_command": mean.clamp(-1, 1) * gain,
                        "trajectory": torch.stack((worlds, episodes[rows]), -1),
                        "step": torch.full_like(worlds, step),
                    }
                    for key, value in values.items():
                        records[name][key].append(value.cpu().clone())
            obs, _, done, _ = env.step(action)
            episodes += done.long()
            for name, finished in zip(env.agents, done.split(env.env.num_envs), strict=True):
                count = int(finished.sum())
                if count:
                    totals[name]["episodes"] += count
                    for key in ("return", "length_s", "progress_m", "survival"):
                        totals[name][key] += count * float(env.last_infos[name]["log"][f"Episode/{key}"])
    data = {name: {key: torch.cat(values) for key, values in fields.items()} for name, fields in records.items()}
    for name, fields in data.items():
        fields["effort_gain"] = (env.env.joint_gears[name] * env.cfg.robots[name].task.action_scale).cpu()
    for result in totals.values():
        if result["episodes"]:
            for key in ("return", "length_s", "progress_m", "survival"):
                result[key] /= result["episodes"]
            result["progress_speed_m_s"] = result["progress_m"] / result["length_s"]
    info = dict(
        metadata,
        seed=args.seed,
        steps=args.steps,
        stride=args.stride,
        worlds=env.env.num_envs,
        simulator_transitions=args.steps * env.num_envs,
        samples_per_body={name: len(fields["flat"]) for name, fields in data.items()},
        metrics=totals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": info, "data": data}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


def load_data(args, expected: dict, device: str) -> dict:
    """Reject teacher/model mismatches and whole-collection seed leakage before fitting."""
    result, used_seeds = {}, set()
    for split in ("train", "validation", "test"):
        parts = []
        for path in getattr(args, f"{split}_data"):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            meta = saved["metadata"]
            for key in ("data_schema", "schema", "teacher_sha256", "descriptor_sha256", "versions"):
                if meta[key] != expected[key]:
                    raise ValueError(f"Demonstration {path} has incompatible {key}.")
            if meta["seed"] in used_seeds:
                raise ValueError("Collection seeds must be distinct within and across train/validation/test splits.")
            used_seeds.add(meta["seed"])
            for name in args.bodies:
                fields = saved["data"][name]
                if not torch.isfinite(fields["flat"]).all() or not torch.isfinite(fields["action"]).all():
                    raise ValueError("Demonstrations contain non-finite state or actions.")
                torch.testing.assert_close(
                    fields["effort_command"], fields["action"].clamp(-1, 1) * fields["effort_gain"]
                )
            parts.append(saved["data"])
        result[split] = {
            name: {key: torch.cat([part[name][key] for part in parts]).to(device) for key in ("flat", "action")}
            for name in args.bodies
        }
    return result


def action_metrics(env, actor, data: dict, train_variance: dict) -> dict:
    """Report raw/clipped actuator errors and error relative to training-action variation."""
    result = {}
    with torch.inference_mode():
        for name, fields in data.items():
            prediction = torch.cat(
                [
                    actor(encode_rows(env, name, flat))[:, : env.graphs[name].num_joints]
                    for flat in fields["flat"].split(1024)
                ]
            )
            target = fields["action"]
            per_joint = (prediction - target).square().mean(0)
            clipped = (prediction.clamp(-1, 1) - target.clamp(-1, 1)).square().mean(0)
            variance = float(train_variance[name].mean())
            result[name] = {
                "raw_mse": float(per_joint.mean()),
                "raw_rmse": float(per_joint.mean().sqrt()),
                "relative_rmse": math.sqrt(float(per_joint.mean()) / variance) if variance > 1e-12 else None,
                "clipped_rmse": float(clipped.mean().sqrt()),
                "worst_joint_clipped_rmse": float(clipped.max().sqrt()),
                "per_joint_raw_rmse": per_joint.sqrt().tolist(),
                "teacher_saturation_fraction": float((target.abs() >= 1).float().mean()),
                "student_saturation_fraction": float((prediction.abs() >= 1).float().mean()),
            }
    return result


def fit(env: MechanicalVecEnv, args, metadata: dict) -> None:
    """Fit balanced source batches; select only by demonstration validation, never by rollout speed."""
    data = load_data(args, metadata, env.device)
    torch.manual_seed(args.seed)
    actor = make_actor(env, args.student)
    actor.log_std.requires_grad_(False)
    if args.overfit_per_body:
        for fields in data["train"].values():
            indices = torch.randperm(len(fields["flat"]), device=env.device)[: args.overfit_per_body]
            for key in fields:
                fields[key] = fields[key][indices]
    # One pass over training state only; freeze these statistics for every fitting/evaluation forward.
    actor.train()
    with torch.no_grad():
        for name, fields in data["train"].items():
            for flat in fields["flat"].split(1024):
                actor.update_normalization(encode_rows(env, name, flat))
    actor.eval()
    train_variance = {name: fields["action"].var(0, unbiased=False) for name, fields in data["train"].items()}
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates, eta_min=args.learning_rate / 10)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source = args.run_dir / "source_snapshot"
    source.mkdir()
    package = (
        Path(__file__).resolve().parents[2] / "source/isaaclab_tasks/isaaclab_tasks/contrib/multi_robot_locomotion"
    )
    for path in (Path(__file__), *package.glob("*.py")):
        shutil.copy2(path, source / path.name)
    info = dict(
        metadata,
        student=args.student,
        routing=args.student,
        split=metadata.get("split", "canonical"),
        seed=args.seed,
        training_agents=args.bodies,
        target_training=metadata.get("target_training", False),
        completed_updates=0,
        imitation_steps=0,
        updates=args.updates,
        batch_per_body=args.batch_per_body,
        learning_rate=args.learning_rate,
        normalization="training demonstrations only; frozen before fitting",
        action_target="raw deterministic teacher means before clipping",
        selection="training MSE" if args.overfit_per_body else "disjoint demonstration validation MSE",
        actor_parameters=sum(p.numel() for p in actor.parameters()),
        overfit_per_body=args.overfit_per_body,
        train_variance={name: values.tolist() for name, values in train_variance.items()},
        dataset_files={
            split: [str(p.resolve()) for p in getattr(args, f"{split}_data")]
            for split in ("train", "validation", "test")
        },
        samples={split: {name: len(fields["flat"]) for name, fields in part.items()} for split, part in data.items()},
    )

    def save(path, step):
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "infos": dict(info, imitation_steps=step),
            },
            path,
        )

    save(args.run_dir / "initial.pt", 0)
    best, history, started = math.inf, [], time.monotonic()
    for step in range(1, args.updates + 1):
        observations, targets = [], []
        for name, fields in data["train"].items():
            indices = torch.randint(len(fields["flat"]), (args.batch_per_body,), device=env.device)
            observations.append(encode_rows(env, name, fields["flat"][indices]))
            targets.append(F.pad(fields["action"][indices], (0, env.num_actions - fields["action"].shape[-1])))
        obs, target = torch.cat(observations), torch.cat(targets)
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_action_mse(actor(obs), target, obs["action_mask"], obs["embodiment"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite imitation loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step == 1 or step % args.log_interval == 0 or step == args.updates:
            selection_data = data["train"] if args.overfit_per_body else data["validation"]
            metrics = action_metrics(env, actor, selection_data, train_variance)
            score = sum(m["raw_mse"] for m in metrics.values()) / len(metrics)
            if score < best:
                best, info["best_step"] = score, step
                save(args.run_dir / "best.pt", step)
            record = {
                "step": step,
                "batch_loss": float(loss.detach()),
                "selection_mse": score,
                "per_body": metrics,
                "seconds": time.monotonic() - started,
            }
            history.append(record)
            (args.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    info["imitation_steps"] = args.updates
    save(args.run_dir / "final.pt", args.updates)
    selected = torch.load(args.run_dir / "best.pt", map_location=env.device, weights_only=False)
    actor.load_state_dict(selected["actor_state_dict"])
    info["selected_step"] = selected["infos"]["imitation_steps"]
    info["action_metrics"] = {split: action_metrics(env, actor, part, train_variance) for split, part in data.items()}
    info["seconds"] = time.monotonic() - started
    (args.run_dir / "experiment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


def main() -> None:
    """Keep source teacher collection, offline fitting and frozen evaluation explicitly separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "fit", "evaluate"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", choices=("message_passing", "attention", "mlp"), default="message_passing")
    parser.add_argument("--bodies", nargs="+", choices=("ant_3", "ant_4"), default=["ant_3", "ant_4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--batch_per_body", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--overfit_per_body", type=int, default=0)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}_data", type=Path, nargs="+")
    args = parser.parse_args()
    if min(args.worlds, args.steps, args.stride, args.updates, args.batch_per_body, args.log_interval) < 1:
        parser.error("Counts must be positive.")
    if args.learning_rate <= 0 or args.overfit_per_body < 0 or len(set(args.bodies)) != len(args.bodies):
        parser.error("Require positive learning rate, nonnegative overfit count, and distinct bodies.")
    if args.mode == "fit":
        if not args.run_dir or any(getattr(args, f"{s}_data") is None for s in ("train", "validation", "test")):
            parser.error("Fitting requires --run_dir and disjoint --train_data, --validation_data, --test_data.")
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            parser.error("Fitting requires a new/empty run directory.")
    elif not args.output or args.output.exists():
        parser.error("Collection/evaluation requires a new --output path.")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint.")
    saved = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher_info = saved["infos"]
    if teacher_info["condition"] != "specialists" or teacher_info["split"] != "canonical":
        parser.error("This diagnostic requires canonical source specialists; target teachers are excluded.")
    versions = {name: importlib.metadata.version(name) for name in teacher_info["versions"]}
    if versions != teacher_info["versions"] or teacher_info["schema"] != SCHEMA_VERSION:
        parser.error("Teacher dependencies or descriptor schema changed; establish a new reference first.")
    cfg = mechanical_ant_cfg("canonical", 2 if args.mode == "fit" else args.worlds, args.seed)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            fingerprint = descriptor_hash(env)
            reference = torch.load(args.teacher.parent / "descriptors.pt", map_location="cpu", weights_only=True)
            reference_hash = hashlib.sha256()
            for fields in reference.values():
                for value in fields.values():
                    reference_hash.update(value.numpy().tobytes())
            if reference_hash.hexdigest() != teacher_info["descriptor_sha256"]:
                raise ValueError("Teacher descriptor snapshot does not match its checkpoint.")
            actual = {
                name: {key: getattr(graph, key).cpu() for key in ("static", "edges", "action_nodes", "context")}
                for name, graph in env.graphs.items()
            }
            roundoff = descriptor_roundoff(actual, reference)
            metadata = {
                "schema": SCHEMA_VERSION,
                "data_schema": DATA_SCHEMA,
                "teacher_checkpoint": str(args.teacher.resolve()),
                "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                "descriptor_sha256": teacher_info["descriptor_sha256"],
                "observed_descriptor_sha256": fingerprint,
                "descriptor_roundoff_max_abs": roundoff,
                "versions": versions,
            }
            if args.mode == "fit":
                fit(env, args, metadata)
            elif args.mode == "collect":
                teacher = make_actor(env, "mlp")
                teacher.load_state_dict(saved["actor_state_dict"])
                teacher.eval()
                collect(env, teacher, args, metadata)
            else:
                checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
                info = checkpoint["infos"]
                for key in ("teacher_sha256", "descriptor_sha256", "schema", "versions"):
                    if info[key] != metadata[key]:
                        raise ValueError(f"Student checkpoint mismatch: {key}.")
                actor = make_actor(env, info["student"])
                actor.load_state_dict(checkpoint["actor_state_dict"])
                actor.eval()
                report = evaluate(env, actor, args.steps)
                report.update(
                    checkpoint=str(args.checkpoint.resolve()),
                    seed=args.seed,
                    steps=args.steps,
                    worlds=args.worlds,
                    trained_bodies=info["training_agents"],
                    imitation_steps=info["imitation_steps"],
                    student=info["student"],
                    target_training=False,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2), flush=True)
        finally:
            raw.close()


if __name__ == "__main__":
    main()
