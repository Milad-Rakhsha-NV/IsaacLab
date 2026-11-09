# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose Ant and Go2 expert demonstrations without changing their physics or observation encodings."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch
from distill_mechanical_ant import collect, encode_rows, fit
from evaluate_mechanical_transfer import transfer_cfg
from train_mechanical_ant import evaluate

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls import MechanicalSpecialists
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import MechanicalAntEnv, mechanical_ant_cfg
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_imitation import descriptor_roundoff
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION, MechanicalAntGraph
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_policy import MechanicalActor
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv

GRAPH_TENSORS = ("static", "edges", "action_nodes", "context")
GRAPH_FIELDS = ("num_bodies", "num_joints", "num_nodes", "num_feet", "base_width", "flat_width", "foot_nodes")


def file_hash(path: Path) -> str:
    """Hash the actual local artifact used in this study."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict:
    """Use the same dependency contract as the source experiments."""
    return {
        name: importlib.metadata.version(name) for name in ("torch", "newton", "mujoco-warp", "warp-lang", "rsl-rl-lib")
    }


class StoredGraph(MechanicalAntGraph):
    """Replay the existing encoder using its captured constants, without a new simulator scene."""

    def __init__(self, snapshot: dict, device: str):
        self.device = device
        for name in GRAPH_TENSORS:
            setattr(self, name, snapshot[name].to(device))
        for name in GRAPH_FIELDS:
            setattr(self, name, snapshot[name])
        if self.base_width != 12 + 3 * self.num_joints + 6 * self.num_feet:
            raise ValueError("Stored graph base observation width is inconsistent.")
        if self.flat_width != self.base_width + 3 + 13 * self.num_bodies:
            raise ValueError("Stored body observation width is inconsistent.")
        if self.num_nodes != len(self.static) or self.num_joints != len(self.action_nodes):
            raise ValueError("Stored graph token/action dimensions are inconsistent.")


class DemonstrationLayout:
    """Provide the unchanged fitting helpers with one layout per source physics configuration.

    Ant descriptors retain 120 Hz physics; Go2 descriptors retain 480 Hz physics. Both
    have 60 Hz actions. Mixing offline batches never changes either teacher's plant.
    """

    def __init__(self, bank: dict, agents: list[str], examples: dict, device: str = "cuda:0"):
        self.device, self.agents = device, agents
        self.graphs = {name: StoredGraph(bank[name], device) for name in agents}
        self.node_width = max(g.num_nodes for g in self.graphs.values())
        self.num_actions = max(g.num_joints for g in self.graphs.values())
        self.critic_width = max(g.flat_width for g in self.graphs.values())
        self.observations = torch.cat([encode_rows(self, name, examples[name][:1].to(device)) for name in agents])

    def get_observations(self):
        return self.observations


def export_layout(env: MechanicalVecEnv, teacher: Path, output: Path) -> None:
    """Verify finalized descriptors and exact live/stored encoding before saving a reusable layout."""
    saved = torch.load(teacher, map_location="cpu", weights_only=False)
    info = saved["infos"]
    if info["schema"] != SCHEMA_VERSION or info["versions"] != versions():
        raise ValueError("Teacher schema or dependencies changed.")
    reference = torch.load(teacher.parent / "descriptors.pt", map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for fields in reference.values():
        for tensor in fields.values():
            digest.update(tensor.numpy().tobytes())
    if digest.hexdigest() != info["descriptor_sha256"]:
        raise ValueError("Teacher descriptor snapshot does not match its checkpoint.")
    actual = {name: {key: getattr(graph, key).cpu() for key in GRAPH_TENSORS} for name, graph in env.graphs.items()}
    differences = descriptor_roundoff(actual, reference)
    bank = {
        name: {**actual[name], **{key: getattr(graph, key) for key in GRAPH_FIELDS}}
        for name, graph in env.graphs.items()
    }
    for index, name in enumerate(env.agents):
        rows = slice(index * env.env.num_envs, (index + 1) * env.env.num_envs)
        flat = env.get_observations()["critic"][rows, : env.graphs[name].flat_width]
        copied = StoredGraph(bank[name], env.device).encode(flat, env.node_width, env.num_actions)
        live = env.graphs[name].encode(flat, env.node_width, env.num_actions)
        for key in live.keys():
            if not torch.equal(copied[key], live[key]):
                raise RuntimeError(f"Stored observation encoder changed {name}/{key}.")
    torch.save(
        {
            "schema": SCHEMA_VERSION,
            "versions": versions(),
            "teacher_sha256": file_hash(teacher),
            "descriptor_sha256": info["descriptor_sha256"],
            "roundoff": differences,
            "graphs": bank,
        },
        output,
    )


def bundle(args) -> None:
    """Combine corresponding source-only collections, checking each teacher separately."""
    ant_layout = torch.load(args.ant_layout, map_location="cpu", weights_only=True)
    go2_layout = torch.load(args.go2_layout, map_location="cpu", weights_only=True)
    layouts = (ant_layout, go2_layout)
    if set(ant_layout["graphs"]) != {"ant_3", "ant_4"} or set(go2_layout["graphs"]) != {"go2"}:
        raise ValueError("The source bank must contain canonical Ants and Go2 only.")
    bank = {**ant_layout["graphs"], **go2_layout["graphs"]}
    args.output.mkdir(parents=True, exist_ok=False)
    teachers = {name: layout["teacher_sha256"] for layout in layouts for name in layout["graphs"]}
    descriptor_digest = hashlib.sha256()
    for fields in bank.values():
        for key in GRAPH_TENSORS:
            descriptor_digest.update(fields[key].numpy().tobytes())
    metadata = {
        "schema": SCHEMA_VERSION,
        "data_schema": "mechanical-cross-family-demonstrations-v1",
        "versions": versions(),
        "teacher_sha256": hashlib.sha256(json.dumps(teachers, sort_keys=True).encode()).hexdigest(),
        "teacher_hashes_by_body": teachers,
        "descriptor_sha256": descriptor_digest.hexdigest(),
        "split": "ant_go2_sources",
        "target_training": True,
        "note": "Go2 is now a training body; A1 and five-leg targets supply no demonstration data.",
        "source_files": {},
        "graphs_file": "layouts.pt",
    }
    torch.save(bank, args.output / "layouts.pt")
    for seed in (1011, 2022, 3033, 4044):
        paths = (args.ant_data / f"data_{seed}.pt", args.go2_data / f"data_{seed}.pt")
        combined = {}
        for path, layout in zip(paths, layouts, strict=True):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            meta = saved["metadata"]
            for key in ("schema", "versions", "teacher_sha256", "descriptor_sha256"):
                if meta[key] != layout[key]:
                    raise ValueError(f"Collection {path} has incompatible {key}.")
            if meta["seed"] != seed or set(saved["data"]) != set(layout["graphs"]):
                raise ValueError("Collection seed or body identity differs.")
            for name, fields in saved["data"].items():
                if fields["flat"].shape[1] != bank[name]["flat_width"]:
                    raise ValueError("Collection observation width changed.")
            combined.update(saved["data"])
            metadata["source_files"][str(path.resolve())] = file_hash(path)
        torch.save({"metadata": dict(metadata, seed=seed), "data": combined}, args.output / f"data_{seed}.pt")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    """Export, collect, compose, fit or evaluate while preserving disjoint demonstration splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export_ant", "collect_go2", "bundle", "fit", "evaluate"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--ant_data", type=Path)
    parser.add_argument("--go2_data", type=Path)
    parser.add_argument("--ant_layout", type=Path)
    parser.add_argument("--go2_layout", type=Path)
    parser.add_argument("--bodies", nargs="+", choices=("ant_3", "ant_4", "go2"), default=["ant_3", "ant_4", "go2"])
    parser.add_argument("--robot", choices=("ant_sources", "go2", "a1", "dev_5"), default="go2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--batch_per_body", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=250)
    args = parser.parse_args()
    if min(args.worlds, args.steps, args.stride, args.updates, args.batch_per_body, args.log_interval) < 1:
        parser.error("Counts must be positive.")
    if args.learning_rate <= 0 or len(set(args.bodies)) != len(args.bodies):
        parser.error("Learning rate must be positive and source bodies distinct.")
    if args.mode == "bundle":
        if not all((args.ant_data, args.go2_data, args.ant_layout, args.go2_layout, args.output)):
            parser.error("Bundling requires both data directories, both layouts and a new output directory.")
        bundle(args)
        return
    if args.mode == "fit":
        if not args.data or not args.run_dir or (args.run_dir.exists() and any(args.run_dir.iterdir())):
            parser.error("Fitting requires a data bundle and a new/empty run directory.")
        metadata = json.loads((args.data / "metadata.json").read_text())
        if metadata["schema"] != SCHEMA_VERSION or metadata["versions"] != versions():
            parser.error("Bundle schema or dependencies changed.")
        bank = torch.load(args.data / "layouts.pt", map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for fields in bank.values():
            for key in GRAPH_TENSORS:
                digest.update(fields[key].numpy().tobytes())
        if digest.hexdigest() != metadata["descriptor_sha256"]:
            raise ValueError("Stored observation layouts differ from the data bundle.")
        first = torch.load(args.data / "data_1011.pt", map_location="cpu", weights_only=True)
        layout = DemonstrationLayout(bank, args.bodies, {n: d["flat"] for n, d in first["data"].items()})
        args.student, args.overfit_per_body = "attention", 0
        args.train_data = [args.data / "data_1011.pt", args.data / "data_2022.pt"]
        args.validation_data, args.test_data = [args.data / "data_3033.pt"], [args.data / "data_4044.pt"]
        args.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), args.run_dir / Path(__file__).name)
        fit(layout, args, dict(metadata, cross_family_source_sha256=file_hash(Path(__file__))))
        return
    if not args.output or args.output.exists():
        parser.error("This mode requires a new output path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("Evaluation requires a checkpoint.")
        cfg = (
            mechanical_ant_cfg("canonical", args.worlds, args.seed)
            if args.robot == "ant_sources"
            else transfer_cfg(args.robot, args.worlds, args.seed)
        )
    else:
        if not args.teacher:
            parser.error("Layout export and collection require a teacher.")
        cfg = (
            mechanical_ant_cfg("canonical", 2, args.seed)
            if args.mode == "export_ant"
            else transfer_cfg("go2", args.worlds, args.seed)
        )
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            if args.mode == "export_ant":
                export_layout(env, args.teacher, args.output)
            elif args.mode == "collect_go2":
                layout_path = args.output.with_suffix(".layout.pt")
                export_layout(env, args.teacher, layout_path)
                saved = torch.load(args.teacher, map_location="cpu", weights_only=False)
                if saved["infos"]["training_agents"] != ["go2"]:
                    raise ValueError("Go2 collection requires its own specialist.")
                teacher = MechanicalSpecialists(
                    env.get_observations(),
                    {},
                    "actor",
                    env.num_actions,
                    [env.graphs["go2"].flat_width],
                    [env.num_actions],
                ).to(env.device)
                teacher.load_state_dict(saved["actor_state_dict"], strict=True)
                teacher.eval()
                before = {k: v.clone() for k, v in teacher.state_dict().items()}
                collect(
                    env,
                    teacher,
                    args,
                    {
                        "schema": SCHEMA_VERSION,
                        "data_schema": "mechanical-ant-demonstrations-v1",
                        "versions": versions(),
                        "teacher_sha256": file_hash(args.teacher),
                        "descriptor_sha256": saved["infos"]["descriptor_sha256"],
                        "teacher_checkpoint": str(args.teacher.resolve()),
                        "layout": str(layout_path.resolve()),
                    },
                )
                assert all(torch.equal(v, before[k]) for k, v in teacher.state_dict().items())
            else:
                saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                info = saved["infos"]
                if info["versions"] != versions() or info["schema"] != SCHEMA_VERSION:
                    raise ValueError("Evaluation schema or dependencies changed.")
                actor = MechanicalActor(
                    env.get_observations(), {}, "actor", env.num_actions, routing=info["routing"]
                ).to(env.device)
                actor.load_state_dict(saved["actor_state_dict"], strict=True)
                actor.eval()
                before = {k: v.clone() for k, v in actor.state_dict().items()}
                report = evaluate(env, actor, args.steps)
                assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
                report.update(
                    checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=file_hash(args.checkpoint),
                    training_agents=info["training_agents"],
                    seed=args.seed,
                    worlds=args.worlds,
                    steps=args.steps,
                    evaluation_updates=0,
                    actor_and_normalization_unchanged=True,
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2), flush=True)
        finally:
            raw.close()


if __name__ == "__main__":
    main()
