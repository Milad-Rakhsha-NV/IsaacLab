# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train matched G1 rigid/RFT controls, validate them and record both inference speeds."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from go2_granular_videos import PRIMARY_VIDEOS, describe_progress, video_cases
from imageio_ffmpeg import count_frames_and_secs
from report_g1_granular import ROOT, publish
from report_go2_materials import policy_report
from run_go2_speed_comparison import write_json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from validate_go2_training import job_state, validation_gates


def mpm_startup_checks(evaluate: Callable[..., dict], checkpoint: Path) -> list[dict]:
    """Compare feedback and pass refinement before the long MPM recordings."""
    results = []
    for speed in (0.5, 1.0):
        for passes, feedback in ((1, 0.1), (1, 1.0), (4, 1.0)):
            name = f"mpm_startup_{speed:g}_passes_{passes}_feedback_{feedback:g}"
            summary = evaluate(
                name,
                "mpm",
                "rft",
                checkpoint,
                [
                    "--steps",
                    "100",
                    "--speed",
                    str(speed),
                    "--bed_depth",
                    "0.118",
                    "--bed_length",
                    "5",
                    "--bed_width",
                    "3.2",
                    "--max_active_cell_count",
                    str(1 << 20),
                    "--settle_seconds",
                    "0.5",
                    "--proxy_iterations",
                    str(passes),
                    "--proxy_relaxation",
                    str(feedback),
                    "--audit_forces",
                ],
                label=f"MPM startup at {speed:.2f} m/s: {passes} coupling passes, feedback {feedback:g}",
            )
            results.append(dict(case=name, **summary))
    return results


def record_video(
    evaluate: Callable[..., dict], name: str, field: str, policy: str, checkpoint: Path, speed: float
) -> None:
    """Verify a final movie before exposing it in the report."""
    options = ["--steps", "500", "--speed", str(speed), "--video"]
    if field == "rft":
        options += ["--substrate_depth", "0.118"]
    if field == "mpm":
        options += [
            "--bed_depth",
            "0.118",
            "--bed_length",
            "12",
            "--bed_width",
            "3.2",
            "--max_active_cell_count",
            str(1 << 20),
            "--settle_seconds",
            "0.5",
            "--proxy_relaxation",
            "1.0",
        ]
    summary = evaluate(name, field, policy, checkpoint, options)
    folder = ROOT / "evaluations" / name
    frames, seconds = count_frames_and_secs(folder / "rollout.mp4")
    assert frames == 500 and abs(seconds - 10) < 0.1
    with np.load(folder / "rollout.npz") as data:
        np.testing.assert_array_equal(data["commands"], np.broadcast_to([speed, 0, 0], data["commands"].shape))
    if field == "mpm" and summary["first_bed_exit_seconds"] is not None:
        raise RuntimeError(f"G1 recording reaches the bed boundary: {name}")


def validate_rigid_reuse(status: dict) -> None:
    """Require accepted held-out checks and both native movies before reusing a baseline."""
    gates = status.get("validation", {}).get("rigid", {})
    required = {name for name, base, _ in video_cases((1.0, 0.5)) if base == "rigid_max_forward"}
    if (
        "rigid" not in status["runs"]
        or not gates
        or not all(gates.values())
        or not required.issubset(status.get("published_videos", []))
    ):
        raise RuntimeError("The rigid baseline must pass validation and recording checks before reuse")


def initialization_options(policy: str, checkpoints: dict, status: dict) -> list[str]:
    """Initialize sand adaptation from the accepted rigid policy and record its provenance."""
    if policy != "rft":
        return []
    initialization = checkpoints["rigid"]
    status["rft_initialization"] = dict(
        checkpoint=str(initialization),
        sha256=hashlib.sha256(initialization.read_bytes()).hexdigest(),
        iteration=1499,
        optimizer_restored=False,
        additional_iterations=1500,
    )
    return ["--checkpoint", str(initialization)]


def main() -> None:
    """Keep the full study durable and reject an invalid rigid baseline before RFT training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse_validated_rigid",
        action="store_true",
        help="Retain the accepted rigid baseline when only the RFT curriculum changes.",
    )
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"validation": {}, "runs": {}, "cases": []}
    status.setdefault("runs", {})
    status.setdefault("cases", [])
    status.setdefault("validation", {})
    status.pop("error", None)
    status.pop("active_case", None)
    status["mpm_profile"] = dict(
        proxy_iterations=1, proxy_relaxation=1.0, substeps=4, bed_width=3.2, max_active_cell_count=1 << 20
    )
    titles = {base: title for base, title, *_ in PRIMARY_VIDEOS}
    video_labels = {name: f"{titles[base]} ({speed:.2f} m/s)" for name, base, speed in video_cases((1.0, 0.5))}
    if args.reuse_validated_rigid:
        validate_rigid_reuse(status)

    def update(stage):
        status.update(stage=stage, updated_unix=time.time())
        write_json(path, status)
        publish()
        lines = [
            "# G1 rigid, RFT and MPM study",
            "",
            stage,
            "",
            "Both stages: 4,096 environments, seed 42, 1,500 PPO iterations. Forward commands: 0.1–1.0 m/s.",
            "Rigid trains from scratch; RFT fine-tunes the accepted rigid policy for 1,500 additional iterations.",
            "No command curriculum. RFT materials change at reset; layers start at 0–10 mm and widen to 0–118 mm.",
            "MPM is inference only; numerical convergence and material equivalence remain unvalidated.",
            "",
            "[Report](http://localhost:18010/g1/report/)",
            "",
            "| Trial | Mean tracking error | First failure threshold |",
            "|---|---:|---|",
        ]
        for case in status["cases"]:
            if case["case"] not in status.get("published_videos", []):
                continue
            first = case["first_failure_seconds"]
            outcome = "None" if first is None else f"{first:.2f} s"
            lines.append(f"| {video_labels[case['case']]} | {case['mean_velocity_error']:.3f} m/s | {outcome} |")
        (ROOT / "STATUS.md").write_text("\n".join(lines) + "\n")

    def evaluate(name, field, policy, checkpoint, options, seed=123, label=None):
        folder = ROOT / "evaluations" / name
        folder.mkdir(parents=True, exist_ok=True)
        worker = Path("scripts/benchmarks/evaluate_go2_granular.py")
        signature = dict(
            checkpoint=str(checkpoint),
            field=field,
            policy=policy,
            options=options,
            seed=seed,
            worker_ast=hashlib.sha256(ast.dump(ast.parse(worker.read_text())).encode()).hexdigest(),
            environment_sha256=hashlib.sha256(
                Path("source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/granular_env_cfg.py").read_bytes()
                + Path("source/isaaclab_tasks/isaaclab_tasks/core/velocity/granular_env.py").read_bytes()
            ).hexdigest(),
        )
        manifest = folder / "case.json"
        summary_file = folder / "summary.json"
        cached = summary_file.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature
        if cached:
            cached = json.loads(summary_file.read_text())["execution_error"] is None
        if not cached:
            status["active_case"] = name
            update("Evaluating " + (label or describe_progress(name.replace("_", " "), (1.0, 0.5))))
            write_json(manifest, signature)
            command = [
                sys.executable,
                str(worker),
                "--robot",
                "g1",
                "--field",
                field,
                "--policy_training_terrain",
                policy,
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(folder),
                "--seed",
                str(seed),
                "--max_forward_command",
                "1.0",
                *options,
            ]
            with (folder / "stdout.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads(summary_file.read_text())
        if summary["execution_error"] or not summary["particles_finite"] or summary["max_rigid_overflow"]:
            raise RuntimeError(f"Invalid G1 evaluation {name}")
        status["cases"] = [case for case in status["cases"] if case["case"] != name]
        status["cases"].append(dict(case=name, **summary))
        status.pop("active_case", None)
        update("Completed " + (label or describe_progress(name.replace("_", " "), (1.0, 0.5))))
        return summary

    try:
        update("G1 native drive/reset checks passed; running the rigid baseline before RFT training")
        checkpoints = {}
        for policy in ("rigid", "rft"):
            family = f"g1_granular_{policy}_max1"
            unit = f"g1-granular-{policy}-training.service"
            initial_iteration = 1499 if policy == "rft" else 0
            final_checkpoint = f"model_{initial_iteration + 1499}.pt"
            if policy not in status["runs"]:
                update(f"Training G1 on {policy.upper()} with 4,096 environments")
                log = (ROOT / f"{policy}-training.log").resolve()
                command = [
                    "systemd-run",
                    "--user",
                    "--unit",
                    unit,
                    "--property",
                    f"WorkingDirectory={Path.cwd()}",
                    "--property",
                    f"StandardOutput=append:{log}",
                    "--property",
                    "StandardError=inherit",
                    shutil.which("uv"),
                    "run",
                    "--no-sync",
                    "isaaclab",
                    "train",
                    "--rl_library",
                    "rsl_rl",
                    "--task",
                    f"Isaac-Velocity-Flat-G1-Granular-{policy.upper() if policy == 'rft' else 'Rigid'}-Direct",
                    "--num_envs",
                    "4096",
                    "--max_iterations",
                    "1500",
                    "--seed",
                    "42",
                    "--experiment_name",
                    family,
                ]
                command += initialization_options(policy, checkpoints, status)
                write_json(path, status)
                if job_state(unit).get("ActiveState") not in ("active", "activating"):
                    subprocess.run(command, check=True)
                while job_state(unit).get("ActiveState") in ("active", "activating"):
                    publish()
                    time.sleep(30)
                state = job_state(unit)
                if state.get("Result") != "success" or state.get("ExecMainStatus") != "0":
                    raise RuntimeError(f"G1 training failed: {state}")
                run = sorted(Path("logs/rsl_rl", family).glob("20*"))[-1]
                if not (run / final_checkpoint).exists():
                    raise RuntimeError(f"Incomplete G1 training: {run}")
                status["runs"][policy] = str(run)
                update(f"G1 {policy.upper()} training budget completed")
            run = Path(status["runs"][policy])
            checkpoints[policy] = run / final_checkpoint
            status.setdefault("checkpoints", {})[policy] = str(checkpoints[policy])
            if policy == "rigid" and args.reuse_validated_rigid:
                continue
            checks = []
            depth = ["--substrate_depth", "0.118"] if policy == "rft" else []
            for label, speed in (("stand", 0.0), ("walk", 0.5), ("maximum", 1.0)):
                checks.append(
                    evaluate(
                        f"{policy}_{label}_validation",
                        policy,
                        policy,
                        checkpoints[policy],
                        [*depth, "--steps", "1000", "--num_envs", "16", "--speed", str(speed)],
                    )
                )
            for seed in (123, 456, 789):
                checks.append(
                    evaluate(
                        f"{policy}_random_{seed}",
                        policy,
                        policy,
                        checkpoints[policy],
                        [*depth, "--steps", "500", "--num_envs", "128", "--randomize"],
                        seed=seed,
                    )
                )
            difficulty = None
            if policy == "rft":
                events = EventAccumulator(str(run)).Reload()
                difficulty = events.Scalars("Curriculum/difficulty")[-1].value
                status["material_difficulty"] = difficulty
            gates = validation_gates(checks, material_difficulty=difficulty)
            status["validation"][policy] = gates
            update(f"G1 {policy.upper()} held-out checks completed")
            if policy == "rigid" and not all(gates.values()):
                raise RuntimeError(f"Rigid G1 baseline requires correction before sand training: {gates}")
            if policy == "rigid":
                for name, base, speed in video_cases((1.0, 0.5)):
                    if base == "rigid_max_forward":
                        record_video(evaluate, name, "rigid", "rigid", checkpoints["rigid"], speed)
                        status["published_videos"] = list(dict.fromkeys([*status.get("published_videos", []), name]))
                        update(f"Verified G1 recording: {video_labels[name]}")
        roles = {
            "rigid_max_forward": ("rigid", "rigid"),
            "rft_max_forward": ("rft", "rft"),
            "mpm_max_forward": ("mpm", "rft"),
            "rigid_mpm_max_forward": ("mpm", "rigid"),
        }
        for name, base, speed in video_cases((1.0, 0.5)):
            if name in status.get("published_videos", []):
                continue
            field, policy = roles[base]
            if field == "mpm":
                if not status.get("mpm_startup_complete"):
                    status["mpm_startup"] = mpm_startup_checks(evaluate, checkpoints["rft"])
                    status["mpm_startup_complete"] = True
                    update("G1 MPM coupling checks completed; recording both policies with full feedback")
            record_video(evaluate, name, field, policy, checkpoints[policy], speed)
            status["published_videos"] = list(dict.fromkeys([*status.get("published_videos", []), name]))
            update(f"Verified G1 recording: {video_labels[name]}")
        for speed in (0.5,):
            material_root = ROOT / f"materials_{speed:g}"
            profile = ROOT / "material-profile.json"
            write_json(profile, status["mpm_profile"])
            for phase in ("materials", "reference"):
                update(f"G1 three-material comparisons at {speed:g} m/s: {phase}")
                command = [
                    sys.executable,
                    "scripts/benchmarks/evaluate_go2_material_bank.py",
                    "--robot",
                    "g1",
                    "--phase",
                    phase,
                    "--output_root",
                    str(material_root),
                    "--speed",
                    str(speed),
                    "--max_forward_command",
                    "1.0",
                    "--bed_length",
                    "5",
                    "--bed_width",
                    "3.2",
                    "--profile",
                    str(profile),
                    "--checkpoint_rigid",
                    str(checkpoints["rigid"]),
                    "--checkpoint_rft",
                    str(checkpoints["rft"]),
                ]
                with (ROOT / f"materials-{speed:g}-{phase}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            analysis = policy_report(material_root, material_root, speed=speed)
            write_json(material_root / "policy-analysis.json", analysis)
        status["complete"] = True
        update("G1 training, acceptance checks, eight paired-speed videos and material plots completed")
    except Exception as error:
        status.pop("active_case", None)
        status["error"] = repr(error)
        update("G1 study stopped; completed outputs remain available. Inspect the recorded error.")
        raise


if __name__ == "__main__":
    main()
