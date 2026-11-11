# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the active Go2 speed study and publish four matched-command recordings.

Run as a durable background service after launching the two training units listed
in ``logs/go2_granular/current-study.json``. MPM is used only after training.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from imageio_ffmpeg import count_frames_and_secs
from monitor_go2_granular import ROOT, load_study
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from validate_go2_training import job_state, validation_gates


def write_json(path: Path, value: dict) -> None:
    """Publish complete snapshots for the dashboard's concurrent readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps(value, indent=2))
    path.with_suffix(".tmp").replace(path)


def main() -> None:
    """Keep checkpoints, commands, validation and recording provenance consistent."""
    study = load_study()
    if not study:
        raise RuntimeError("Create current-study.json before starting the comparison")
    case_root = ROOT / study["case_directory"]
    case_root.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / study["status_file"]
    progress = json.loads(status_path.read_text()) if status_path.exists() else dict(cases=[])
    progress.setdefault("cases", [])
    progress.pop("error", None)
    maximum = study["max_forward_command"]

    def publish(stage: str, *, render: bool = False) -> None:
        progress.update(stage=stage, updated_unix=time.time())
        write_json(status_path, progress)
        lines = [
            "# Go2 speed comparison status",
            "",
            stage,
            "",
            f"Both policies: {study['num_envs']:,} environments, {study['iterations']:,} iterations, "
            f"seed {study['seed']}.",
            f"Forward commands: 0.1–{maximum:g} m/s from the first reset; no command curriculum.",
            "RFT material randomization is unchanged: reset-based material curriculum, 0–118 mm layers.",
            "MPM is inference only. Its force-convergence and material-equivalence gates remain unaccepted.",
            "",
            "| Trial | Mean tracking error | First failure threshold |",
            "|---|---:|---|",
        ]
        for row in progress["cases"]:
            failure = row["first_failure_seconds"]
            outcome = "None" if failure is None else f"{failure:.2f} s"
            lines.append(f"| {row['case']} | {row['mean_velocity_error']:.3f} m/s | {outcome} |")
        for field in ("rigid", "rft"):
            gates = progress.get(f"{field}_validation_gates")
            if gates:
                failed = [key.replace("_", " ") for key, passed in gates.items() if not passed]
                lines += [
                    "",
                    f"{field.upper()} acceptance: "
                    + ("passed." if not failed else "failed: " + ", ".join(failed) + "."),
                ]
        lines += ["", "[Report](http://localhost:18010/report/) · [Live training](http://localhost:18010/live/)", ""]
        (case_root / "STATUS.md").write_text("\n".join(lines))
        if render:
            subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)

    def evaluate(name: str, field: str, checkpoint: Path, options: list[str], *, seed: int = 123) -> dict:
        folder = case_root / name
        folder.mkdir(parents=True, exist_ok=True)
        source = Path("scripts/benchmarks/evaluate_go2_granular.py").read_text()
        source_hash = hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()
        manifest = dict(
            checkpoint=str(checkpoint),
            field=field,
            options=options,
            seed=seed,
            maximum=maximum,
            source_ast_sha256=source_hash,
        )
        manifest_path = folder / "case.json"
        summary_path = folder / "summary.json"
        cached = summary_path.exists() and manifest_path.exists() and json.loads(manifest_path.read_text()) == manifest
        summary = json.loads(summary_path.read_text()) if cached else None
        if summary is None or summary.get("execution_error") is not None:
            write_json(manifest_path, manifest)
            publish(f"Evaluating {name.replace('_', ' ')}", render=True)
            command = [
                sys.executable,
                "scripts/benchmarks/evaluate_go2_granular.py",
                "--field",
                field,
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(folder),
                "--seed",
                str(seed),
                "--max_forward_command",
                str(maximum),
                *options,
            ]
            with (folder / "stdout.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads(summary_path.read_text())
        if summary["execution_error"] is not None or not summary["particles_finite"]:
            raise RuntimeError(f"{name}: nonfinite state or execution error")
        if summary["max_rigid_overflow"]:
            raise RuntimeError(f"{name}: rigid-contact buffer overflow")
        progress["cases"] = [row for row in progress["cases"] if row["case"] != name]
        progress["cases"].append(dict(case=name, **summary))
        publish(f"Completed {name.replace('_', ' ')}", render=True)
        return summary

    try:
        publish("Training both policies with forward commands up to 1.0 m/s", render=True)
        while True:
            states = {row[0]: job_state(row[3]) for row in study["runs"]}
            if any(state["ActiveState"] == "failed" for state in states.values()):
                raise RuntimeError(f"A training service failed: {states}")
            if all(state["ActiveState"] == "inactive" for state in states.values()):
                if any(
                    state.get("Result") != "success" or state.get("ExecMainStatus") != "0" for state in states.values()
                ):
                    raise RuntimeError(f"Training did not exit successfully: {states}")
                break
            time.sleep(30)
        time.sleep(12)
        checkpoints = {}
        for key, _, family, _, target in study["runs"]:
            directory = Path(study["run_directories"][key])
            candidates = list(directory.glob("model_*.pt"))
            checkpoint = max(candidates, key=lambda path: int(path.stem.split("_")[-1]))
            if int(checkpoint.stem.split("_")[-1]) < target - 1:
                raise RuntimeError(f"Incomplete training budget in {directory}")
            cfg = yaml.load((directory / "params/env.yaml").read_text(), Loader=yaml.FullLoader)
            assert tuple(cfg["forward_command_range"]) == (0.1, maximum)
            assert cfg["scene"]["num_envs"] == study["num_envs"]
            assert directory.parent.name == family
            checkpoints[key] = checkpoint
        progress["checkpoints"] = {key: str(path) for key, path in checkpoints.items()}
        events = EventAccumulator(str(checkpoints["rft"].parent)).Reload()
        difficulty = events.Scalars("Curriculum/difficulty")[-1].value
        progress["achieved_material_difficulty"] = difficulty
        validations = {}
        for field in ("rigid", "rft"):
            options = ["--substrate_depth", "0.118"] if field == "rft" else []
            checks = []
            for name, speed in (("stand", 0.0), ("walk", 0.5), ("maximum", maximum)):
                checks.append(
                    evaluate(
                        f"{field}_{name}_validation",
                        field,
                        checkpoints[field],
                        [*options, "--steps", "1000", "--num_envs", "16", "--speed", str(speed)],
                    )
                )
            for seed in (123, 456, 789):
                checks.append(
                    evaluate(
                        f"{field}_commands_{seed}",
                        field,
                        checkpoints[field],
                        [*options, "--steps", "500", "--num_envs", "128", "--randomize"],
                        seed=seed,
                    )
                )
            validations[field] = checks
            gates = validation_gates(checks, material_difficulty=difficulty if field == "rft" else None)
            progress[f"{field}_validation_gates"] = gates
            progress[f"{field}_control_accepted"] = all(gates.values())
            publish(f"Completed {field.upper()} policy validation", render=True)

        for name, field, policy in (
            ("rigid_max_forward", "rigid", "rigid"),
            ("rft_max_forward", "rft", "rft"),
            ("mpm_max_forward", "mpm", "rft"),
            ("rigid_mpm_max_forward", "mpm", "rigid"),
        ):
            options = ["--steps", "500", "--speed", str(maximum), "--video", "--policy_training_terrain", policy]
            if field == "rft":
                options += ["--substrate_depth", "0.118"]
            if field == "mpm":
                options += [
                    "--bed_depth",
                    "0.118",
                    "--bed_length",
                    str(max(7.0, 10 * maximum + 2)),
                    "--bed_width",
                    "1.6",
                    "--settle_seconds",
                    "0.5",
                ]
            summary = evaluate(name, field, checkpoints[policy], options)
            assert summary["steps"] == 500
            with np.load(case_root / name / "rollout.npz") as data:
                np.testing.assert_array_equal(
                    data["commands"], np.broadcast_to([maximum, 0, 0], data["commands"].shape)
                )
            frames, duration = count_frames_and_secs(case_root / name / "rollout.mp4")
            assert frames == 500 and abs(duration - 10.0) < 0.1
            if field == "mpm" and summary["first_bed_exit_seconds"] is not None:
                raise RuntimeError(f"{name}: bed-edge exit prevents a clean material comparison")
            if field != "mpm":
                gates = validation_gates(
                    validations[field], summary, material_difficulty=difficulty if field == "rft" else None
                )
                progress[f"{field}_validation_gates"] = gates
                progress[f"{field}_control_accepted"] = all(gates.values())
            study = load_study()
            study["published_videos"] = list(dict.fromkeys([*study.get("published_videos", []), name]))
            write_json(ROOT / "current-study.json", study)
            publish(f"Verified recording: {name.replace('_', ' ')}", render=True)
        study = load_study()
        study["videos_ready"] = True
        write_json(ROOT / "current-study.json", study)
        publish("All four recordings completed; evaluating the new policies across three materials", render=True)
        material_root = case_root / "materials"
        for phase in ("materials", "reference"):
            command = [
                sys.executable,
                "scripts/benchmarks/evaluate_go2_material_bank.py",
                "--phase",
                phase,
                "--output_root",
                str(material_root),
                "--speed",
                str(maximum),
                "--max_forward_command",
                str(maximum),
                "--bed_length",
                str(3 * maximum + 2),
                "--bed_width",
                "1.6",
                "--checkpoint_rigid",
                str(checkpoints["rigid"]),
                "--checkpoint_rft",
                str(checkpoints["rft"]),
            ]
            with (case_root / f"{phase}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        from report_go2_materials import policy_report

        material_result = policy_report(material_root, ROOT / "report", speed=maximum)
        write_json(material_root / "results.json", material_result)
        study["material_comparison_ready"] = True
        write_json(ROOT / "current-study.json", study)
        publish(
            "Training, all four recordings and three-material comparisons completed; see acceptance gates", render=True
        )
        subprocess.run([sys.executable, "scripts/benchmarks/monitor_go2_granular.py", "--once"], check=True)
    except Exception as error:
        progress["error"] = repr(error)
        publish("Speed comparison stopped with an error; completed trials remain available", render=True)
        raise


if __name__ == "__main__":
    main()
