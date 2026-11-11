# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the final frozen Go2 policies when training finishes.

Uses the two training services and run families from ``monitor_go2_granular.py``.
MPM runs only after the new RFT policy passes its validation gates. Progress is
published for the live dashboard; physical failures remain in the recordings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from monitor_go2_granular import ROOT, RUNS
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def job_state(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState,Result,ExecMainStatus"],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def latest_checkpoint(family: str) -> Path | None:
    runs = sorted(Path("logs/rsl_rl", family).glob("20*"))
    if not runs:
        return None
    paths = [path for path in runs[-1].glob("model_*.pt") if time.time() - path.stat().st_mtime > 10]
    return max(paths, key=lambda path: int(path.stem.split("_")[-1])) if paths else None


def main(rigid_only: bool = False) -> None:
    row = RUNS[1] if rigid_only else RUNS[0]
    field = "rigid" if rigid_only else "rft"
    prefix = "rigid_gait" if rigid_only else "rft_knee_depth"
    status_path = ROOT / "training" / ("evaluation_status.json" if rigid_only else "rft-evaluation-status.json")
    progress = dict(stage="Training in progress; final-policy validation will follow", cases=[])

    def publish(stage: str) -> None:
        progress["stage"] = stage
        progress["updated_unix"] = time.time()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.with_suffix(".tmp").write_text(json.dumps(progress, indent=2))
        status_path.with_suffix(".tmp").replace(status_path)

    def evaluate(name: str, checkpoint: Path, *options: str, seed: int = 123, label: str) -> dict:
        publish(label)
        output = ROOT / name
        command = [
            sys.executable,
            "scripts/benchmarks/evaluate_go2_granular.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--seed",
            str(seed),
            *options,
        ]
        with (status_path.parent / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / "summary.json").read_text())
        if summary["execution_error"] is not None:
            raise RuntimeError(f"{name}: {summary['execution_error']}")
        progress["cases"].append(dict(case=name, **summary))
        publish("Completed policy evaluation")
        return summary

    publish(progress["stage"])
    try:
        while True:
            state = job_state(row[3])
            if state["ActiveState"] == "failed":
                raise RuntimeError(f"Training job failed: {state}")
            if state["ActiveState"] == "inactive":
                if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                    raise RuntimeError(f"Training did not finish successfully: {state}")
                break
            time.sleep(30)
        time.sleep(12)
        checkpoint = latest_checkpoint(row[2])
        if checkpoint is None:
            raise RuntimeError("Missing final training checkpoint")
        options = ["--field", field]
        achieved = 1.0
        depth = None
        if not rigid_only:
            events = EventAccumulator(str(checkpoint.parent)).Reload()
            achieved = events.Scalars("Curriculum/difficulty")[-1].value
            manifest = json.loads((checkpoint.parent / "material_curriculum.json").read_text())
            initial = manifest["initial_depth_range_m"][1]
            target = manifest["target_depth_range_m"][1]
            depth = initial + achieved * (target - initial)
            options += ["--substrate_depth", str(depth)]
            progress.update(achieved_material_difficulty=achieved, evaluation_layer_depth_m=depth)
        checks = []
        for name, speed in (("stand", "0.0"), ("walk", "0.3"), ("maximum", "0.5")):
            checks.append(
                evaluate(
                    f"{prefix}_{name}_validation",
                    checkpoint,
                    *options,
                    "--steps",
                    "1000",
                    "--num_envs",
                    "16",
                    "--speed",
                    speed,
                    label=f"Checking {field.upper()} policy: {name} command",
                )
            )
        for seed in (123, 456, 789):
            checks.append(
                evaluate(
                    f"{prefix}_commands_{seed}",
                    checkpoint,
                    *options,
                    "--steps",
                    "500",
                    "--num_envs",
                    "128",
                    "--randomize",
                    seed=seed,
                    label=f"Checking randomized {field.upper()} trials, seed {seed}",
                )
            )
        if achieved < 0.999:
            evaluate(
                "rft_knee_depth_achieved_validation",
                checkpoint,
                *options,
                "--steps",
                "500",
                "--num_envs",
                "128",
                "--randomize",
                "--difficulty",
                str(achieved),
                label="Checking the achieved material curriculum range",
            )
        video = evaluate(
            "rigid_max_forward" if rigid_only else "rft_max_forward",
            checkpoint,
            *options,
            "--steps",
            "500",
            "--speed",
            "0.5",
            "--video",
            "--policy_training_terrain",
            "rigid" if rigid_only else "rft",
            label=f"Recording the final {field.upper()} policy",
        )
        gates = dict(
            no_falls=all(case["failure_fraction"] == 0 for case in checks + [video]),
            foot_placement=all(case["gait_pass_fraction"] == 1 for case in checks + [video]),
            tracking=all(case["mean_velocity_error"] < 0.10 for case in checks),
            standing=checks[0]["mean_velocity_error"] < 0.03,
            contact_slip=all(case["mean_contact_slip_m_s"] < 0.10 for case in checks + [video]),
            capacity=all(case["max_rigid_overflow"] == 0 for case in checks + [video]),
            every_foot_steps=all(min(counts) >= 3 for case in checks[1:3] for counts in case["steps_per_foot"]),
        )
        if not rigid_only:
            gates["full_material_curriculum"] = achieved >= 0.999
        progress[f"{field}_validation_gates"] = gates
        progress[f"{field}_control_accepted"] = all(gates.values())
        if not rigid_only and all(gates.values()):
            mpm = evaluate(
                "mpm_max_forward",
                checkpoint,
                "--field",
                "mpm",
                "--steps",
                "500",
                "--speed",
                "0.5",
                "--bed_depth",
                str(depth),
                "--bed_length",
                "7.0",
                "--video",
                "--policy_training_terrain",
                "rft",
                label="Evaluating the same new policy in MPM (inference only)",
            )
            progress["mpm_transfer_passed"] = mpm["failure_fraction"] == 0
        result = f"{field.upper()} validation passed" if all(gates.values()) else f"{field.upper()} validation failed"
        if progress.get("mpm_transfer_passed") is False:
            result += "; the MPM inference rollout failed"
        publish(result)
        subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
    except Exception as error:
        progress["error"] = repr(error)
        publish("Evaluation stopped with an error; inspect the validation service log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rigid-only", action="store_true", help="Validate the rigid control instead of the new RFT run."
    )
    main(parser.parse_args().rigid_only)
