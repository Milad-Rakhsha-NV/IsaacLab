# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the final frozen Go2 policies when training finishes.

Uses the two training services and run families from ``monitor_go2_granular.py``.
MPM runs only after both training jobs have exited successfully. Progress is
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

STATUS = ROOT / "training/evaluation_status.json"


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
    progress = dict(stage="Waiting for training to finish", cases=[])

    def publish(stage: str) -> None:
        progress["stage"] = stage
        progress["updated_unix"] = time.time()
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.with_suffix(".tmp").write_text(json.dumps(progress, indent=2))
        STATUS.with_suffix(".tmp").replace(STATUS)

    def evaluate(name: str, checkpoint: Path, *options: str, seed: int = 123) -> dict:
        publish(f"Evaluating {name}")
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
        with (STATUS.parent / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / "summary.json").read_text())
        if summary["execution_error"] is not None:
            raise RuntimeError(f"{name}: {summary['execution_error']}")
        progress["cases"].append(dict(case=name, **summary))
        publish(f"Finished {name}")
        return summary

    publish(progress["stage"])
    try:
        while True:
            states = [job_state(row[3]) for row in (RUNS[1:] if rigid_only else RUNS)]
            if any(state["ActiveState"] == "failed" for state in states):
                raise RuntimeError(f"Training job failed: {states}")
            if all(state["ActiveState"] == "inactive" for state in states):
                if any(state["Result"] != "success" or state["ExecMainStatus"] != "0" for state in states):
                    raise RuntimeError(f"Training did not finish successfully: {states}")
                break
            time.sleep(30)
        # Allow the final checkpoint's write and filesystem metadata to settle.
        time.sleep(12)
        rigid = latest_checkpoint(RUNS[1][2])
        if rigid is None:
            raise RuntimeError("Missing final rigid training checkpoint")
        if rigid_only:
            checks = []
            for name, speed in (("stand", "0.0"), ("walk", "0.3"), ("maximum", "0.5")):
                checks.append(
                    evaluate(
                        f"rigid_gait_{name}_validation",
                        rigid,
                        "--field",
                        "rigid",
                        "--steps",
                        "1000",
                        "--num_envs",
                        "16",
                        "--speed",
                        speed,
                    )
                )
            for seed in (123, 456, 789):
                checks.append(
                    evaluate(
                        f"rigid_gait_commands_{seed}",
                        rigid,
                        "--field",
                        "rigid",
                        "--steps",
                        "500",
                        "--num_envs",
                        "128",
                        "--randomize",
                        seed=seed,
                    )
                )
            video = evaluate(
                "rigid_gait_control", rigid, "--field", "rigid", "--steps", "1700", "--command_schedule", "--video"
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
            progress["rigid_validation_gates"] = gates
            progress["rigid_control_accepted"] = all(gates.values())
            publish("Rigid-control validation passed" if all(gates.values()) else "Rigid-control validation failed")
            subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
            return
        rft = latest_checkpoint(RUNS[0][2])
        if rft is None:
            raise RuntimeError("Missing final RFT training checkpoint")
        events = EventAccumulator(str(rft.parent)).Reload()
        achieved = events.Scalars("Curriculum/difficulty")[-1].value
        progress["achieved_material_difficulty"] = achieved
        evaluate("corrected_rft_curriculum_nominal", rft, "--field", "rft", "--steps", "500", "--video")
        for name, difficulty in (("material_achieved_validation", achieved), ("material_full_validation", 1.0)):
            evaluate(
                name,
                rft,
                "--field",
                "rft",
                "--steps",
                "500",
                "--num_envs",
                "128",
                "--randomize",
                "--difficulty",
                str(difficulty),
            )
        evaluate("corrected_rigid_final", rigid, "--field", "rigid", "--steps", "500", "--video")
        evaluate(
            "corrected_zero_layer_policy",
            rigid,
            "--field",
            "rft",
            "--substrate_depth",
            "0",
            "--steps",
            "500",
            "--video",
        )
        for name, depth in (("005", "0.005"), ("020", "0.02"), ("050", "0.05")):
            evaluate(
                f"corrected_thin_layer_{name}",
                rigid,
                "--field",
                "rft",
                "--substrate_depth",
                depth,
                "--steps",
                "500",
                "--video",
            )
        evaluate("corrected_rft_curriculum_mpm", rft, "--field", "mpm", "--steps", "100", "--video")
        publish("Building final plots and video gallery")
        subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
        publish("Evaluation complete; inspect held-out results and MPM limitations before claiming convergence")
    except Exception as error:
        progress["error"] = repr(error)
        publish("Evaluation stopped with an error; inspect the validation service log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rigid-only", action="store_true", help="Validate the repaired rigid control before RFT work."
    )
    main(parser.parse_args().rigid_only)
