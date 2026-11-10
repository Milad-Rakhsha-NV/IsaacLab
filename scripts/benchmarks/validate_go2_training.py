# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record an early Go2 preview, then evaluate the frozen policies when training finishes.

Uses the two training services and run families from ``monitor_go2_granular.py``.
MPM runs only after both training jobs have exited successfully. Progress is
published for the live dashboard; physical failures remain in the recordings.
"""

from __future__ import annotations

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


def main() -> None:
    progress = dict(stage="Waiting for checkpoint 100 preview", cases=[])

    def publish(stage: str) -> None:
        progress["stage"] = stage
        progress["updated_unix"] = time.time()
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.with_suffix(".tmp").write_text(json.dumps(progress, indent=2))
        STATUS.with_suffix(".tmp").replace(STATUS)

    def evaluate(name: str, checkpoint: Path, *options: str) -> None:
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
            "123",
            *options,
        ]
        with (STATUS.parent / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / "summary.json").read_text())
        if summary["execution_error"] is not None:
            raise RuntimeError(f"{name}: {summary['execution_error']}")
        progress["cases"].append(dict(case=name, **summary))
        publish(f"Finished {name}")

    publish(progress["stage"])
    try:
        preview_done = False
        while True:
            states = [job_state(row[3]) for row in RUNS]
            if any(state["ActiveState"] == "failed" for state in states):
                raise RuntimeError(f"Training job failed: {states}")
            checkpoint = latest_checkpoint(RUNS[0][2])
            if not preview_done and checkpoint and int(checkpoint.stem.split("_")[-1]) >= 100:
                evaluate("material_rft_preview", checkpoint, "--field", "rft", "--steps", "500", "--video")
                preview_done = True
                publish("Training continues; waiting to run final frozen-policy evaluations")
            if all(state["ActiveState"] == "inactive" for state in states):
                if any(state["Result"] != "success" or state["ExecMainStatus"] != "0" for state in states):
                    raise RuntimeError(f"Training did not finish successfully: {states}")
                break
            time.sleep(30)
        # Allow the final checkpoint's write and filesystem metadata to settle.
        time.sleep(12)
        rft = latest_checkpoint(RUNS[0][2])
        rigid = latest_checkpoint(RUNS[1][2])
        if rft is None or rigid is None:
            raise RuntimeError("Missing final training checkpoint")
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
    main()
