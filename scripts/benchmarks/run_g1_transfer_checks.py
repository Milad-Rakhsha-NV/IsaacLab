# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate frozen G1 continuations without using MPM transitions for training."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from imageio_ffmpeg import count_frames_and_secs
from validate_go2_training import validation_gates

ROOT = Path("logs/go2_granular/g1/refinement")
PYTHONS = {
    "control": Path(".venv/bin/python").absolute(),
    "matched": Path("/home/horde/.cache/rft-study/g1-fixed-venv/bin/python"),
    "rigid": Path("/home/horde/.cache/rft-study/g1-fixed-venv/bin/python"),
}
MATERIALS = {
    "nominal": (0.118, 1600.0, 0.4),
    "shallow_loose": (0.060, 1400.0, 0.3),
    "dense": (0.118, 1800.0, 0.5),
}


def write(path, record):
    """Atomically publish serial evaluation outcomes."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main(args):
    """Reuse only identical completed cases, keeping videos off the live page until verified."""
    PYTHONS["control"] = args.control_python.absolute()
    PYTHONS["matched"] = PYTHONS["rigid"] = args.corrected_python.absolute()
    state_path = ROOT / "evaluation-status.json"
    status = json.loads(state_path.read_text()) if state_path.exists() else dict(cases=[], validation={})
    training = json.loads((ROOT / "training-status.json").read_text())
    baseline = json.loads(Path("logs/go2_granular/g1/status.json").read_text())
    checkpoints = {"rigid": Path(baseline["checkpoints"]["rigid"])}
    for policy in ("control", "matched"):
        record = training["runs"].get(policy, {})
        if "checkpoint" in record:
            checkpoints[policy] = Path(record["checkpoint"])

    def evaluate(name, policy, field, options, label, video=False):
        folder = ROOT / "evaluations" / name
        folder.mkdir(parents=True, exist_ok=True)
        worker = Path("scripts/benchmarks/evaluate_go2_granular.py")
        command = [
            str(PYTHONS[policy]),
            str(worker),
            "--robot",
            "g1",
            "--field",
            field,
            "--checkpoint",
            str(checkpoints[policy]),
            "--policy_training_terrain",
            "rigid" if policy == "rigid" else "rft",
            "--output",
            str(folder),
            "--max_forward_command",
            "1.0",
            "--seed",
            "123",
            *options,
        ]
        if field == "rft" and policy == "matched":
            command += ["--rft_profile", str(ROOT / "rft-profile.json")]
        signature = dict(
            command=command,
            checkpoint_sha256=hashlib.sha256(checkpoints[policy].read_bytes()).hexdigest(),
            worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
            profile=json.loads((ROOT / "rft-profile.json").read_text()) if policy == "matched" else None,
        )
        cached = (folder / "summary.json").exists() and (folder / "case.json").exists()
        cached = cached and json.loads((folder / "case.json").read_text()) == signature
        if not cached:
            status.update(stage=label, active_case=name, complete=False)
            write(state_path, status)
            write(folder / "case.json", signature)
            with (folder / "stdout.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((folder / "summary.json").read_text())
        if summary["execution_error"] or not summary["particles_finite"] or summary["max_rigid_overflow"]:
            raise RuntimeError(f"Invalid numerical evaluation: {name}")
        if video:
            frames, seconds = count_frames_and_secs(folder / "rollout.mp4")
            assert frames == summary["steps"] and abs(seconds - summary["simulated_seconds"]) < 0.1
            with np.load(folder / "rollout.npz") as data:
                speed = summary["commands"][0][0]
                np.testing.assert_array_equal(data["commands"], np.broadcast_to([speed, 0, 0], data["commands"].shape))
            if field == "mpm" and summary["first_bed_exit_seconds"] is not None:
                raise RuntimeError(f"Recording reaches material boundary: {name}")
        row = dict(name=name, policy=policy, field=field, summary=summary, label=label, video=video)
        status["cases"] = [r for r in status["cases"] if r["name"] != name] + [row]
        status.pop("active_case", None)
        status["stage"] = "Completed: " + label
        write(state_path, status)
        return summary

    def mpm_options(material, length, steps, speed):
        depth, density, friction = MATERIALS[material]
        return [
            "--steps",
            str(steps),
            "--speed",
            str(speed),
            "--bed_depth",
            str(depth),
            "--bed_length",
            str(length),
            "--bed_width",
            "2.0",
            "--density",
            str(density),
            "--friction",
            str(friction),
            "--roughness",
            "1.0",
            "--voxel_size",
            "0.02",
            "--substeps",
            "8",
            "--mpm_iterations",
            "200",
            "--transfer_scheme",
            "apic",
            "--proxy_relaxation",
            "1.0",
            "--settle_seconds",
            "0.5",
            "--failure_tail_seconds",
            "0.5",
        ]

    if args.phase in ("control", "matched"):
        policy = args.phase
        checks = []
        for label, speed in (("stand", 0.0), ("walk", 0.5), ("maximum", 1.0)):
            checks.append(
                evaluate(
                    f"{policy}_{label}",
                    policy,
                    "rft",
                    ["--substrate_depth", "0.118", "--steps", "1000", "--num_envs", "16", "--speed", str(speed)],
                    f"{label.capitalize()} on RFT sand at {speed:.2f} m/s",
                )
            )
        for seed in (123, 456, 789):
            checks.append(
                evaluate(
                    f"{policy}_random_{seed}",
                    policy,
                    "rft",
                    [
                        "--substrate_depth",
                        "0.118",
                        "--steps",
                        "500",
                        "--num_envs",
                        "128",
                        "--randomize",
                        "--seed",
                        str(seed),
                    ],
                    f"128 independently randomized RFT environments; reset seed {seed}",
                )
            )
        status["validation"][policy] = validation_gates(checks, material_difficulty=1.0)
        write(state_path, status)
    elif args.phase == "timestep":
        for policy in args.policies:
            field = "rigid" if policy == "rigid" else "rft"
            depth = ["--substrate_depth", "0.118"] if field == "rft" else []
            for speed in (0.5, 1.0):
                evaluate(
                    f"{policy}_timestep8_{speed:g}",
                    policy,
                    field,
                    [*depth, "--steps", "1000", "--num_envs", "16", "--speed", str(speed), "--substeps", "8"],
                    f"{field.upper()} at the MPM rigid timestep: 8 substeps; {speed:.2f} m/s",
                )
    elif args.phase == "reference":
        for voxel in (0.02, 0.01, 0.008):
            evaluate(
                f"matched_reference_{1000 * voxel:g}mm",
                "matched",
                "mpm",
                [*mpm_options("nominal", 2.4, 100, 0.5), "--bed_width", "0.8", "--voxel_size", str(voxel)],
                f"Whole-robot MPM grid sensitivity: {1000 * voxel:g} mm; 2.4 × 0.8 m bed; 0.50 m/s",
            )
    elif args.phase == "mpm":
        for material in MATERIALS:
            for speed in (0.5, 1.0):
                for policy in args.policies:
                    video = material == "nominal" and policy in ("matched", "rigid")
                    options = mpm_options(material, 5, 150, speed)
                    if video:
                        options.append("--video")
                    evaluate(
                        f"{policy}_mpm_{material}_{speed:g}",
                        policy,
                        "mpm",
                        options,
                        f"MPM {material.replace('_', ' ')} material at {speed:.2f} m/s (3 s)",
                        video=video,
                    )
    else:
        for policy, field in (("rigid", "rigid"), ("matched", "rft")):
            for speed in (0.5, 1.0):
                options = ["--steps", "500", "--speed", str(speed), "--video"]
                if field == "rft":
                    options += ["--substrate_depth", "0.118"]
                evaluate(
                    f"{policy}_{field}_video_{speed:g}",
                    policy,
                    field,
                    options,
                    f"{field.upper()} recording at {speed:.2f} m/s",
                    video=True,
                )
    status.update(stage=f"{args.phase.capitalize()} evaluation phase completed", complete=True)
    write(state_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("control", "matched", "timestep", "reference", "mpm", "videos"), required=True
    )
    parser.add_argument("--policies", nargs="+", choices=tuple(PYTHONS), default=list(PYTHONS))
    parser.add_argument("--control_python", type=Path, default=PYTHONS["control"])
    parser.add_argument("--corrected_python", type=Path, default=PYTHONS["matched"])
    main(parser.parse_args())
