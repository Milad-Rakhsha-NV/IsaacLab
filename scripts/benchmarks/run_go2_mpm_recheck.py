# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finish the paired-speed Go2 study with controlled MPM startup sensitivity checks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from monitor_go2_granular import ROOT, load_study
from report_go2_materials import policy_report
from run_go2_speed_comparison import write_json
from validate_go2_training import job_state


def main() -> None:
    """Keep policy, bed and command fixed while changing one numerical control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--startup_only", action="store_true", help="Run the short checks before the video queue resumes."
    )
    args = parser.parse_args()
    if not args.startup_only:
        while job_state("go2-paired-speed-evaluation.service")["ActiveState"] in ("active", "activating"):
            time.sleep(30)
        state = job_state("go2-paired-speed-evaluation.service")
        if state.get("Result") != "success":
            raise RuntimeError(f"Paired-speed video generation failed: {state}")
    study = load_study()
    root = ROOT / study["case_directory"]
    checkpoints = {key: Path(folder) / "model_1499.pt" for key, folder in study["run_directories"].items()}
    status_path = ROOT / study["status_file"]
    progress = json.loads(status_path.read_text())
    progress["stage"] = "Checking MPM time-step and coupling sensitivity before final comparison publication"
    write_json(status_path, progress)
    out = root / "mpm-recheck"
    out.mkdir(exist_ok=True)
    results = []
    for name, substeps, passes in (
        ("baseline", 4, 1),
        ("substeps-8", 8, 1),
        ("substeps-16", 16, 1),
        ("coupling-passes-4", 4, 4),
    ):
        folder = out / name
        folder.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "scripts/benchmarks/evaluate_go2_granular.py",
            "--field",
            "mpm",
            "--policy_training_terrain",
            "rft",
            "--checkpoint",
            str(checkpoints["rft"]),
            "--output",
            str(folder),
            "--steps",
            "100",
            "--speed",
            "1.0",
            "--max_forward_command",
            "1.0",
            "--seed",
            "123",
            "--bed_depth",
            "0.118",
            "--bed_length",
            "5",
            "--bed_width",
            "1.6",
            "--settle_seconds",
            "0.5",
            "--substeps",
            str(substeps),
            "--proxy_iterations",
            str(passes),
            "--audit_forces",
        ]
        worker = Path("scripts/benchmarks/evaluate_go2_granular.py")
        manifest = dict(
            command=command, worker_ast=hashlib.sha256(ast.dump(ast.parse(worker.read_text())).encode()).hexdigest()
        )
        metadata = folder / "case.json"
        cached = (
            metadata.exists() and (folder / "summary.json").exists() and json.loads(metadata.read_text()) == manifest
        )
        if cached:
            cached = json.loads((folder / "summary.json").read_text())["execution_error"] is None
        if not cached:
            write_json(metadata, manifest)
            with (folder / "stdout.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((folder / "summary.json").read_text())
        if summary["execution_error"] or not summary["particles_finite"] or summary["max_rigid_overflow"]:
            raise RuntimeError(f"Invalid MPM recheck {name}: {summary['execution_error']}")
        results.append(dict(name=name, substeps=substeps, coupling_passes=passes, summary=summary))
        write_json(out / "results.json", dict(cases=results))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for case in results:
        with np.load(out / case["name"] / "rollout.npz") as data:
            s = data["state"][:, 0]
            metrics = data["metrics"][:, 0]
            support = data["support_wrenches"]
            for ax, values in zip(
                axes.flat, (s[:, 7], s[:, 4], metrics[:, 3], support[:, 2] + support[:, 5]), strict=True
            ):
                ax.plot(s[:, 0], values, label=case["name"])
    reference = root / "rft_max_forward/rollout.npz"
    if reference.exists():
        with np.load(reference) as data:
            state = data["state"][:100, 0]
            tilt = data["metrics"][:100, 0, 3]
            for ax, values in zip(axes.flat, (state[:, 7], state[:, 4], tilt), strict=False):
                ax.plot(state[:, 0], values, "k--", label="RFT reference")
    axes[0, 0].axhline(0.16, color="grey", linestyle=":")
    for ax, label in zip(
        axes.flat,
        (
            "Height above initial surface [m]",
            "Forward speed [m/s]",
            "Torso tilt [rad]",
            "Whole-robot vertical support [N]",
        ),
        strict=True,
    ):
        ax.set(xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen RFT policy in MPM · 1.0 m/s · identical prepared 118 mm bed")
    fig.tight_layout()
    fig.savefig(ROOT / "report/mpm-startup-recheck.png", dpi=150)
    plt.close(fig)
    if args.startup_only:
        write_json(out / "startup-complete.json", dict(completed_unix=time.time()))
        return
    material_root = root / "materials_cmd_0p5"
    for phase in ("materials", "reference"):
        command = [
            sys.executable,
            "scripts/benchmarks/evaluate_go2_material_bank.py",
            "--phase",
            phase,
            "--output_root",
            str(material_root),
            "--speed",
            "0.5",
            "--max_forward_command",
            "1.0",
            "--bed_length",
            "5",
            "--bed_width",
            "1.6",
            "--checkpoint_rigid",
            str(checkpoints["rigid"]),
            "--checkpoint_rft",
            str(checkpoints["rft"]),
        ]
        with (root / f"{phase}_cmd_0p5.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    policy_report(material_root, material_root, speed=0.5)
    study = load_study()
    study["half_speed_material_comparison_ready"] = True
    study["mpm_recheck_ready"] = True
    write_json(ROOT / "current-study.json", study)
    progress = json.loads(status_path.read_text())
    progress["stage"] = "Both speeds: all eight videos, plots and controlled MPM startup checks completed"
    write_json(status_path, progress)
    subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
    write_json(root / "followup-complete.json", dict(completed_unix=time.time(), mpm_recheck_cases=len(results)))


if __name__ == "__main__":
    main()
