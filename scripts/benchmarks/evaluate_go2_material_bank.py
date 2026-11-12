# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate existing Go2 checkpoints under recorded MPM coupling and material variations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tomllib

ROOT = Path("logs/go2_granular/investigation")
CHECKPOINTS = {
    "rigid": Path("logs/rsl_rl/go2_rigid_gait_control/2026-11-11_00-02-21/model_599.pt"),
    "rft": Path("logs/rsl_rl/go2_rft_knee_depth_curriculum/2026-11-11_02-42-39/model_1499.pt"),
}


def coupling_cases():
    """Change one coupling control relative to the settled baseline."""
    return [
        ("baseline-audit", "rft", dict(proxy_relaxation=0.1, proxy_iterations=1, audit_forces=True)),
        ("relaxation-0.3", "rft", dict(proxy_relaxation=0.3, proxy_iterations=1)),
        ("relaxation-1", "rft", dict(proxy_relaxation=1.0, proxy_iterations=1)),
        ("iterations-2", "rft", dict(proxy_relaxation=0.1, proxy_iterations=2)),
        ("iterations-4", "rft", dict(proxy_relaxation=0.1, proxy_iterations=4)),
        ("substeps-8", "rft", dict(proxy_relaxation=0.1, proxy_iterations=1, substeps=8)),
        ("substeps-16", "rft", dict(proxy_relaxation=0.1, proxy_iterations=1, substeps=16)),
        ("relaxation-1-iterations-4", "rft", dict(proxy_relaxation=1.0, proxy_iterations=4)),
    ]


def material_cases(profile, reference=False):
    """Held-out settings inside the requested RFT parameter bounds; keep both policies frozen."""
    cases = []
    for material, values in (
        ("lower-density-friction", dict(density=1400, friction=0.3, roughness=0.6, bed_depth=0.06)),
        ("nominal", dict(density=1600, friction=0.4, roughness=1.0, bed_depth=0.118)),
        ("higher-density-friction", dict(density=1800, friction=0.5, roughness=0.8, bed_depth=0.10)),
    ):
        for policy in ("rft",) if reference else ("rigid", "rft"):
            options = profile | values | dict(audit_forces=True)
            if reference:
                options.update(field="rft", substrate_depth=values["bed_depth"])
            cases.append((f"{material}-{policy}", policy, options))
    return cases


def publish(progress, phase, robot="go2"):
    """Publish progress independently of the native-probe worker."""
    (ROOT / f"{phase}-status.json").write_text(json.dumps(progress, indent=2))
    robot_name = "G1" if robot == "g1" else "Go2"
    report_url = "/g1/report/" if robot == "g1" else "/report/"
    numerics_url = f"{report_url}#mpm-numerics" if robot == "g1" else "/investigation/analysis.html"
    rows = []
    for case in progress["results"]:
        s, c = case["summary"], case["settings"]
        failure = s.get("first_failure_seconds")
        rows.append(
            f"<tr><td>{html.escape(case['name'])}</td><td>{case['policy']}</td>"
            f"<td>{c.get('proxy_relaxation', 0.1)}</td><td>{c.get('proxy_iterations', 1)}</td>"
            f"<td>{c.get('substeps', 4)}</td><td>{s.get('mean_velocity_error', float('nan')):.3f}</td>"
            f"<td>{failure}</td><td>{s.get('particles_finite')}</td>"
            f"<td>{s.get('mean_external_vertical_N', float('nan')):.1f} / "
            f"{s.get('mean_rigid_contact_vertical_N', float('nan')):.1f}</td></tr>"
        )
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:20px}"
        "table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:8px}img{width:100%}</style>"
        f"<h1>{robot_name}: {html.escape(phase)} investigation</h1>"
        f'<p><a href="{numerics_url}">Material-model numerics</a> · '
        f'<a href="{report_url}">Policy videos</a></p>'
        f"<p>{html.escape(progress['stage'])}: {len(progress['results'])} / {progress['total']} cases.</p>"
        f"<p>Existing frozen policies; {progress.get('command_speed', 0.5):.2f} m/s forward command. "
        "The bed is prepared with the robot held fixed. "
        "Force audit values are whole-robot external body loads and rigid-contact resultants "
        "inferred from momentum balance.</p>"
        "<table><tr><th>Case</th><th>Policy</th><th>Relaxation</th><th>Passes</th><th>Substeps</th>"
        "<th>Tracking error [m/s]</th><th>Failure time [s]</th><th>Particles finite</th>"
        "<th>Granular / rigid support [N]</th></tr>" + "".join(rows) + "</table>"
    )
    (ROOT / f"{phase}.html").write_text(page)


def main(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.profile.read_text()) if args.profile else {}
    checkpoints = CHECKPOINTS | {
        key: path for key, path in (("rigid", args.checkpoint_rigid), ("rft", args.checkpoint_rft)) if path is not None
    }
    cases = coupling_cases() if args.phase == "coupling" else material_cases(profile, args.phase == "reference")
    progress = dict(stage="Running frozen-policy comparisons", total=len(cases), results=[], command_speed=args.speed)
    worker = Path("scripts/benchmarks/evaluate_go2_granular.py")
    source_hash = hashlib.sha256(ast.dump(ast.parse(worker.read_text())).encode()).hexdigest()
    dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
    newton_requirement = next(value for value in dependencies if value.startswith("newton[sim]"))
    publish(progress, args.phase, args.robot)
    for name, policy, options in cases:
        folder = ROOT / args.phase / name
        folder.mkdir(parents=True, exist_ok=True)
        settings = (
            dict(
                steps=100 if args.phase == "coupling" else 150,
                speed=args.speed,
                max_forward_command=args.max_forward_command,
                bed_depth=0.118,
                bed_length=2.0 if args.phase == "coupling" else 3.0,
                settle_seconds=0.5,
            )
            | options
        )
        if args.bed_length is not None:
            settings["bed_length"] = args.bed_length
        settings["bed_width"] = args.bed_width
        field = settings.pop("field", "mpm")
        manifest = dict(
            settings=settings,
            robot=args.robot,
            field=field,
            checkpoint=str(checkpoints[policy]),
            source_ast_sha256=source_hash,
            newton_requirement=newton_requirement,
        )
        record = folder / "case.json"
        cached = (folder / "summary.json").exists() and record.exists() and json.loads(record.read_text()) == manifest
        if not cached:
            record.write_text(json.dumps(manifest, indent=2))
            command = [
                sys.executable,
                str(worker),
                "--field",
                field,
                "--seed",
                "123",
                "--robot",
                args.robot,
                "--checkpoint",
                str(checkpoints[policy]),
                "--policy_training_terrain",
                policy,
                "--output",
                str(folder),
            ]
            for key, value in settings.items():
                command.append("--" + key)
                if value is not True:
                    command.append(str(value))
            print(json.dumps(dict(stage="running", case=name, settings=settings)), flush=True)
            with (folder / "stdout.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not (folder / "summary.json").exists():
                progress.update(stage=f"Evaluation stopped: {name}; inspect its stdout.log")
                publish(progress, args.phase, args.robot)
                raise RuntimeError(f"{name}: execution failure")
        summary = json.loads((folder / "summary.json").read_text())
        with np.load(folder / "rollout.npz") as data:
            support = data["support_wrenches"]
            if len(support):
                after_transient = support[min(25, len(support) // 2) :]
                summary["mean_external_vertical_N"] = float(after_transient[:, 2].mean())
                summary["mean_rigid_contact_vertical_N"] = float(after_transient[:, 5].mean())
        progress["results"].append(dict(name=name, policy=policy, settings=settings, summary=summary))
        progress["updated_unix"] = time.time()
        publish(progress, args.phase, args.robot)
        print(
            json.dumps(
                dict(
                    stage="completed",
                    case=name,
                    error=summary.get("mean_velocity_error"),
                    failure=summary.get("first_failure_seconds"),
                    finite=summary.get("particles_finite"),
                )
            ),
            flush=True,
        )
    progress["stage"] = "Cases completed; inspect numerical sensitivity and failure checks before accepting transfer"
    publish(progress, args.phase, args.robot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go2", "g1"), default="go2")
    parser.add_argument("--phase", choices=("coupling", "materials", "reference"), default="coupling")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output_root", type=Path, default=ROOT)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--max_forward_command", type=float, default=0.5)
    parser.add_argument("--bed_length", type=float)
    parser.add_argument("--bed_width", type=float, default=1.2)
    parser.add_argument("--checkpoint_rigid", type=Path)
    parser.add_argument("--checkpoint_rft", type=Path)
    args = parser.parse_args()
    ROOT = args.output_root
    main(args)
