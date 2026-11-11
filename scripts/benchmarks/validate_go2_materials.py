# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run and publish reproducible local MPM/RFT numerical investigations."""

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

ROOT = Path("logs/go2_granular/investigation")


def numerical_cases():
    """Vary resolution, time step, solve budget, projection and domain separately."""
    cases = []
    for mode in ("vertical", "drag"):
        for grid in (0.02, 0.01, 0.005):
            cases.append((f"sphere-{mode}-grid-{grid}", dict(mode=mode, voxel_size=grid)))
        for substeps in (8, 16):
            cases.append((f"sphere-{mode}-time-{substeps}", dict(mode=mode, voxel_size=0.01, substeps=substeps)))
        for iterations in (100, 200):
            cases.append(
                (
                    f"sphere-{mode}-iterations-{iterations}",
                    dict(mode=mode, voxel_size=0.01, substeps=8, iterations=iterations),
                )
            )
        for label, extra in (("projection", dict(project=True)), ("domain", dict(domain_scale=1.5))):
            cases.append(
                (f"sphere-{mode}-{label}", dict(mode=mode, voxel_size=0.01, substeps=8, iterations=200, **extra))
            )
        for grid in (0.02, 0.01, 0.005):
            cases.append(
                (
                    f"plate-{mode}-grid-{grid}",
                    dict(geometry="plate", mode=mode, voxel_size=grid, substeps=8, iterations=200),
                )
            )
    return cases


def transfer_cases():
    """Check whether affine-preserving transfer reduces the observed PIC time sensitivity."""
    cases = []
    for mode in ("vertical", "drag"):
        for substeps in (8, 16, 32):
            cases.append(
                (
                    f"sphere-{mode}-apic-time-{substeps}",
                    dict(mode=mode, voxel_size=0.01, substeps=substeps, iterations=200, transfer_scheme="apic"),
                )
            )
        cases.append(
            (
                f"sphere-{mode}-apic-fine",
                dict(mode=mode, voxel_size=0.005, substeps=16, iterations=200, transfer_scheme="apic"),
            )
        )
        for grid in (0.01, 0.005):
            cases.append(
                (
                    f"plate-{mode}-apic-grid-{grid}",
                    dict(
                        mode=mode,
                        geometry="plate",
                        voxel_size=grid,
                        substeps=16,
                        iterations=200,
                        transfer_scheme="apic",
                    ),
                )
            )
    return cases


def calibration_cases():
    """Hold out depth/rate combinations and vary materials within training bounds."""
    cases = []
    for mode in ("vertical", "drag"):
        for label, values in (
            ("shallow-slow", dict(depth=0.02, speed=0.05)),
            ("deep-fast", dict(depth=0.06, speed=0.2)),
            ("lower-material", dict(density=1400, friction=0.3, roughness=0.6, bed_depth=0.06)),
            ("higher-material", dict(density=1800, friction=0.5, roughness=0.8, bed_depth=0.10)),
        ):
            cases.append(
                (
                    f"sphere-{mode}-{label}",
                    dict(mode=mode, voxel_size=0.01, substeps=16, iterations=200, transfer_scheme="apic", **values),
                )
            )
    return cases


def rate_cases():
    """Change speed at fixed penetration, then record a complete insertion/unloading cycle."""
    cases = []
    for mode in ("vertical", "drag"):
        for speed in (0.05, 0.2):
            cases.append(
                (
                    f"sphere-{mode}-rate-{speed}",
                    dict(mode=mode, speed=speed, voxel_size=0.01, substeps=16, iterations=200, transfer_scheme="apic"),
                )
            )
    cases.append(
        (
            "sphere-vertical-unloading",
            dict(voxel_size=0.01, substeps=16, iterations=200, transfer_scheme="apic", unload=True),
        )
    )
    return cases


def publish(progress):
    """Keep an inspectable live page and machine-readable numerical records."""
    ROOT.mkdir(parents=True, exist_ok=True)
    phase = progress.get("phase", "numerics")
    status_name = "status" if phase in ("numerics", "transfer") else f"{phase}-status"
    (ROOT / f"{status_name}.tmp").write_text(json.dumps(progress, indent=2))
    (ROOT / f"{status_name}.tmp").replace(ROOT / f"{status_name}.json")
    rows = []
    for result in progress["results"]:
        s = result["summary"]
        c = s["settings"]
        force = s["mean_force_N"]
        rft = s["mean_rft_force_N"]
        rows.append(
            f"<tr><td>{html.escape(c['geometry'])} / {html.escape(c['mode'])}<br>"
            f"ρ={c['density']:g}, µ={c['friction']:g}, ratio={c['roughness']:g}, depth={1000 * c['depth']:g} mm</td>"
            f"<td>{1000 * c['voxel_size']:g}</td><td>{c['substeps']}</td><td>{c['iterations']}</td>"
            f"<td>{force[0]:.3f} / {force[2]:.3f}</td><td>{rft[0]:.3f} / {rft[2]:.3f}</td>"
            f"<td>{s['settled_rms_speed_m_s']:.4f}</td><td>{1000 * s['maximum_probe_penetration_m']:.3f}</td>"
            f"<td>{c['domain_scale']:g} / {c['project']}</td></tr>"
        )
    images = "".join(
        f"<figure><img src='{name}' alt='{label}'><figcaption>{label}</figcaption></figure>"
        for name, label in (
            ("convergence.png", "MPM numerical sensitivity"),
            ("force-response.png", "RFT and MPM force responses"),
        )
        if (ROOT / name).exists()
    )
    page = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="30"><title>Go2 material validation</title>'
        "<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:24px auto;padding:20px;"
        "background:#f1f5f9;color:#0f172a}table{border-collapse:collapse;width:100%;background:white}"
        "th,td{padding:8px;border:1px solid #cbd5e1;text-align:left}img{width:100%}a{color:#1d4ed8}</style>"
        '<h1>Go2 material validation</h1><p><a href="../report/">Policy comparison report</a></p>'
        f"<p><strong>{html.escape(progress['stage'])}</strong>: "
        f"{len(progress['results'])} / {progress['total']} cases.</p>"
        "<p>Prescribed 44 mm foot sphere and 50 mm plate. A settled 118 mm layer rests on a rigid floor, "
        "inside a small container with frictionless sidewalls. Domain size is checked separately. "
        "Forces are integrated collider impulses, compared with the RFT training closure on the same trajectory.</p>"
        "<p>These are numerical investigations. Acceptance requires refinement and material-response checks; "
        "finite particle states alone are insufficient.</p>"
        + images
        + '<div style="overflow:auto"><table><tr><th>Probe / motion</th><th>Grid [mm]</th><th>Substeps</th>'
        "<th>Iterations</th><th>MPM Fx / Fz [N]</th><th>RFT Fx / Fz [N]</th><th>Settled RMS speed [m/s]</th>"
        "<th>Max intruder penetration [mm]</th><th>Domain scale / projection</th></tr>"
        + "".join(rows)
        + '</table></div><p><a href="status.json">Settings, raw records and source hashes</a></p>'
    )
    page_name = "index" if phase in ("numerics", "transfer") and not (ROOT / "analysis.html").exists() else phase
    (ROOT / f"{page_name}.tmp").write_text(page)
    (ROOT / f"{page_name}.tmp").replace(ROOT / f"{page_name}.html")


def main(phase):
    """Execute each physical fixture in a fresh process to bound GPU memory."""
    worker = Path("scripts/benchmarks/investigate_go2_granular.py")
    source_hash = hashlib.sha256(ast.dump(ast.parse(worker.read_text())).encode()).hexdigest()
    generators = dict(numerics=numerical_cases, transfer=transfer_cases, calibration=calibration_cases, rate=rate_cases)
    cases = generators[phase]()
    prior = []
    if phase == "transfer" and (ROOT / "status.json").exists():
        prior = [r for r in json.loads((ROOT / "status.json").read_text())["results"] if "apic" not in r["name"]]
    progress = dict(
        stage="Checking MPM numerical convergence", phase=phase, total=len(prior) + len(cases), results=prior
    )
    publish(progress)
    for name, settings in cases:
        folder = ROOT / phase / name
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / "case.json"
        record = dict(settings=settings, source_ast_sha256=source_hash)
        cached = (folder / "summary.json").exists() and manifest.exists() and json.loads(manifest.read_text()) == record
        if not cached:
            manifest.write_text(json.dumps(record, indent=2))
            command = [sys.executable, str(worker), "--output", str(folder)]
            for key, value in settings.items():
                command.append("--" + key)
                if value is not True:
                    command.append(str(value))
            print(json.dumps(dict(stage="running", case=name, settings=settings)), flush=True)
            with (folder / "stdout.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                progress.update(stage=f"Case failed: {name}; inspect its stdout.log")
                publish(progress)
                raise RuntimeError(f"{name}: worker exited with {result.returncode}")
        summary = json.loads((folder / "summary.json").read_text())
        progress["results"].append(dict(name=name, summary=summary))
        progress["updated_unix"] = time.time()
        publish(progress)
        print(
            json.dumps(
                dict(stage="completed", case=name, force_N=summary["mean_force_N"], seconds=summary["wall_seconds"])
            ),
            flush=True,
        )
    progress["stage"] = "Numerical cases completed; force calibration and policy evaluation follow"
    publish(progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("numerics", "transfer", "calibration", "rate"), default="numerics")
    main(parser.parse_args().phase)
