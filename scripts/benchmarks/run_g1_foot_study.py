# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run G1 foot convergence fixtures serially and publish their numerical evidence."""

import argparse
import ast
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("logs/go2_granular/g1/refinement")


def compare(first, reference):
    """Compare phase-mean wrenches with fixed absolute floors near zero load."""
    with np.load(first / "trace.npz") as a, np.load(reference / "trace.npz") as b:
        phase_a, phase_b = a["phase"], b["phase"]
        force_a, force_b = a["mpm_wrench"], b["mpm_wrench"]
        names = json.loads((reference / "summary.json").read_text())["phases"]
        rows = []
        for i, phase in enumerate(names):
            if phase["name"] in ("preparation", "released"):
                continue
            x, y = force_a[phase_a == i].mean(0), force_b[phase_b == i].mean(0)
            rows.append(
                dict(
                    phase=phase["name"],
                    force_error=float(np.linalg.norm(x[:3] - y[:3]) / max(np.linalg.norm(y[:3]), 20)),
                    torque_error=float(np.linalg.norm(x[3:] - y[3:]) / max(np.linalg.norm(y[3:]), 1)),
                )
            )
        return dict(
            phases=rows,
            passed=all(max(r["force_error"], r["torque_error"]) < 0.1 for r in rows),
            force_target=0.1,
            torque_target=0.1,
            force_normalization_floor_N=20,
            torque_normalization_floor_Nm=1,
            interpretation=(
                "Phase-mean self-convergence, with 2 N / 0.1 Nm absolute allowances near zero. Not physical validation."
            ),
        )


def publish(status):
    """Expose progress and paired force/torque traces without altering baseline results."""
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "status.tmp").write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "status.tmp").replace(ROOT / "status.json")
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for result in status["results"]:
        name = result["name"]
        folder = ROOT / "probes" / name
        s = result["summary"]
        with np.load(folder / "trace.npz") as data:
            for j, ax in enumerate(axes.flat):
                ax.plot(data["statistics"][:, 0], data["mpm_wrench"][:, j], label=name)
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{s['particle_count']:,}</td>"
            f"<td>{s['wall_seconds']:.1f}</td><td>{s['settled_rms_speed_m_s']:.4g}</td>"
            f"<td>{s['momentum_balance_relative']:.4g}</td></tr>"
        )
    for ax, title in zip(axes.flat, ("Fx [N]", "Fy [N]", "Fz [N]", "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"), strict=True):
        ax.set(xlabel="Time [s]", ylabel=title)
        ax.grid(alpha=0.2)
    if status["results"]:
        axes[0, 0].legend(fontsize=6)
    fig.suptitle("G1 foot: identical insertion, hold, drag, pitch, roll and withdrawal")
    fig.tight_layout()
    fig.savefig(ROOT / "foot-convergence.png", dpi=150)
    plt.close(fig)
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:30px auto;padding:20px}"
        "img{width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>"
        '<h1>G1 force-model investigation</h1><p><a href="../report/">Baseline policy report</a> · '
        '<a href="training.html">Retraining status and curves</a></p>'
        '<p><a href="analysis.html">Force calibration and loaded-foot response</a></p>'
        f"<p>{html.escape(status['stage'])}</p>"
        "<p>Exact finalized G1 foot mesh, prescribed ankle poses, all substep collider impulses. "
        "No policy or rigid integrator. The rigid bottom acts on material only. "
        "Comparisons separate numerical convergence from RFT calibration.</p>"
        "<table><tr><th>Case</th><th>Particles</th><th>Simulation wall time [s]</th>"
        "<th>Settled RMS speed [m/s]</th><th>Momentum residual / weight impulse</th></tr>"
        + "".join(rows)
        + '</table><img src="foot-convergence.png">'
        "<h2>Refinement checks</h2><pre>" + html.escape(json.dumps(status.get("checks", {}), indent=2)) + "</pre>"
    )
    (ROOT / "index.tmp").write_text(page)
    (ROOT / "index.tmp").replace(ROOT / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    profiles = {
        "apic-10mm-8": dict(voxel_size=0.01, substeps=8, iterations=200),
        "pic-20mm-4": dict(voxel_size=0.02, substeps=4, iterations=50, transfer_scheme="pic"),
        "apic-5mm-8": dict(voxel_size=0.005, substeps=8, iterations=200),
        "apic-5mm-16": dict(voxel_size=0.005, substeps=16, iterations=200),
        "apic-5mm-16-400": dict(voxel_size=0.005, substeps=16, iterations=400),
        "apic-10mm-8-wide": dict(voxel_size=0.01, substeps=8, iterations=200, domain_scale=1.5),
        "apic-8mm-8": dict(voxel_size=0.008, substeps=8, iterations=200),
        "apic-3p33mm-8": dict(voxel_size=1 / 300, substeps=8, iterations=200),
        "apic-5mm-8-ppc1": dict(voxel_size=0.005, substeps=8, iterations=200, particles_per_cell=1),
        "apic-5mm-16-ppc1": dict(voxel_size=0.005, substeps=16, iterations=200, particles_per_cell=1),
        "apic-8mm-8-ppc1": dict(voxel_size=0.008, substeps=8, iterations=200, particles_per_cell=1),
        "apic-20mm-8-s2": dict(voxel_size=0.02, substeps=8, iterations=200, collider_basis="S2"),
        "apic-10mm-8-s2": dict(voxel_size=0.01, substeps=8, iterations=200, collider_basis="S2"),
        "apic-5mm-8-s2": dict(voxel_size=0.005, substeps=8, iterations=200, collider_basis="S2"),
    }
    selected = args.cases or list(profiles)
    worker = Path("scripts/benchmarks/investigate_g1_granular.py")
    prior = json.loads((ROOT / "status.json").read_text()).get("results", []) if (ROOT / "status.json").exists() else []
    status = dict(
        stage="Starting foot comparisons",
        results=[row for row in prior if row["name"] not in selected],
        checks={},
        complete=False,
    )
    for name in selected:
        options = profiles[name]
        folder = ROOT / "probes" / name
        folder.mkdir(parents=True, exist_ok=True)
        done = sum(row["name"] in selected for row in status["results"])
        status["stage"] = f"Running {name}: {done}/{len(selected)} requested cases complete"
        publish(status)
        record = dict(
            settings=options,
            source_ast=hashlib.sha256(ast.dump(ast.parse(worker.read_text())).encode()).hexdigest(),
            geometry_sha256=hashlib.sha256((ROOT / "geometry.json").read_bytes()).hexdigest(),
        )
        path = folder / "case.json"
        if not (folder / "summary.json").exists() or not path.exists() or json.loads(path.read_text()) != record:
            path.write_text(json.dumps(record, indent=2) + "\n")
            command = [
                sys.executable,
                str(worker),
                "--geometry_file",
                str(ROOT / "geometry.json"),
                "--output",
                str(folder),
            ]
            for key, value in options.items():
                command += ["--" + key, str(value)]
            with (folder / "stdout.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["results"].append(dict(name=name, summary=json.loads((folder / "summary.json").read_text())))
        for label, a, b in (
            ("grid", "apic-10mm-8", "apic-5mm-8"),
            ("time", "apic-5mm-8", "apic-5mm-16"),
            ("iterations", "apic-5mm-16", "apic-5mm-16-400"),
            ("domain", "apic-10mm-8", "apic-10mm-8-wide"),
            ("grid_8mm", "apic-8mm-8", "apic-5mm-8"),
            ("grid_fine", "apic-5mm-8", "apic-3p33mm-8"),
            ("particle_sampling", "apic-5mm-8", "apic-5mm-8-ppc1"),
            ("particle_time", "apic-5mm-8-ppc1", "apic-5mm-16-ppc1"),
            ("grid_8mm_ppc1", "apic-8mm-8-ppc1", "apic-5mm-8-ppc1"),
            ("s2_20mm", "apic-20mm-8-s2", "apic-3p33mm-8"),
            ("s2_10mm", "apic-10mm-8-s2", "apic-3p33mm-8"),
            ("s2_5mm", "apic-5mm-8-s2", "apic-3p33mm-8"),
        ):
            if all((ROOT / "probes" / p / "summary.json").exists() for p in (a, b)):
                status["checks"][label] = compare(ROOT / "probes" / a, ROOT / "probes" / b)
        publish(status)
    status.update(stage="Foot convergence runs completed; inspect gates before accepting calibration", complete=True)
    publish(status)


if __name__ == "__main__":
    main()
