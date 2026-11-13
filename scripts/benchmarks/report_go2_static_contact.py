# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish articulated static-reaction checks separately from historical trained policies."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEWTON_REVISION = "e73147c247a5df0a366548aaaac709e882b9f1cf"
STANCE = (
    ("projected-go2-stance-narrow", "0.02 m/s transition · 16 iterations"),
    ("projected-go2-stance-wide", "0.20 m/s transition · 16 iterations"),
    ("projected-go2-stance-4", "0.20 m/s transition · 4 iterations"),
    ("projected-go2-stance-8sub", "0.20 m/s · 4 iterations · half timestep"),
)
CASES = (
    *[name for name, _ in STANCE],
    "projected-go2-stance-weak-wide",
    "static-go2-video",
    "projected-sphere-32x4",
    "projected-sphere-at-mpm-depth",
    "projected-sphere-4x4",
    "preloaded-mpm-18N-5mm-i400",
    "static-rft-lateral-1N",
    "static-mpm-lateral-1N",
    "static-rft-lateral-10N",
    "static-mpm-lateral-10N",
    "static-mpm-lateral-10N-5mm",
    "static-rft-unload",
    "static-mpm-unload",
)


def trace(root, name):
    """Load a completed probe record."""
    return np.load(root / name / "trace.npz")["trace"]


def read(path):
    """Read a completed run or configuration record."""
    return json.loads(path.read_text())


def finish(fig, path):
    """Save a standalone figure for the portable report."""
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def copy_artifact(source, destination):
    """Allow report regeneration from either live logs or its portable archive."""
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def build(args):
    """Collect measurements, draw common-axis comparisons, and write linked provenance."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        (out / name).mkdir(exist_ok=True)
        for filename in ("trace.npz", "summary.json"):
            copy_artifact(args.input / name / filename, out / name / filename)
    for filename in ("stance.mp4", "preview.jpg"):
        copy_artifact(args.input / "static-go2-video" / filename, out / "static-go2-video" / filename)
    for live_name, archived_name in (
        ("static-contact-performance.json", "performance.json"),
        ("fixed-union-performance.json", "previous-performance.json"),
    ):
        source = args.input / live_name
        copy_artifact(source if source.exists() else args.input / archived_name, out / archived_name)
    profile_source = args.input / "profile.json"
    copy_artifact(
        profile_source if profile_source.exists() else Path("scripts/benchmarks/configs/go2_rft_static_contact.json"),
        out / "profile.json",
    )

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for name, label in STANCE:
        data = trace(out, name)
        axes[0, 0].plot(data[:, 0], 1000 * data[:, 1], label=label)
        axes[0, 1].semilogy(data[:, 0], np.maximum(data[:, 3], 1e-8), label=label)
        axes[1, 0].plot(data[:, 0], data[:, 6])
        axes[1, 1].semilogy(data[:, 0], np.maximum(data[:, 4], 1e-8))
    axes[0, 0].set(ylabel="Body height (mm)", title="Full Go2 articulation · fixed joint PD targets")
    axes[0, 1].set(ylabel="Maximum foot COM speed (m/s)", title="No rigid backing floor or bottom clipping")
    axes[1, 0].axhline(147.3364, color="black", linestyle=":", label="Robot weight")
    axes[1, 0].set(ylabel="Total vertical RFT support (N)", xlabel="Time (s)")
    axes[1, 1].set(ylabel="Projected reaction residual (m/s)", xlabel="Time (s)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend()
    finish(fig, out / "articulated-stance.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for name, label in (
        ("projected-sphere-32x4", "RFT · same shallow starting depth"),
        ("preloaded-mpm-18N-5mm-i400", "MPM · 5 mm cells · 400-iteration cap"),
        ("projected-sphere-at-mpm-depth", "RFT · initialized near settled MPM depth"),
    ):
        data = trace(out, name)
        axes[0, 0].plot(data[:, 0], 1000 * data[:, 4], label=label)
        axes[1, 0].semilogy(data[:, 0], np.maximum(abs(data[:, 2]), 1e-10))
    axes[0, 0].axhline(118, color="black", linestyle=":", label="Backing-floor depth")
    axes[0, 0].set(ylabel="Sole depth (mm)", title="18.42 N load · stationary does not mean material matched")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(ylabel="Vertical speed magnitude (m/s)", xlabel="Time after release (s)")
    for name, label in (("static-rft-unload", "RFT"), ("static-mpm-unload", "MPM · 10 mm cells")):
        data = trace(out, name)
        mask = data[:, 0] >= 0.9
        axes[0, 1].plot(data[mask, 0] - 1, 1000 * data[mask, 4], label=label)
        axes[1, 1].plot(data[mask, 0] - 1, data[mask, 5], label=label)
    axes[0, 1].axhline(0, color="black", linestyle=":", label="Sole at free surface")
    axes[0, 1].set(ylabel="Sole depth (mm)", title="Load changes from 18.42 N down to 5 N up")
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].set(ylabel="Delivered granular vertical force (N)", xlabel="Time relative to unloading (s)")
    finish(fig, out / "holding-and-unloading.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    lateral_rows = []
    for column, load in enumerate((1, 10)):
        names = [(f"static-rft-lateral-{load}N", "RFT"), (f"static-mpm-lateral-{load}N", "MPM · 10 mm cells")]
        if load == 10:
            names.append(("static-mpm-lateral-10N-5mm", "MPM · 5 mm cells"))
        for name, label in names:
            data = trace(out, name)
            start = max(0, np.flatnonzero(data[:, 8] > 0)[0] - 1)
            mask = data[:, 0] >= data[start, 0]
            displacement = 1000 * (data[:, 6] - data[start, 6])
            axes[0, column].plot(data[mask, 0] - 1, displacement[mask], label=label)
            axes[1, column].plot(data[mask, 0] - 1, 1000 * data[mask, 4], label=label)
            lateral_rows.append(
                dict(
                    case=name,
                    title=f"{label} · {load} N sideways",
                    displacement_mm=float(displacement[-1]),
                    sole_depth_mm=float(1000 * data[-1, 4]),
                    floor_contact=read(out / name / "summary.json")["backing_floor_proximity"],
                )
            )
        axes[0, column].set(
            title=f"{load} N sideways after 1 s under vertical load", ylabel="Sideways displacement (mm)"
        )
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(118, color="black", linestyle=":", label="Rigid backing floor")
        axes[1, column].set(ylabel="Sole depth (mm)", xlabel="Time after sideways load (s)")
    finish(fig, out / "lateral-comparison.png")

    perf, previous = read(out / "performance.json"), read(out / "previous-performance.json")
    ratio = perf["control_step_seconds"] / previous["control_step_seconds"]
    collection_hours = 1500 * 24 * perf["control_step_seconds"] / 3600
    result = dict(
        newton_revision=NEWTON_REVISION,
        historical_vertical_mpm_revision="f59677bd8c5037d68d4196c323ea0b4459c18a9c",
        training_started=False,
        training_ready=False,
        lateral=lateral_rows,
        step_cost_ratio=ratio,
        collection_only_1500_iterations_hours=collection_hours,
        numerical_validation="33 focused Newton cases, 50 additional proxy regressions, 10 Isaac Lab cases",
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows_html = "".join(
        f"<tr><td>{html.escape(row['title'])}</td><td>{row['displacement_mm']:.3f}</td>"
        f"<td>{row['sole_depth_mm']:.2f}</td><td>{'Yes' if row['floor_contact'] else 'No'}</td></tr>"
        for row in lateral_rows
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Go2 — articulated RFT support and material checks</title>
<style>body{{font:17px/1.6 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;background:#f4f6f9;color:#172033}}
section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}h1,h2{{line-height:1.25}}
img,video{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
code{{overflow-wrap:anywhere}}a{{color:#0369a1}}.status{{border-left:5px solid #d97706}}</style>
<h1>Go2: articulated RFT support and MPM checks</h1><p>13 November 2026 ·
<a href="../">Geometry and previous low-speed investigation</a> · <a href="results.json">Numerical summary</a></p>
<section class="status"><h2>Support selection is implemented; material matching remains open</h2>
<p>The new projected reaction solve holds a loaded foot and the full Go2 articulation without a backing floor.
The explicit moving law remains the default. The extension selects compressive pressure and Coulomb friction
near rest, and rewinds native solver history between trials.</p>
<p><strong>No replacement policy has been trained.</strong> The current insertion-fit material still misses the
stronger lateral MPM response. Historical walking-policy videos use their original solver and checkpoints.</p></section>
<section><h2>Stationary Go2 supported only by RFT</h2>
<video controls preload="metadata" poster="static-go2-video/preview.jpg" src="static-go2-video/stance.mp4"></video>
<p>This 10-second recording uses fixed joint PD targets, with no learned controller, rigid backing floor,
or lower clipping plane. The field is an unbounded half-space whose surface is at 118 mm.
Green arrows show interval-averaged granular body forces. All limbs remain coupled; each whole spherical
foot has 128 parameter triangles and 4608 analytic integration nodes.</p>
<p>This isolates numerical support using nominal RFT strength, density 1600 kg/m³ and friction 0.4.
It is not the calibrated 118 mm-deep MPM material. Four reaction iterations, a 0.20 m/s transition and
1.25 ms rigid steps are used. Final support is about 147.34 N, equal to robot weight.</p>
<img src="articulated-stance.png" alt="Go2 stance: body height, foot speed, support and reaction residual">
<p>The 0.02 m/s transition chatters. At 0.20 m/s, four and sixteen iterations settle within 0.34 mm
in body height; halving the four-iteration timestep changes height by 0.32 mm. Foot motion reaches the
sub-micrometre-per-second range. Zero residual on a flowing patch does not certify stable dynamics.</p></section>
<section><h2>Holding and unloading</h2>
<img src="holding-and-unloading.png" alt="Loaded sphere depth and speed, plus withdrawal force">
<p>The isolated sphere has radius 22 mm, inertial mass 1 kg and independent downward load 18.42 N.
The 118 mm layer has a separate rigid backing floor. A 0.373 strength multiplier matches a previous
slow-insertion force fit. Starting both from 52.32 mm depth produces different settled depths:
106.37 mm in RFT and 79.73 mm in the fine MPM reference. RFT initialized at the settled MPM depth
stops near 82.47 mm. Stationary support alone therefore does not establish equivalent loading history.</p>
<p>In the two-axis unloading comparison the foot is first held under load for one second, then pulled
upward with 5 N. Both methods leave the bed; granular force vanishes after the whole sphere clears it.
Compressive resistance on a buried upper surface is physically different from adhesion on a withdrawing
lower surface. The flat-plate regression separately checks release, friction threshold and dissipation.</p></section>
<section><h2>Independent sideways-load comparison</h2>
<img src="lateral-comparison.png" alt="RFT and MPM displacement and sinking under 1 N and 10 N">
<table><tr><th>Method and sideways load</th><th>Displacement (mm)</th><th>Sole depth (mm)</th>
<th>Near backing floor</th></tr>{rows_html}</table>
<p>The two translational DOFs are vertical and forward; orientation is constrained. MPM is slowly inserted
to 52.32 mm and released for one second before the sideways force starts. RFT begins near the previous
settled MPM depth and is also held for one second. The coarse MPM and RFT depths and applied load
are close when pushing begins;
MPM retains its particle/material history. RFT has no compaction or ploughed-terrain history.</p>
<p><strong>The 10 N case is a failed material comparison.</strong> RFT obtains most vertical support from the
floor after sinking; that support is not credited to the granular model. Refining MPM cells from 10 to
5 mm changes displacement from 12.75 to 17.13 mm (34%) and the starting sole depth from 84.3 to 77.2 mm;
the RFT trial starts near 84.0 mm. The reference is not yet grid converged or perfectly state matched.
These short fixture comparisons do not certify whole-robot MPM convergence.</p></section>
<section><h2>Cost at 4096 environments</h2><p>The four-iteration profile measures
<strong>{perf["environment_steps_per_second"]:,.0f} environment transitions/s</strong>, versus
{previous["environment_steps_per_second"]:,.0f} for the corrected explicit profile:
{ratio:.2f} times the control-step cost. GPU memory used is about {perf["gpu_used_bytes"] / 2**30:.2f} GiB.</p>
<p>At that measured rate, 1500 PPO iterations of 24 control steps need approximately {collection_hours:.1f} hours
of simulation collection alone, before neural-network updates. MPM is used only for validation. Inner
reaction trials account for this cost; moving-force evaluation itself remains about 11–21 ms
on the common pose.</p></section>
<section><h2>Implementation and validation</h2><p>Newton revision <code>{NEWTON_REVISION}</code> contains
the reaction solve and native history rewind. The Isaac Lab adapter resets reaction warm starts only
in selected environments. Material curriculum resets now preserve configured nominal density, friction,
roughness and strength. CPU/GPU checks cover static holding, frictional slip, unloading dissipation,
empty-layer rigid equivalence, selected-world reset and CUDA graph replay.</p>
<p>Passed: 33 focused Newton cases, 50 additional proxy regressions and 10 Isaac Lab cases. The new native
clock/activation regression fails before the history fix and passes afterwards. These numerical checks
do not resolve the material mismatch above.</p><p>Next: fit or revise the lateral/normal closure using the
grid-checked MPM probes, validate the combined profile, then address collection cost before a 4096-world
training run with reset material randomization and full commands through 1 m/s. G1 remains on hold.</p>
<details><summary>Raw measurements and provenance</summary><ul>"""
    for name in CASES:
        page += (
            f'<li><a href="{name}/summary.json">{html.escape(name)} settings</a> · '
            f'<a href="{name}/trace.npz">trace</a></li>'
        )
    page += (
        '</ul><p><a href="profile.json">Experimental profile (training not cleared)</a> · '
        '<a href="performance.json">4096-world timing</a> · <a href="manifest.json">Artifact hashes</a>'
        "</p></details></section></html>"
    )
    (out / "index.html").write_text(page + "\n")
    manifest = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("logs/go2_granular/go2/low-speed"))
    parser.add_argument("--output", type=Path, default=Path("results/go2-contact-refinement/low-speed/static-contact"))
    build(parser.parse_args())
