# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish G1 force verification, calibration, retraining and held-out transfer evidence."""

import argparse
import html
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("logs/go2_granular/g1/refinement")
LABELS = {
    "control": "Original RFT model · more training",
    "matched": "Force-informed RFT model",
    "rigid": "Rigid-trained baseline",
}
GATES = {
    "grid": "Grid: 10 → 5 mm",
    "time": "Substeps: 8 → 16 at 5 mm",
    "iterations": "Stress iterations: 200 → 400",
    "domain": "Container width and length ×1.5",
    "grid_8mm": "Grid: 8 → 5 mm",
    "grid_fine": "Grid: 5 → 3.33 mm",
    "particle_sampling": "Particles per voxel: 8 → 1 (5 mm)",
    "particle_time": "Substeps: 8 → 16 with 1 particle/voxel",
    "grid_8mm_ppc1": "Grid: 8 → 5 mm with 1 particle/voxel",
    "s2_20mm": "20 mm S2 contact vs 3.33 mm reference",
    "s2_10mm": "10 mm S2 contact vs 3.33 mm reference",
    "s2_5mm": "5 mm S2 contact vs 3.33 mm reference",
}


def read(path):
    """Read completed or atomically published records."""
    return json.loads(path.read_text()) if path.exists() else {}


def loaded_plot():
    """Keep prescribed-motion reactions separate from load-controlled equilibrium."""
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for path in sorted((ROOT / "loaded").glob("*/summary.json")):
        s = read(path)
        c = s["settings"]
        if c["field"] == "mpm":
            label = f"MPM · {1000 * c['voxel_size']:g} mm · {c['mass']:g} kg · {c['load']:g} N · {c['passes']} passes"
        else:
            version = "Original" if path.parent.name in ("rft-bearing-original", "rft-bearing-zero") else "Corrected"
            label = (
                f"{version} RFT · bearing {c['bearing']:.3g} · strength {c.get('strength_multiplier', 1):.3g}"
                f" · {c['load']:g} N"
            )
        with np.load(path.parent / "trace.npz") as d:
            a = d["trace"]
            for ax, column, scale in zip(axes, (4, 2, 3), (1000, 1, 1 / c["load"]), strict=True):
                ax.plot(a[:, 0], a[:, column] * scale, label=label)
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{1000 * s['final_sole_depth_m']:.1f}</td>"
            f"<td>{s['mean_final_speed_m_s']:.4f}</td><td>{s['mean_final_total_support_N']:.1f}</td>"
            f"<td>{'Yes' if s['backing_floor_proximity'] else 'No'}</td></tr>"
        )
    for ax, label in zip(
        axes, ("Sole depth [mm]", "Vertical speed [m/s]", "Total upward force / applied load"), strict=True
    ):
        ax.set(xlabel="Time after release [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].axhline(118, color="grey", ls="--")
    axes[2].axhline(1, color="grey", ls="--")
    if rows:
        axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(ROOT / "loaded-settling.png", dpi=150)
    plt.close(fig)
    return "".join(rows)


def publish():
    """Show measured outcomes and failed gates without inferring physical validation."""
    foot = read(ROOT / "status.json")
    training = read(ROOT / "training-status.json")
    calibration = read(ROOT / "calibration-final/calibration.json")
    surface = read(ROOT / "surface-convergence.json")
    loaded_rows = loaded_plot()
    gate_rows = []
    for name, check in foot.get("checks", {}).items():
        maximum_f = max(p["force_error"] for p in check["phases"])
        maximum_t = max(p["torque_error"] for p in check["phases"])
        gate_rows.append(
            f"<tr><td>{html.escape(GATES.get(name, name))}</td><td>{maximum_f:.1%}</td>"
            f"<td>{maximum_t:.1%}</td><td>{'Pass' if check['passed'] else 'Fail'}</td></tr>"
        )
    page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1: force matching and transfer</title><style>
body{font:16px/1.5 system-ui;max-width:1250px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
section,article{background:white;padding:20px;border-radius:10px;margin:24px 0}video,img{width:100%}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
table{border-collapse:collapse}td,th{padding:8px;border:1px solid #cbd5e1}code{overflow-wrap:anywhere}
a,summary{color:#1d4ed8}.status{background:#dbeafe;padding:16px}</style>
<h1>G1: force matching and policy transfer</h1>
<p><a href="../report/">Original G1 study</a> · <a href="training.html">Live training status</a></p>
<p>Prescribed-foot tests isolate force response, loaded-foot tests examine support and settling,
and matched training continuations examine learned robustness. MPM supplies no training transitions.
Numerical self-convergence, force agreement and policy transfer are separate acceptance questions.</p>"""
    work = read(ROOT / "bearing-work.json")
    if work:
        before, after = work["before"][0], work["after"][0]
        page += (
            "<section><h2>Bearing-model correction</h2>"
            "<p>The experimental near-rest bearing term acted on withdrawing surface patches. "
            "A closed 40 mm penetration cycle on the exact G1 foot returned more work than supplied: "
            f"net input <strong>{before['net_input_work_J']:.3f} J</strong>. "
            "The corrected closure releases bearing pressure on outward-moving patches; "
            f"the same cycle now dissipates <strong>{after['net_input_work_J']:.3f} J</strong>. "
            "The Newton regression passes on CPU and CUDA. This correction concerns the added bearing extension.</p>"
            "<p>Original Go2/G1 checkpoints used the earlier extension. Their recordings remain historical results. "
            "This issue could permit learned exploitation, but does not by itself establish the cause of every fall. "
            "The revised policy uses Newton 277c302; the additional-training control retains 32f455b.</p></section>"
        )
    page += """
<section><h2>Foot-level numerical verification</h2>
<p>The exact G1 collision mesh is approximately 203 × 65 × 19 mm. Both solvers receive the same ankle
poses and velocities. Forces and torques include every solver substep. A 118 mm bed is prepared for
0.5 s before insertion, hold, drag, pitch, roll and withdrawal. The container has frictionless sidewalls.
Only particle reactions contribute to the measured foot wrench.</p>"""
    page += f'<p class="status">{html.escape(foot.get("stage", "Preparing measurements"))}</p>'
    page += """<p>Each loaded phase must change by less than 10% in both mean force and mean torque.
Normalization floors of 20 N and 1 Nm give allowances of 2 N and 0.1 Nm near zero.
An isolated-foot convergence check does not establish whole-robot coupling convergence.</p>
<table><tr><th>Refinement</th><th>Largest force change</th><th>Largest torque change</th><th>Gate</th></tr>"""
    page += "".join(gate_rows) + '</table><img src="foot-convergence.png" alt="G1 foot force and torque refinement">'
    if surface:
        first = surface["results"][0]
        page += (
            "<p>The original RFT surface integration differs from the finest mesh by "
            f"{first['force_relative_error']:.1%} "
        )
        page += f"in force and {first['torque_relative_error']:.1%} in torque on four saved walking snapshots.</p>"
    page += "</section><section><h2>Support under an applied load</h2>"
    page += """<p>A vertical prismatic joint fixes lateral translation and foot orientation. Inertial mass and downward
load are specified independently. The bed is prepared with the foot fixed, then the foot is released.
Total support is inferred from vertical momentum balance and includes backing-floor contact if reached.
Final speed and support average the last 0.5 s.</p>
<p>A prescribed zero-velocity hold and a freely loaded equilibrium impose different boundary conditions.
The unloaded hold alone cannot determine the admissible static support under body weight.</p>
<table><tr><th>Fixture</th><th>Final sole depth [mm]</th><th>Final mean speed [m/s]</th>
<th>Final mean support [N]</th><th>Backing-floor proximity</th></tr>"""
    page += loaded_rows + '</table><img src="loaded-settling.png" alt="Loaded foot depth, speed and support"></section>'
    page += "<section><h2>RFT force approximation</h2>"
    if calibration:
        c = calibration["selected"]
        outcome = "passed" if calibration["holdout_accepted"] else "failed"
        page += f"<p><strong>Held-out force matching: {outcome}.</strong> "
        page += "The candidate remains an approximation; fitting a subset does not establish material equivalence.</p>"
        page += """<p>Moving insertion and drag measurements determine traction response. A settled loaded-foot
measurement constrains near-rest support when available. Unloaded holds, rotations and withdrawal remain
held out. The local model has no evolving surface or plastic loading history.</p>"""
        page += (
            f"<p>Candidate: internal friction {c['internal_friction']:g}, effective interface/internal ratio "
            f"{c['roughness']:g}, strength multiplier {c['strength_multiplier']:.3f}, bearing {c['bearing']:.3f}, "
            f"velocity regularization {c['regularization']:g} m/s.</p>"
        )
        fit_path = "calibration-final"
        loaded = c.get("loaded_result", {})
        if loaded:
            page += (
                f"<p>Calibration load: {loaded['settings']['load']:.2f} N. Corrected RFT settles to "
                f"{1000 * loaded['final_sole_depth_m']:.1f} mm versus "
                f"{1000 * calibration['loaded_measurement']['final_sole_depth_m']:.1f} mm in fine MPM. "
                "The fitted interface ratio is an effective RFT parameter; MPM retains ratio 1.0.</p>"
            )
        page += f'<img src="{fit_path}/force-fit.png" alt="Original and fitted RFT versus MPM wrenches">'
        phase_rows = []
        for row in calibration["phases"]:
            phase_rows.append(
                f"<tr><td>{row['phase'].replace('_', ' ')}</td><td>{row['split']}</td>"
                f"<td>{row['measured_wrench'][2]:.1f}</td><td>{row['original_wrench'][2]:.1f}</td>"
                f"<td>{row['fitted_wrench'][2]:.1f}</td><td>{row['force_error']:.1%}</td></tr>"
            )
        page += (
            """<details><summary>Force measurements by phase</summary><table><tr><th>Phase</th><th>Split</th>
<th>MPM Fz [N]</th><th>Original RFT Fz [N]</th><th>Candidate RFT Fz [N]</th>
<th>Mean force-vector error</th></tr>"""
            + "".join(phase_rows)
            + "</table></details>"
        )
    else:
        page += "<p>Calibration awaits the loaded-foot reference.</p>"
    page += "</section><section><h2>Training at full material difficulty</h2>"
    page += """<p>The original-model control and force-informed variant start from the same original RFT
checkpoint, with reset optimizer state, seed 42 and 4,096 environments. Each continuation receives
1,000 additional PPO iterations with full material distributions from its first episode.
Forward commands remain 0.1–1.0 m/s; there is no command curriculum.
The force-informed model and wider randomization form an exploratory combined change.</p>"""
    train_rows = []
    for name, record in training.get("runs", {}).items():
        train_rows.append(
            f"<tr><td>{html.escape(LABELS.get(name, name))}</td>"
            f"<td>{record.get('completed_iterations', 0):,} / {record['iterations']:,}</td>"
            f"<td>{html.escape(record['stage'])}</td></tr>"
        )
    page += (
        "<table><tr><th>Policy</th><th>Additional iterations</th><th>Status</th></tr>"
        + "".join(train_rows)
        + "</table>"
    )
    if (ROOT / "retraining.png").exists():
        page += '<img src="retraining.png" alt="Matched full-range learning curves">'
    profile = read(ROOT / "rft-profile.json")
    if profile:
        page += (
            "<details><summary>Force-informed material settings</summary><pre>"
            + html.escape(json.dumps(profile, indent=2))
            + "</pre></details>"
        )
    page += "</section>"
    evaluations = read(ROOT / "evaluation-status.json")
    if evaluations:
        page += "<section><h2>Held-out policy measurements</h2><table><tr><th>Policy</th><th>Trial</th>"
        page += "<th>Planar error [m/s]</th><th>Body failure threshold</th></tr>"
        for row in evaluations.get("cases", []):
            s = row["summary"]
            first = s["first_failure_seconds"]
            outcome = f"{first:.2f} s" if first is not None else f"None in {s['simulated_seconds']:g} s"
            page += (
                f"<tr><td>{html.escape(LABELS.get(row['policy'], row['policy']))}</td>"
                f"<td>{html.escape(row['label'])}</td><td>{s['mean_velocity_error']:.3f}</td><td>{outcome}</td></tr>"
            )
        page += (
            "</table><p>Metrics include acceleration and time after failure. "
            "Rollouts continue without resets.</p></section>"
        )
        videos = [row for row in evaluations.get("cases", []) if row.get("video")]
        if videos:
            page += "<section><h2>Final policy recordings</h2>"
            for policy, field in dict.fromkeys((row["policy"], row["field"]) for row in videos):
                title = (
                    LABELS.get(policy, policy)
                    + " on "
                    + {"rft": "RFT sand", "mpm": "MPM sand", "rigid": "a rigid floor"}[field]
                )
                page += f'<article><h3>{html.escape(title)}</h3><div class="pair">'
                for row in (r for r in videos if r["policy"] == policy and r["field"] == field):
                    s = row["summary"]
                    name = row["name"]
                    speed = s["commands"][0][0]
                    page += (
                        f"<div><h4>Command: {speed:.2f} m/s forward</h4>"
                        f'<video controls preload="metadata" src="evaluations/{name}/rollout.mp4" '
                        f'poster="evaluations/{name}/preview.jpg"></video>'
                        f"<p>{s['simulated_seconds']:g} s; mean planar error "
                        f"{s['mean_velocity_error']:.3f} m/s.</p></div>"
                    )
                page += "</div></article>"
            page += "</section>"
    else:
        page += "<section><h2>Held-out inference</h2><p>Native and MPM comparisons follow training.</p></section>"
    page += (
        '<script>setInterval(()=>{if(![...document.querySelectorAll("video")]'
        ".some(v=>!v.paused&&!v.ended))location.reload()},30000)</script>"
    )
    temporary = ROOT / "analysis.tmp"
    temporary.write_text(page)
    temporary.replace(ROOT / "analysis.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        publish()
        if not args.watch:
            break
        time.sleep(30)
