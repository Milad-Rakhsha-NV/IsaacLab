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
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path("logs/go2_granular/g1/refinement")
LABELS = {
    "control": "Original RFT model · more training",
    "matched": "Revised RFT-trained policy",
    "rigid": "Rigid-trained policy",
    "initial": "Original RFT-trained checkpoint",
}
COLORS = {
    ("rigid", "rigid"): "#64748b",
    ("matched", "rft"): "#15803d",
    ("matched", "mpm"): "#2563eb",
    ("rigid", "mpm"): "#d97706",
    ("control", "mpm"): "#9333ea",
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
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def geometry_plot():
    """Show the collider dimensions and the ankle origin used for measured moments."""
    output = ROOT / "probe-geometry.png"
    if output.exists():
        return
    geometry = read(ROOT / "geometry.json")
    if not geometry:
        return
    foot = next(r for r in geometry["surfaces"] if r["body_label"].endswith("/left_ankle_roll_link"))
    vertices = np.array(foot["vertices"]) * 1000
    triangles = vertices[np.array(foot["indices"]).reshape(-1, 3)]
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(triangles, facecolor="#93c5fd", edgecolor="#1d4ed8", alpha=0.35))
    ax.scatter(0, 0, 0, color="#dc2626", s=35)
    ax.text(-65, -45, 40, "O: ankle origin\nMoment reference", fontsize=9)
    for direction, label, color in (
        ((1, 0, 0), "x: forward", "#2563eb"),
        ((0, 1, 0), "y: left", "#15803d"),
        ((0, 0, 1), "z: up", "#d97706"),
    ):
        vector = np.array(direction) * 45
        ax.quiver(0, 0, 0, *vector, color=color)
        ax.text(*vector, label, color=color, fontsize=8)
    ax.set(
        xlim=(-80, 160),
        ylim=(-60, 70),
        zlim=(-50, 60),
        xlabel="Body x [mm]",
        ylabel="Body y [mm]",
        zlabel="Body z [mm]",
    )
    ax.set_box_aspect((240, 130, 110))
    ax.view_init(elev=25, azim=-60)
    fig.suptitle("Exact G1 foot collider: 203 × 65 × 19 mm\nAuthored offset from the ankle is retained")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def loaded_plot():
    """Keep prescribed-motion reactions separate from load-controlled equilibrium."""
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    paths = sorted((ROOT / "loaded").glob("*/summary.json"))
    candidate = ROOT / "calibration-final/bearing-0.45/summary.json"
    if candidate.exists():
        paths.append(candidate)
    plotted = {
        "mpm-mass16-5mm-pass1-alpha1": "MPM · 5 mm grid",
        "mpm-mass16-3p33mm-pass1-alpha1": "MPM · 3.33 mm grid",
        "rft-bearing-original": "Original RFT before correction",
        "rft-bearing-release-fixed": "Corrected RFT before calibration",
        "bearing-0.45": "Corrected and calibrated RFT",
    }
    for path in paths:
        s = read(path)
        if not s:
            continue
        c = s["settings"]
        if c["field"] == "mpm":
            label = (
                f"MPM · {1000 * c['voxel_size']:g} mm · {c['mass']:g} kg · {c['load']:g} N"
                f" · {c['passes']} passes · {c['substeps']} substeps"
            )
        else:
            version = "Original" if path.parent.name in ("rft-bearing-original", "rft-bearing-zero") else "Corrected"
            label = (
                f"{version} RFT · bearing {c['bearing']:.3g} · strength {c.get('strength_multiplier', 1):.3g}"
                f" · {c['load']:g} N · {c['substeps']} substeps · {c['seconds']:g} s"
            )
        if path.parent.name in plotted:
            with np.load(path.parent / "trace.npz") as d:
                a = d["trace"]
                for ax, column, scale in zip(axes, (4, 2, 3), (1000, 1, 1 / c["load"]), strict=True):
                    ax.plot(a[:, 0], a[:, column] * scale, label=plotted[path.parent.name])
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
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "loaded-settling.png", dpi=150)
    plt.close(fig)
    return "".join(rows)


def long_hold_plot():
    """Expose accumulated near-rest drift that a short depth comparison can miss."""
    paths = sorted((ROOT / "loaded").glob("rft-calibrated-long-*/summary.json"))
    if not paths:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows, timesteps, slopes, records = [], [], [], []
    mpm = ROOT / "loaded/mpm-10mm-long10s/summary.json"
    if mpm.exists():
        paths.append(mpm)
    for path in paths:
        s = read(path)
        if not s:
            continue
        c = s["settings"]
        with np.load(path.parent / "trace.npz") as data:
            trace = data["trace"]
        early = (trace[:, 0] >= 1.5) & (trace[:, 0] <= 2)
        slope = 1000 * float(np.polyfit(trace[early, 0], trace[early, 4], 1)[0])
        at_two = 1000 * float(np.interp(2, trace[:, 0], trace[:, 4]))
        dt = 5 / c["substeps"]
        label = f"RFT · {c['substeps']} substeps" if c["field"] == "rft" else "MPM · 10 mm grid"
        axes[0].plot(trace[:, 0], trace[:, 4] * 1000, label=label)
        if c["field"] == "rft":
            timesteps.append(dt)
            slopes.append(slope)
        records.append(
            dict(
                fixture=path.parent.name,
                field=c["field"],
                timestep_ms=dt,
                depth_at_2s_mm=at_two,
                depth_at_10s_mm=1000 * s["final_sole_depth_m"],
                drift_2_to_10s_mm=1000 * s["final_sole_depth_m"] - at_two,
                early_drift_mm_s=slope,
                backing_floor_proximity=s["backing_floor_proximity"],
            )
        )
        rows.append(
            f"<tr><td>{label}</td><td>{dt:g}</td><td>{at_two:.1f}</td>"
            f"<td>{1000 * s['final_sole_depth_m']:.1f}</td><td>{slope:.3f}</td>"
            f"<td>{'Yes' if s['backing_floor_proximity'] else 'No'}</td></tr>"
        )
    if timesteps:
        order = np.argsort(timesteps)
        dt, drift = np.array(timesteps)[order], np.array(slopes)[order]
        axes[1].loglog(dt, drift, "o-", label="Measured RFT drift")
        axes[1].loglog(dt, drift[0] * dt / dt[0], "--", label="Proportional to timestep")
        axes[1].legend(fontsize=8)
    axes[0].axvline(2, color="grey", ls=":")
    axes[0].axhline(118, color="grey", ls="--", label="Backing floor")
    axes[0].set(xlabel="Time after release [s]", ylabel="Sole depth [mm]")
    axes[1].set(xlabel="Rigid/RFT timestep [ms]", ylabel="Depth drift at 1.5–2 s [mm/s]")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "static-drift.png", dpi=150)
    plt.close(fig)
    analysis = dict(
        results=records,
        maximum_allowed_drift_mm=2.0,
        rft_static_accepted=all(
            abs(r["drift_2_to_10s_mm"]) <= 2 and not r["backing_floor_proximity"]
            for r in records
            if r["field"] == "rft"
        ),
        interpretation="Prolonged hold: at most 2 mm depth change from 2 to 10 s and no backing-floor proximity.",
    )
    (ROOT / "static-hold-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return "".join(rows)


def standing_support():
    """Separate recorded granular support from geometric proximity to the substrate."""
    geometry = read(ROOT / "geometry.json")
    if not geometry:
        return {}
    vertices = np.array(
        next(s for s in geometry["surfaces"] if s["body_label"].endswith("/left_ankle_roll_link"))["vertices"]
    )
    rows = []
    for name, start, end in (("control_stand", 1.5, 5), ("matched_stand", 1.5, 5), ("matched_stand", 15, 20)):
        path = ROOT / "evaluations" / name
        summary = read(path / "summary.json")
        if not summary:
            continue
        feet = [
            i
            for i, label in enumerate(summary["body_labels"])
            if label.endswith(("/left_ankle_roll_link", "/right_ankle_roll_link"))
        ]
        with np.load(path / "rollout.npz") as data:
            t = data["state"][:, 0, 0]
            mask = (t >= start) & (t <= end)
            pose = data["body_pose"][mask][:, feet]
            xyz, scalar = pose[:, :, None, 3:6], pose[:, :, None, 6:7]
            rotated = vertices[None, None] + 2 * np.cross(xyz, np.cross(xyz, vertices) + scalar * vertices)
            sole = (pose[:, :, None, :3] + rotated)[:, :, :, 2].min(-1)
            force = data["body_wrench"][mask, :, 2].sum(1) / len(summary["commands"])
        rows.append(
            dict(
                case=name,
                window_seconds=[start, end],
                granular_upward_force_N=float(force.mean()),
                weight_N=summary["robot_weight_N"],
                granular_force_over_weight=float(force.mean() / summary["robot_weight_N"]),
                sole_within_1mm_of_floor_fraction=float((sole <= 0.001).mean()),
                minimum_sole_z_m=float(sole.min()),
            )
        )
    record = dict(
        results=rows,
        interpretation=(
            "Granular force is recorded as an interval average. Geometric sole proximity is not a contact-force "
            "measurement. Other support is inferred from steady momentum balance, not directly measured."
        ),
    )
    (ROOT / "diagnostics").mkdir(exist_ok=True)
    (ROOT / "diagnostics/standing-support.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def support_section(support):
    """Explain why low standing drift alone does not validate granular support."""
    for row in support.get("results", []):
        if row["case"] == "matched_stand" and row["window_seconds"][0] == 15:
            return (
                "<section><h2>Backing-floor support in the standing trial</h2>"
                "<p>During seconds 15–20, the revised policy's mean upward granular force is "
                f"{row['granular_upward_force_N']:.1f} N against {row['weight_N']:.1f} N weight "
                f"({row['granular_force_over_weight']:.1%}). The exact transformed foot mesh is within "
                "1 mm of the backing floor throughout this window.</p>"
                "<p>Most remaining support is inferred to come from the substrate under steady momentum balance. "
                "Rigid-contact force was not directly recorded in this trial. Low standing drift therefore "
                "does not validate granular sticking.</p></section>"
            )
    return ""


def policy_plots(cases):
    """Compare measured speed, clearance and tilt at each fixed command."""
    for speed in (0.5, 1.0):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        count = 0
        for row in cases:
            s = row["summary"]
            if s["commands"][0][0] != speed:
                continue
            if not row.get("video") and row["name"] != f"control_mpm_nominal_{speed:g}":
                continue
            with np.load(ROOT / "evaluations" / row["name"] / "rollout.npz") as d:
                state, metrics = d["state"][:, 0], d["metrics"][:, 0]
                key = row["policy"], row["field"]
                label = LABELS[row["policy"]] + " / " + row["field"].upper()
                for ax, values in zip(axes, (state[:, 4], state[:, 7], metrics[:, 3]), strict=True):
                    ax.plot(state[:, 0], values, label=label, color=COLORS[key])
                failure = s["first_failure_seconds"]
                if failure is not None:
                    axes[1].axvline(failure, color=COLORS[key], ls=":", alpha=0.5)
                count += 1
        axes[0].axhline(speed, color="black", ls="--", label="Forward command")
        axes[1].axhline(0.4, color="black", ls="--", label="Low-clearance threshold")
        for ax, label in zip(axes, ("Forward body speed [m/s]", "Base clearance [m]", "Base tilt [rad]"), strict=True):
            ax.set(xlabel="Time [s]", ylabel=label)
            ax.grid(alpha=0.2)
        if count:
            axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(ROOT / f"comparison-{speed:g}.png", dpi=150)
        plt.close(fig)


def path_plot(cases):
    """Expose heading drift and finite-bed coverage separately from body-frame tracking."""
    selected = [row for row in cases if row.get("video")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, speed in zip(axes, (0.5, 1.0), strict=True):
        beds = set()
        for row in selected:
            s = row["summary"]
            if s["commands"][0][0] != speed:
                continue
            with np.load(ROOT / "evaluations" / row["name"] / "rollout.npz") as data:
                state = data["state"][:, 0]
            ax.plot(
                state[:, 1],
                state[:, 2],
                label=LABELS[row["policy"]] + " / " + row["field"].upper(),
                color=COLORS[(row["policy"], row["field"])],
            )
            dimensions = s["bed_length"], s["bed_width"]
            if row["field"] == "mpm" and dimensions not in beds:
                ax.add_patch(
                    Rectangle(
                        (-0.7, -s["bed_width"] / 2),
                        s["bed_length"],
                        s["bed_width"],
                        fill=False,
                        edgecolor="black",
                        linestyle="--",
                        label=f"MPM bed: {dimensions[0]:g} × {dimensions[1]:g} m",
                    )
                )
                beds.add(dimensions)
        ax.set(title=f"Forward command {speed:.2f} m/s", xlabel="World x [m]", ylabel="World y [m]")
        ax.axis("equal")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "world-paths.png", dpi=150)
    plt.close(fig)


def material_plot(cases):
    """Show short-window outcomes across materials without treating censoring as a fall."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    materials = ("nominal", "shallow_loose", "dense")
    for column, speed in enumerate((0.5, 1.0)):
        for i, policy in enumerate(("rigid", "control", "matched")):
            bank = {r["name"]: r["summary"] for r in cases if r["field"] == "mpm" and r["policy"] == policy}
            x = np.arange(3) + (i - 1) * 0.24
            times, errors = [], []
            for material in materials:
                row = bank.get(f"{policy}_mpm_{material}_{speed:g}")
                if row:
                    first = row["first_failure_seconds"]
                    times.append(first if first is not None else row["simulated_seconds"])
                    error = row.get("mean_pre_failure_velocity_error")
                    errors.append(row["mean_velocity_error"] if error is None else error)
                else:
                    times.append(np.nan)
                    errors.append(np.nan)
            axes[0, column].bar(x, times, width=0.23, label=LABELS[policy])
            axes[1, column].bar(x, errors, width=0.23, label=LABELS[policy])
        axes[0, column].set(title=f"Forward command {speed:.2f} m/s", ylabel="Failure time, capped at 3 s")
        axes[1, column].set(ylabel="Pre-failure planar tracking error [m/s]")
        for ax in axes[:, column]:
            ax.set_xticks(np.arange(3), ("Nominal", "Shallow / loose", "Dense"))
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "material-transfer.png", dpi=150)
    plt.close(fig)


def reference_plot(cases):
    """Compare whole-robot trajectories while keeping the refined bed dimensions fixed."""
    selected = [r for r in cases if "_reference_" in r["name"]]
    if not selected:
        return ""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    rows = []
    for row in selected:
        s = row["summary"]
        label = f"{1000 * s['voxel_size']:g} mm grid"
        with np.load(ROOT / "evaluations" / row["name"] / "rollout.npz") as d:
            state, metrics = d["state"][:, 0], d["metrics"][:, 0]
            for ax, values in zip(axes, (state[:, 4], state[:, 7], metrics[:, 3]), strict=True):
                ax.plot(state[:, 0], values, label=label)
        outcome = s["first_failure_seconds"]
        failure = "None" if outcome is None else f"{outcome:.2f} s"
        rows.append(
            f"<tr><td>{label}</td><td>{s['simulated_seconds']:.2f}</td>"
            f"<td>{failure}</td><td>{s['mean_velocity_error']:.3f}</td>"
            f"<td>{s['first_bed_exit_seconds']}</td></tr>"
        )
    axes[0].axhline(0.5, color="black", ls="--")
    axes[1].axhline(0.4, color="black", ls="--")
    for ax, label in zip(axes, ("Forward body speed [m/s]", "Base clearance [m]", "Base tilt [rad]"), strict=True):
        ax.set(xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "whole-robot-refinement.png", dpi=150)
    plt.close(fig)
    return "".join(rows)


def foot_domain_coverage(cases):
    """Check actual foot vertices against each recorded MPM bed at control frames."""
    surfaces = read(ROOT / "foot-quadrature-convergence.json").get("foot_surfaces", [])
    records = {}
    for row in cases:
        if row["field"] != "mpm" or not row.get("video") or not surfaces:
            continue
        summary, feet = row["summary"], []
        with np.load(ROOT / "evaluations" / row["name"] / "rollout.npz") as data:
            for surface in surfaces:
                body = summary["body_labels"].index(surface["body"])
                pose = data["body_pose"][:, body]
                vertices = np.array(surface["vertices"])[None]
                xyz, scalar = pose[:, None, 3:6], pose[:, None, 6:7]
                points = pose[:, None, :3] + vertices + 2 * np.cross(xyz, np.cross(xyz, vertices) + scalar * vertices)
                margin = np.minimum.reduce(
                    (
                        points[:, :, 0] + 0.7,
                        summary["bed_length"] - 0.7 - points[:, :, 0],
                        summary["bed_width"] / 2 - abs(points[:, :, 1]),
                    )
                )
                outside = np.flatnonzero((margin < 0).any(axis=1))
                submerged = np.flatnonzero(((margin < 0) & (points[:, :, 2] <= 0)).any(axis=1))
                feet.append(
                    dict(
                        body=surface["body"],
                        minimum_planar_margin_m=float(margin.min()),
                        first_vertex_exit_seconds=None if not len(outside) else float(data["state"][outside[0], 0, 0]),
                        first_submerged_vertex_exit_seconds=None
                        if not len(submerged)
                        else float(data["state"][submerged[0], 0, 0]),
                    )
                )
        records[row["name"]] = feet
    (ROOT / "foot-domain-coverage.json").write_text(
        json.dumps(
            dict(
                sampling=("Exact foot collider vertices at recorded 20 ms control frames; not a substep certificate."),
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )
    return records


def render_policy_videos(cases, coverage):
    """Keep terrain roles, commands and pending recordings clear in a fixed order."""
    roles = (
        (
            "rigid",
            "rigid",
            "Rigid-trained policy on a rigid floor",
            "The accepted rigid-ground checkpoint is evaluated on its training surface.",
        ),
        (
            "matched",
            "rft",
            "Revised RFT-trained policy on RFT sand",
            "The final revised checkpoint is evaluated at fixed nominal properties in its RFT training simulator.",
        ),
        (
            "matched",
            "mpm",
            "Revised RFT-trained policy on MPM sand",
            "The exact same revised weights are tested in a particle bed after training. "
            "MPM provides no training data.",
        ),
        (
            "rigid",
            "mpm",
            "Rigid-trained policy on MPM sand",
            "The same rigid checkpoint as the rigid-floor clip is tested in the identical MPM bed and command.",
        ),
    )
    videos = {(r["policy"], r["field"], r["summary"]["commands"][0][0]): r for r in cases if r.get("video")}
    page = (
        '<section id="videos"><h2>Final policy recordings</h2>'
        "<p>Green arrows show granular body forces at 1 m per 200 N. RFT arrows are control-interval averages; "
        "MPM arrows show the last coupling feedback. Recording durations are stated below each clip.</p>"
    )
    for policy, field, title, description in roles:
        page += f'<article><h3>{html.escape(title)}</h3><p>{description}</p><div class="pair">'
        for speed in (0.5, 1.0):
            page += f"<div><h4>Command: {speed:.2f} m/s forward</h4>"
            row = videos.get((policy, field, speed))
            if row is None:
                page += "<p>Recording pending.</p></div>"
                continue
            s, name = row["summary"], row["name"]
            if field == "mpm" and s["simulated_seconds"] < 9.9:
                page += (
                    "<p><strong>Short diagnostic preview; full 10-second replacement is being generated.</strong></p>"
                )
            failure = s["first_failure_seconds"]
            outcome = "No body threshold reached" if failure is None else f"First body threshold: {failure:.2f} s"
            page += (
                f'<video controls preload="metadata" src="evaluations/{name}/rollout.mp4" '
                f'poster="evaluations/{name}/preview.jpg"></video>'
                f"<p>{s['simulated_seconds']:g} s; mean planar error {s['mean_velocity_error']:.3f} m/s. "
                f"{outcome}. Mean yaw-rate error {s['mean_yaw_error']:.3f} rad/s.</p>"
            )
            if field == "mpm":
                page += (
                    f"<p>Bed: {s['bed_length']:g} × {s['bed_width']:g} m; "
                    "118 mm depth. The recording continues after a fall without resetting the robot.</p>"
                )
                if name in coverage:
                    margin = min(foot["minimum_planar_margin_m"] for foot in coverage[name])
                    page += f"<p>Minimum recorded foot-vertex margin to a bed edge: {1000 * margin:.0f} mm.</p>"
            page += "</div>"
        page += "</div></article>"
    return page + "</section>"


def foot_quadrature_section():
    """Separate sole integration resolution from the closed foot collider."""
    audit = read(ROOT / "foot-quadrature-convergence.json")
    page = """<section id="foot-resolution"><h2>Foot surface integration</h2>
<p>Counts below refer to one foot. The current training mesh has <strong>2 triangles on the sole</strong>
and 12 over the complete foot collision box. Torso and leg triangles do not measure sole resolution.
The flat box geometry is exact, but force integration can still need refinement when rotation changes
depth, velocity direction or activation across a face. RFT integrates each triangle at three quadrature
points after clipping it against the material layer.</p>"""
    if not audit:
        return page + "<p>The foot-only convergence audit is running.</p></section>"
    foot_mesh_plot(audit)
    page += (
        f"<p>The audit prescribes {audit['motion_samples']} recorded poses with exact COM twists from the "
        "final policy at both commands, swept over 41 material/height combinations. Each foot is evaluated "
        "independently; a parity check reproduces the original full-model foot wrenches. "
        "Errors below are pooled L2 differences against a 10 mm maximum-edge reference. "
        "Sole-only reactions exclude side and upper faces; whole-foot reactions include them. "
        "Moments are about the actual foot body COM. This checks numerical quadrature, not material validity.</p>"
        "<table><tr><th>Maximum edge</th><th>Sole triangles per foot</th><th>All triangles per foot</th>"
        "<th>Largest sole force / torque error</th><th>Largest whole-foot force / torque error</th></tr>"
    )
    for row in audit["results"]:
        edge = row["max_edge_length"]
        label = "Current (no subdivision)" if edge is None else f"{1000 * edge:g} mm"
        sole_f = max(f["sole_only"]["force_relative_error"] for f in row["feet"])
        sole_t = max(f["sole_only"]["torque_relative_error"] for f in row["feet"])
        whole_f = max(f["whole_foot"]["force_relative_error"] for f in row["feet"])
        whole_t = max(f["whole_foot"]["torque_relative_error"] for f in row["feet"])
        page += (
            f"<tr><td>{label}</td><td>{row['sole_triangles_per_foot'][0]}</td>"
            f"<td>{row['triangles_per_foot']}</td><td>{sole_f:.1%} / {sole_t:.1%}</td>"
            f"<td>{whole_f:.1%} / {whole_t:.1%}</td></tr>"
        )
    return page + (
        '</table><img src="foot-quadrature-mesh.png" alt="Sole triangulations with 2, 128 and 512 triangles per foot">'
        "<p>The 40 mm edge mesh reduces the largest pooled sole-torque difference from 18.0% to 0.6%. "
        "It is a practical refinement candidate for a future training configuration. "
        "It does not resolve the separate static-coupling failure.</p>"
        "<p>Each error column reports the larger of the left and right foot errors. "
        "The raw audit also includes loaded-sample 95th percentiles and maximum errors with explicit "
        "20 N / 1 Nm denominator floors. Existing checkpoints retain the current coarse mesh; "
        "refinement results are diagnostics and do not silently change the trained simulator.</p></section>"
    )


def foot_mesh_plot(audit):
    """Draw the actual bottom-face subdivision, with one-foot counts."""
    output = ROOT / "foot-quadrature-mesh.png"
    if output.exists():
        return
    geometry = read(ROOT / "geometry.json")
    foot = next(r for r in geometry["surfaces"] if r["body_label"].endswith("/left_ankle_roll_link"))
    vertices = np.array(foot["vertices"])
    triangles = vertices[np.array(foot["indices"]).reshape(-1, 3)]
    bottom = triangles[np.all(np.isclose(triangles[:, :, 2], vertices[:, 2].min()), axis=1)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    for ax, row in zip(axes, audit["results"][:3], strict=True):
        pending, prepared = list(bottom), []
        edge = row["max_edge_length"]
        while pending:
            triangle = pending.pop()
            if edge is not None and np.max(np.linalg.norm(triangle - np.roll(triangle, 1, axis=0), axis=1)) > edge:
                a, b, c = triangle
                ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
                pending.extend(([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
            else:
                prepared.append(triangle)
        assert len(prepared) == row["sole_triangles_per_foot"][0]
        ax.add_collection(
            PolyCollection(np.array(prepared)[:, :, :2] * 1000, facecolor="#dbeafe", edgecolor="#1d4ed8", linewidth=0.5)
        )
        label = "Current" if edge is None else f"Maximum edge {1000 * edge:g} mm"
        ax.set(title=f"{label}: {len(prepared)} sole triangles on one foot", xlabel="Body x [mm]", ylabel="Body y [mm]")
        ax.autoscale_view()
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def go2_resolution_section():
    """Keep the small curved Go2 contact shape separate from the G1 sole study."""
    audit = read(ROOT / "go2-foot-resolution.json")
    if audit.get("reference_segments") != 384:
        return ""
    page = """<section id="go2-foot-resolution"><h2>Go2: curved foot resolution</h2>
<p>The Go2 collider is a sphere of radius 22 mm. Its current UV tessellation has edges up to 15.6 mm.
Curved feet require accurate surface normals as well as enough integration points. Subdividing existing
flat facets retains their coarse curvature. Regenerating the sphere, or using a uniform icosphere mesh,
improves both its shape and force integration. These RFT surfaces can be refined independently of
the rigid/MPM collision geometry and robot inertial properties.</p>
<p>This preliminary test prescribes 150 centered-sphere states at four friction/interface combinations:
1–44 mm penetration, insertion, translation, withdrawal and 8 rad/s rotation at three orientations.
The corrected force law is used, with moments about the sphere center. The numerical reference has
384 UV segments (about 0.51 mm edges).
It is a surface self-convergence study, rather than robot-policy or physical-material validation.</p>
<img src="go2-foot-meshes.png" alt="Current and uniformly refined lower surfaces of a Go2 spherical foot">
<table><tr><th>Mesh</th><th>Triangles per foot</th><th>Maximum edge [mm]</th>
<th>Pooled force / torque difference</th><th>95th-percentile force / torque difference</th></tr>"""
    for row in audit["results"]:
        if (row["kind"], row["resolution"]) not in (
            ("uv", 12),
            ("uv", 64),
            ("icosphere", 3),
            ("icosphere", 4),
            ("icosphere", 5),
            ("uv", 384),
        ):
            continue
        page += (
            f"<tr><td>{row['label']}</td><td>{row['integrated_triangles_per_foot']:,}</td>"
            f"<td>{1000 * row['maximum_edge_m']:.2f}</td>"
            f"<td>{row['force_relative_error']:.1%} / {row['torque_relative_error']:.1%}</td>"
            f"<td>{row['force_error_p95']:.1%} / {row['torque_error_p95']:.1%}</td></tr>"
        )
    return (
        page
        + """</table>
<p>Percentiles normalize by reference magnitude with 1 N / 0.01 Nm floors. Pooled errors weight larger
wrenches more strongly; difficult deeply immersed spinning cases retain larger errors. Per-depth
results and raw wrenches are saved separately. Solver counts include negligible pole faces on UV meshes.
The diagrams classify lower-half faces by their centroids at the nominal orientation; the submerged
leading patch changes during motion.</p>
<img src="go2-foot-convergence.png" alt="Go2 force and torque convergence for UV and uniform sphere meshes">
<p>A 1,280-triangle uniform sphere is an economical candidate, with 5,120 triangles as a stricter comparison.
Actual recorded foot motion, per-depth errors and throughput at 4,096 environments must determine the
training resolution. Refinement should target foot colliders and be cached at model construction.
Existing task defaults and checkpoints still use the coarse surfaces. New policies require retraining
after the resolution is selected. The separate near-rest support failure remains unresolved.</p></section>"""
    )


def publish():
    """Show measured outcomes and failed gates without inferring physical validation."""
    foot = read(ROOT / "status.json")
    training = read(ROOT / "training-status.json")
    calibration = read(ROOT / "calibration-final/calibration.json")
    geometry_plot()
    loaded_rows = loaded_plot()
    drift_rows = long_hold_plot()
    support = standing_support()
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
<p><a href="../baseline-report/">Original G1 study</a> · <a href="training.html">Training status</a></p>
<p><a href="#videos">Videos</a> · <a href="#comparison-050">0.50 m/s comparison</a> ·
<a href="#comparison-100">1.00 m/s comparison</a> · <a href="#training">Learning curves</a> ·
<a href="#static">Static-support failure</a></p>
<p>Prescribed-foot tests isolate force response, loaded-foot tests examine support and settling,
and matched training continuations examine learned robustness. MPM supplies no training transitions.
Numerical self-convergence, force agreement and policy transfer are separate acceptance questions.</p>"""
    revised = training.get("runs", {}).get("matched", {})
    if revised:
        page += (
            f'<p class="status">Revised training: {revised.get("completed_iterations", 0):,} / '
            f"{revised['iterations']:,} additional iterations. {html.escape(revised['stage'])}.</p>"
        )
    inference = read(ROOT / "evaluation-status.json")
    if inference.get("active_case"):
        progress = read(ROOT / "evaluations" / inference["active_case"] / "progress.json")
        detail = ""
        if progress:
            detail = f" {progress['completed_steps']} / {progress['total_steps']} simulation frames completed."
        page += f'<p class="status">{html.escape(inference["stage"])}.{detail}</p>'
    elif "Foot-resolution" in inference.get("stage", ""):
        page += f'<p class="status">{html.escape(inference["stage"])}.</p>'
    if drift_rows:
        page += (
            '<p class="status"><strong>Static-support and held-out force-matching gates fail.</strong> '
            "The revised policy is an exploratory transfer study; "
            "the solver is not validated as an equivalent material.</p>"
        )
    work = read(ROOT / "bearing-work.json")
    if work:
        before, after = work["before"][0], work["after"][0]
        page += (
            "<section><h2>Bearing-model correction</h2>"
            "<p>For a closed prescribed cycle, input work is −∮(F·dx + τ·dθ), where F and τ are "
            "material reactions. A memoryless resistive law must dissipate, rather than create, cycle energy.</p>"
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
Only particle reactions contribute to the measured foot wrench.</p>
<img src="probe-geometry.png" alt="G1 foot collider dimensions, axes and ankle moment reference"
style="max-width:800px">"""
    page += f'<p class="status">{html.escape(foot.get("stage", "Preparing measurements"))}</p>'
    page += """<p>Each loaded phase must change by less than 10% in both mean force and mean torque.
Normalization floors of 20 N and 1 Nm give allowances of 2 N and 0.1 Nm near zero.
An isolated-foot convergence check does not establish whole-robot coupling convergence.</p>
<table><tr><th>Refinement</th><th>Largest force change</th><th>Largest torque change</th><th>Gate</th></tr>"""
    page += "".join(gate_rows) + '</table><img src="foot-convergence.png" alt="G1 foot force and torque refinement">'
    page += "</section>" + foot_quadrature_section() + go2_resolution_section()
    page += "<section><h2>Support under an applied load</h2>"
    page += """<p>A vertical prismatic joint fixes lateral translation and foot orientation. Inertial mass and downward
load are specified independently. The bed is prepared with the foot fixed, then the foot is released.
Total support is inferred from vertical momentum balance and includes backing-floor contact if reached.
Final speed and support average the last 0.5 s.</p>
<p>A prescribed zero-velocity hold and a freely loaded equilibrium impose different boundary conditions.
The unloaded hold alone cannot determine the admissible static support under body weight.</p>
<p>The graph compares a 158.13 N load with 16 kg inertial mass throughout.</p>
<img src="loaded-settling.png" alt="Loaded foot depth, speed and support">
<details><summary>All loaded-foot fixtures, including held-out loads and coupling checks</summary>
<table><tr><th>Fixture</th><th>Final sole depth [mm]</th><th>Final mean speed [m/s]</th>
<th>Final mean support [N]</th><th>Backing-floor proximity</th></tr>"""
    page += loaded_rows + "</table></details></section>"
    if drift_rows:
        page += (
            """<section id="static"><h2>Static support remains unvalidated</h2>
<p><strong>The short-time depth fit does not establish static equilibrium.</strong> Under the same
158.13 N load, corrected RFT continues to creep. At the training timestep (four substeps), it reaches
the backing floor around 7.7 s. At ten seconds, sole depths are approximately 118.1, 109.9, 100.2 and
94.7 mm with 4, 8, 16 and 32 substeps respectively. Early drift approximately halves with each timestep
refinement. This implicates explicit near-rest coupling; the force-law correction alone does not resolve it.</p>
<p>The earlier 4.1% agreement is a comparison at two seconds. The revised training profile remains
exploratory, and prolonged static support is a failed acceptance gate. Finite-layer locomotion can draw
support from the rigid backing floor. An implicit sticking/reaction treatment requires separate solver work.</p>
<p>This load-controlled fixture already uses a 20 mm maximum triangle edge (512 sole triangles).
The observed creep is present with refined foot quadrature.</p>
<p>The prolonged-hold criterion allows at most 2 mm additional penetration between 2 and 10 s,
with no backing-floor proximity.</p>
<p>The 10 mm MPM control changes by approximately 0.03 mm between 2 and 10 s. Its depth is biased
by the coarse grid; this additional run checks long-hold stability. The finer calibration references
are reported separately.</p>
<img src="static-drift.png" alt="Long loaded holds reveal timestep-dependent RFT creep">
<table><tr><th>Fixture</th><th>Timestep [ms]</th><th>Depth at 2 s [mm]</th><th>Depth at 10 s [mm]</th>
<th>Early drift [mm/s]</th><th>Backing-floor proximity</th></tr>"""
            + drift_rows
            + "</table></section>"
        )
    page += "<section><h2>RFT force approximation</h2>"
    verification = read(ROOT / "diagnostics/loaded-refinement.json")
    if verification.get("mpm_coupling_relative_depth_change") is not None:
        page += (
            "<p>Loaded-foot depth changes by "
            f"{verification['mpm_grid_relative_depth_change']:.1%} for the 5 → 3.33 mm MPM grid refinement, "
            f"{verification['mpm_time_relative_depth_change']:.1%} when doubling MPM substeps at 5 mm, and "
            f"{verification['mpm_coupling_relative_depth_change']:.1%} for one → four coupling passes at 10 mm. "
            "Each passes the 10% depth-change gate for this fixture. Corrected RFT changes by "
            f"{verification['rft_time_relative_depth_change']:.1%} for four → eight substeps.</p>"
        )
    if calibration:
        c = calibration["selected"]
        outcome = "passed" if calibration["holdout_accepted"] else "failed"
        page += f"<p><strong>Held-out force matching: {outcome}.</strong> "
        page += "The candidate remains an approximation; fitting a subset does not establish material equivalence.</p>"
        page += (
            f"<p>Weighted moving-wrench fit error is {c['fit_normalized_error']:.1%}. "
            "The held-out gate allows at most 20% error in each phase's mean force and torque vectors, "
            "using 20 N and 1 Nm normalization floors near zero.</p>"
        )
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
                f"<p>Calibration load: {loaded['settings']['load']:.2f} N. At two seconds, corrected RFT reaches "
                f"{1000 * loaded['final_sole_depth_m']:.1f} mm versus "
                f"{1000 * calibration['loaded_measurement']['final_sole_depth_m']:.1f} mm in fine MPM. "
                "The fitted interface ratio is an effective RFT parameter; MPM retains ratio 1.0.</p>"
            )
        page += f'<img src="{fit_path}/force-fit.png" alt="Original and fitted RFT versus MPM wrenches">'
        low_mpm = read(ROOT / "loaded/mpm-5mm-load-80/summary.json")
        low_rft = read(ROOT / "loaded/rft-calibrated-load-80/summary.json")
        if low_mpm and low_rft:
            error = abs(low_rft["final_sole_depth_m"] / low_mpm["final_sole_depth_m"] - 1)
            page += (
                "<p>At the held-out 80 N load and two seconds, RFT reaches "
                f"{1000 * low_rft['final_sole_depth_m']:.1f} mm "
                f"versus {1000 * low_mpm['final_sole_depth_m']:.1f} mm in 5 mm MPM, a {error:.1%} difference. "
                "At 316.26 N, RFT reaches the backing floor; MPM settles about 2.7 mm above it. "
                "The high-load case is strongly affected by the finite substrate.</p>"
            )
        phase_rows = []
        for row in calibration["phases"]:
            phase_rows.append(
                f"<tr><td>{row['phase'].replace('_', ' ')}</td><td>{row['split']}</td>"
                f"<td>{row['measured_wrench'][2]:.1f}</td><td>{row['original_wrench'][2]:.1f}</td>"
                f"<td>{row['fitted_wrench'][2]:.1f}</td><td>{row['force_error']:.1%}</td>"
                f"<td>{row['torque_error']:.1%}</td></tr>"
            )
        page += (
            """<details><summary>Force measurements by phase</summary><table><tr><th>Phase</th><th>Split</th>
<th>MPM Fz [N]</th><th>Original RFT Fz [N]</th><th>Candidate RFT Fz [N]</th>
<th>Mean force-vector error</th><th>Mean torque-vector error</th></tr>"""
            + "".join(phase_rows)
            + "</table></details>"
        )
    else:
        page += "<p>Calibration awaits the loaded-foot reference.</p>"
    page += '</section><section id="training"><h2>Training at full material difficulty</h2>'
    page += """<p>The original-model control and force-informed variant start from the same original RFT
checkpoint, with reset optimizer state, seed 42 and 4,096 environments. Each continuation receives
1,000 additional PPO iterations with full material distributions from its first episode.
Forward commands remain 0.1–1.0 m/s; there is no command curriculum.
The force-informed model and wider randomization form an exploratory combined change.</p>"""
    page += (
        "<p>The accepted rigid baseline received 1,500 iterations from scratch. Both continued RFT policies "
        "already include that rigid pretraining and 1,500 earlier RFT adaptation iterations, "
        "then receive the 1,000-iteration continuation above. Their total training exposure differs "
        "from the rigid baseline.</p>"
    )
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
    convergence = read(ROOT / "training-convergence.json")
    if convergence:
        c = convergence["runs"]["matched"]
        change = c["Train/mean_reward"]["relative_change"]
        page += (
            f"<p>Over the final two 100-iteration windows, revised-policy mean return changes by {change:+.1%}; "
            f"final mean episode length is {c['Train/mean_episode_length']['final100_mean']:.1f} / 500 steps. "
            "These describe a late training trend, rather than prove optimality or robustness. "
            "The variants sample different material distributions; "
            "held-out rollouts provide separate acceptance checks.</p>"
        )
    profile = read(ROOT / "rft-profile.json")
    if profile:
        page += (
            "<details><summary>Force-informed material settings</summary><pre>"
            + html.escape(json.dumps(profile, indent=2))
            + "</pre></details>"
        )
    page += "</section>"
    evaluations = read(ROOT / "evaluation-status.json")
    for filename in ("native-evaluation-status.json", "matched-mpm-evaluation-status.json"):
        extra = read(ROOT / filename)
        if extra:
            combined = {r["name"]: r for r in evaluations.get("cases", [])}
            combined.update({r["name"]: r for r in extra.get("cases", [])})
            evaluations["cases"] = list(combined.values())
            evaluations["validation"] = evaluations.get("validation", {}) | extra.get("validation", {})
    if evaluations:
        policy_plots(evaluations.get("cases", []))
        path_plot(evaluations.get("cases", []))
        material_plot(evaluations.get("cases", []))
        reference_rows = reference_plot(evaluations.get("cases", []))
        page += "<section><h2>Frozen-policy acceptance</h2>"
        gates = evaluations.get("validation", {})
        for policy, results in gates.items():
            page += f"<p><strong>{LABELS[policy]}: {sum(results.values())}/{len(results)} gates passed.</strong></p>"
            page += (
                "<ul>"
                + "".join(
                    f"<li>{html.escape(name.replace('_', ' '))}: {'Pass' if value else 'Fail'}</li>"
                    for name, value in results.items()
                )
                + "</ul>"
            )
        page += """<p>Checks retain the original thresholds: no body failures or foot-placement violations,
mean planar error below 0.10 m/s, standing error below 0.03 m/s, at least three steps per foot,
no rigid capacity overflow and full material coverage. The 0.10 m/s loaded-foot velocity criterion is
a contact-conditioned speed proxy, rather than measured grain-relative slip. Randomized validation uses
three independent reset banks of 128 environments; nominal trials use 16 environments for 20 s.</p></section>
<section><h2>Whole-robot MPM inference</h2>
<p>These frozen-policy transfer trials use a 20 mm APIC grid, eight particles per voxel, eight rigid/material
substeps, up to 200 stress iterations, one coupling pass and full feedback (1.0). The bed is 5 × 2 m and
prepared for 0.5 s. Bulk/interface friction use the same invariant convention; MPM interface/internal
ratio is 1.0. The force-informed RFT interface ratio is an effective fitted parameter.</p>
<p><strong>This feasible whole-robot grid has not established force convergence.</strong> Fine isolated-foot
convergence cannot be substituted for a converged whole-biped reference. Results diagnose numerical
transfer under the disclosed configuration, rather than validate real sand. The material-bank diagnostics
run for up to 3 s, ending 0.5 s after a first body failure. Avoiding a threshold for 3 s does not establish
long-duration stability. The showcase recordings below run for a full 10 s in a 4 × 8 m bed and continue
after a fall without resets or an early stop. All other nominal material and solver settings remain the same.
Videos show actual simulated frames; each duration is stated beside its recording. An initial 12 × 2 m
attempt drifted sideways out of the bed at 4.0 s before a body threshold at 5.4 s; it is excluded from
the main comparison. The wider domain checks this boundary confound. Every selected recording must
remain within the conservative base-position interior for its full duration.</p>
<p>The material bank contains nominal (118 mm, 1,600 kg/m³, μ = 0.4), shallow/loose
(60 mm, 1,400 kg/m³, μ = 0.3) and dense (118 mm, 1,800 kg/m³, μ = 0.5) beds. Physical inputs are
within the RFT reset ranges, but equal density/friction/depth does not imply equivalent force response.</p>
<img src="material-transfer.png" alt="Failure time and tracking error across three MPM materials and two commands">
<p>A bar reaching 3 s means the short observation window ended; it does not indicate a failure at 3 s.
The tracking bars average only measurements before the first threshold, including acceleration.
The detailed table also retains full-rollout errors. Each material/command/policy combination is
one deterministic trial. These are distinct conditions, rather than repeated trials supporting statistical claims.</p>
<p>Both continued RFT policies avoid the body thresholds in all three 0.50 m/s material trials.
At 1.00 m/s, the revised policy reaches a threshold in all three beds; the extra-training control
does so in two. The rigid policy also avoids the threshold in the shallow/loose 0.50 m/s bed.
The corrected profile therefore does not demonstrate better overall MPM robustness than the
extra-training control in this bank. Correction, fit and wider randomization change together in
the revised variant; this comparison cannot assign their individual effects.</p>
</section>"""
        page += support_section(support)
        if reference_rows:
            page += (
                """<section><h2>Whole-robot grid sensitivity</h2>
<p>The revised policy is also tested at 0.50 m/s on a fixed 2.4 × 0.8 m bed with 20, 10 and 8 mm grids,
eight particles per voxel and eight substeps. These trials last up to 2 s, ending 0.5 s after a failure.
The narrower bed bounds memory use and differs from the main material bank.</p>
<p>This measures closed-loop policy sensitivity. Force convergence requires identical prescribed motions;
the foot tests provide that separate check. These whole-robot trajectories do not establish a converged
full-biped reference, especially when a boundary exit is recorded.</p>
<img src="whole-robot-refinement.png" alt="Whole-robot MPM trajectories at three grid resolutions">
<table><tr><th>Grid</th><th>Duration [s]</th><th>First body failure</th><th>Planar error [m/s]</th>
<th>First base-position boundary exit [s]</th></tr>"""
                + reference_rows
                + "</table></section>"
            )
        nominal = {r["name"]: r for r in evaluations.get("cases", []) if "_mpm_nominal_" in r["name"]}
        if nominal:
            page += """<section><h2>Nominal MPM: checkpoints under identical settings</h2>
<p>Completed checkpoints are compared using the same MPM settings. Original-checkpoint re-evaluation
is pending while the coarse-policy batch is stopped for the foot-resolution study. The two continued policies
share their initial weights and exposure; the corrected variant combines the bearing correction,
force-informed parameters and wider reset ranges. Each comparison remains a single short trial.</p>
<table><tr><th>Policy</th><th>Total PPO iterations</th><th>Command [m/s]</th><th>Duration [s]</th>
<th>First body failure</th><th>Pre-failure planar error [m/s]</th></tr>"""
            for policy, exposure in (("rigid", 1500), ("initial", 3000), ("control", 4000), ("matched", 4000)):
                for speed in (0.5, 1.0):
                    row = nominal.get(f"{policy}_mpm_nominal_{speed:g}")
                    if row is None:
                        continue
                    s = row["summary"]
                    first = s["first_failure_seconds"]
                    failure = "None" if first is None else f"{first:.2f} s"
                    error = s.get("mean_pre_failure_velocity_error")
                    error_text = "—" if error is None else f"{error:.3f}"
                    page += (
                        f"<tr><td>{LABELS[policy]}</td><td>{exposure:,}</td><td>{speed:.2f}</td>"
                        f"<td>{s['simulated_seconds']:.2f}</td><td>{failure}</td><td>{error_text}</td></tr>"
                    )
            page += "</table></section>"
        for speed in (0.5, 1.0):
            page += (
                f'<section id="comparison-{int(speed * 100):03d}"><h2>Comparison at {speed:.2f} m/s</h2>'
                f'<img src="comparison-{speed:g}.png" '
                'alt="Forward velocity, base clearance and tilt comparisons">'
                "<p>Dotted lines mark first body-failure thresholds. Curves continue after failure; "
                "different durations are stated beside each recording.</p></section>"
            )
        page += "<details><summary>All held-out measurements</summary>"
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
            "Rollouts continue without resets.</p></section></details>"
        )
        page += (
            '<section><h2>World paths of the recorded policies</h2><img src="world-paths.png" '
            'alt="World x/y trajectories and finite MPM bed boundaries at both commands">'
            "<p>Commanded forward speed is measured in the robot frame. Heading drift can therefore "
            "produce a curved world path even with modest forward-speed error. Dashed rectangles "
            "show the MPM domains; native rigid/RFT fields extend beyond those rectangles. "
            "These paths use the same recordings shown below.</p></section>"
        )
        coverage = foot_domain_coverage(evaluations.get("cases", []))
        page += render_policy_videos(evaluations.get("cases", []), coverage)
    else:
        page += "<section><h2>Held-out inference</h2><p>Native and MPM comparisons follow training.</p></section>"
    page += (
        '<script>setInterval(()=>{if(![...document.querySelectorAll("video")]'
        ".some(v=>!v.paused&&!v.ended))location.reload()},30000)</script>"
    )
    temporary = ROOT / "analysis.tmp"
    temporary.write_text(page)
    temporary.replace(ROOT / "analysis.html")
    if foot.get("complete"):
        (ROOT / "index.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        publish()
        if not args.watch:
            break
        time.sleep(30)
