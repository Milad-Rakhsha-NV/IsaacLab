# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the Go2 low-speed integration and matched MPM probe measurements."""

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
import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from matplotlib.patches import Patch
from newton.solvers import SolverRFT

MPM_BASELINE_REVISION = "6cf5b879d576c19279183ee9da0dc4de039a080f"
FIXED_UNION_REVISION = "f59677bd8c5037d68d4196c323ea0b4459c18a9c"
CASES = (
    ("preloaded-mpm-18N-10mm", "118 mm bed · 10 mm cells · 4 passes", MPM_BASELINE_REVISION),
    ("preloaded-mpm-18N-10mm-p1", "118 mm bed · 10 mm cells · 1 pass", MPM_BASELINE_REVISION),
    ("preloaded-mpm-18N-5mm-p1", "118 mm bed · 5 mm cells · 1 pass", MPM_BASELINE_REVISION),
    ("preloaded-mpm-18N-10mm-deep", "236 mm bed · 10 mm cells · 1 pass", MPM_BASELINE_REVISION),
    ("preloaded-mpm-18N-5mm-i400", "118 mm bed · 5 mm cells · 400-iteration cap", FIXED_UNION_REVISION),
)


def read(path):
    """Read a completed numerical record."""
    return json.loads(path.read_text())


def section_plot(output):
    """Show the actual fixed foot/calf overlap in a central collision-geometry section."""
    inventory = read(output / "fixed-union-motion.json")["source_inventory"]
    poses = np.load(output / "compound-after.npz")["body_pose"]
    calf = next(row for row in inventory if row["body_label"].endswith("/FL_calf"))
    calf_pose = wp.transform(wp.vec3(poses[calf["body"], :3]), wp.quat(poses[calf["body"], 3:]))
    x, z = np.meshgrid(np.linspace(-0.04, 0.06, 500), np.linspace(-0.245, -0.13, 550))
    points = np.stack((x, np.zeros_like(x), z), axis=-1)
    calf_distance = np.full(x.shape, np.inf)
    foot_distance = np.full(x.shape, np.inf)
    for row in inventory:
        if not row["body_label"].endswith(("/FL_calf", "/FL_foot")):
            continue
        p, s = row["local_pose"], row["scale"]
        body_pose = wp.transform(wp.vec3(poses[row["body"], :3]), wp.quat(poses[row["body"], 3:]))
        relative = wp.mul(wp.transform_inverse(calf_pose), body_pose)
        pose = wp.mul(relative, wp.transform(wp.vec3(p[:3]), wp.quat(p[3:])))
        rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
        local = (points - np.asarray(wp.transform_get_translation(pose))) @ rotation
        if row["kind"] == "SPHERE":
            foot_distance = np.minimum(foot_distance, np.linalg.norm(local, axis=-1) - s[0])
        else:
            distance = np.maximum(np.linalg.norm(local[..., :2], axis=-1) - s[0], abs(local[..., 2]) - s[1])
            calf_distance = np.minimum(calf_distance, distance)
    regions = (calf_distance < 0).astype(int) + 2 * (foot_distance < 0).astype(int)
    colors = ["#f8fafc", "#38bdf8", "#fbbf24", "#c084fc"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(1000 * x, 1000 * z, regions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=colors)
    ax.contour(1000 * x, 1000 * z, np.minimum(calf_distance, foot_distance), levels=[0], colors="#172033")
    ax.contour(
        1000 * x,
        1000 * z,
        np.where(foot_distance < 0, calf_distance, np.nan),
        levels=[0],
        colors="#b91c1c",
        linestyles="--",
        linewidths=2,
    )
    ax.set(
        aspect="equal",
        xlabel="Calf-frame x (mm)",
        ylabel="Calf-frame z (mm)",
        title="Actual Go2 collider section at y = 0",
    )
    ax.legend(
        handles=[
            Patch(color=colors[1], label="Calf"),
            Patch(color=colors[2], label="Foot"),
            Patch(color=colors[3], label="Overlap"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output / "fixed-link-section.png", dpi=170)
    plt.close(fig)


def build(args):
    """Retain raw records, fit prescribed insertion forces, and render the findings."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in [row[0] for row in CASES] + ["scalar-hold", "scalar-load-cycle", "preloaded-mpm-37N-10mm"]:
        (output / name).mkdir(exist_ok=True)
        for filename in ("trace.npz", "summary.json"):
            shutil.copy2(args.input / name / filename, output / name / filename)
    for filename in ("compound-before.json", "compound-before.npz", "compound-after.json", "compound-after.npz"):
        shutil.copy2(args.input / filename, output / filename)
    for filename in (
        "compound-comparison.json",
        "fixed-union-motion.json",
        "fixed-union-motion.npz",
        "fixed-union-performance.json",
    ):
        shutil.copy2(args.input / filename, output / filename)
    shutil.copy2("scripts/benchmarks/configs/go2_rft_fixed_union.json", output / "profile.json")
    section_plot(output)
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1, inertia=wp.mat33(np.eye(3)), label="foot")
    builder.add_shape_sphere(body, radius=0.022, label="foot/sphere", cfg=builder.ShapeConfig(density=0))
    model = builder.finalize(device="cpu")
    surfaces, _ = collision_surfaces(model, ".*", 12, resolutions=[dict(shape_pattern=".*", sphere_subdivisions=2)])
    solver = SolverRFT(model, surfaces, surface_shape_ids=[0], surface_quadrature_orders=[6])
    state = model.state()
    comparison, summaries = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, label, revision in CASES:
        data = np.load(output / name / "trace.npz")
        preload = data["preload"]
        summary = read(output / name / "summary.json")
        summaries[name] = summary
        trace = data["trace"].copy()
        # The first, historical four-pass fixture counted capture as execution.
        # Its recorded release timestamps are one 5 ms frame ahead of simulation.
        correction = 0.0 if summary.get("warmup_included_in_trace") else -0.005
        trace[:, 0] += correction
        axes[0].plot(trace[:, 0], 1000 * trace[:, 4], label=label)
        predictions = []
        for t, center_z, _ in preload:
            speed = -0.02 if t < (0.0523165 + 0.01) / 0.02 else 0.0
            state.body_q.assign([[0, 0, center_z, 0, 0, 0, 1]])
            state.body_qd.assign([[0, 0, speed, 0, 0, 0]])
            predictions.append(float(solver.evaluate(state).numpy()[0, 2]))
        predictions = np.asarray(predictions)
        depth = 0.022 - preload[:, 1]
        mask = (depth > 0.015) & (depth < 0.05) & (predictions > 0)
        strength = float(np.dot(predictions[mask], preload[mask, 2]) / np.dot(predictions[mask], predictions[mask]))
        fit_error = float(
            np.linalg.norm(preload[mask, 2] - strength * predictions[mask]) / np.linalg.norm(preload[mask, 2])
        )
        axes[1].plot(1000 * depth[mask], preload[mask, 2], label=label)
        if name == "preloaded-mpm-18N-5mm-p1":
            axes[1].plot(1000 * depth[mask], predictions[mask], "--", color="#111827", label="Nominal published RFT")
            axes[1].plot(
                1000 * depth[mask],
                strength * predictions[mask],
                ":",
                color="#111827",
                label=f"Insertion-only fit: {strength:.3f} × RFT",
            )
        comparison.append(
            dict(
                case=name,
                newton_revision=revision,
                label=label,
                final_depth_mm=1000 * summary["final_sole_depth_m"],
                insertion_strength_ratio=strength,
                insertion_fit_relative_error=fit_error,
                release_timestamp_correction_s=correction,
            )
        )
        np.savez_compressed(
            output / name / "rft-preload-comparison.npz", preload=preload, rft_force=predictions, fit_mask=mask
        )
    axes[0].set(xlabel="Time after release (s)", ylabel="Sole depth (mm)", title="18.42 N applied load")
    axes[1].set(xlabel="Sole depth during insertion (mm)", ylabel="Upward force (N)", title="Insertion at 0.02 m/s")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "mpm-reference.png", dpi=170)
    plt.close(fig)
    hold = np.load(output / "scalar-hold/trace.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, load in zip(axes, (18, 37), strict=True):
        for method, label in (
            ("bearing", "Current explicit bearing"),
            ("flow", "Explicit flowing RFT"),
            ("implicit_1d", "Implicit 1D reaction"),
        ):
            values = hold[f"{method}_4_{load}"]
            ax.plot(values[:, 0], 1000 * values[:, 1], label=label)
        ax.axhline(118, color="#991b1b", ls=":", label="118 mm reference; no floor")
        ax.set(title=f"{load} N load", xlabel="Time (s)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Sole depth (mm)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "support-integration.png", dpi=170)
    plt.close(fig)
    cycle = np.load(output / "scalar-load-cycle/trace.npz")["implicit_1d_4_18"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(cycle[:, 0], cycle[:, 4], label="Applied load", lw=2)
    axes[0].plot(cycle[:, 0], cycle[:, 3], "--", label="Selected reaction")
    axes[0].set(ylabel="Force (N)", title="Scalar reaction: hold, unload, reverse, overload, hold")
    axes[0].legend()
    axes[1].plot(cycle[:, 0], 1000 * cycle[:, 1])
    axes[1].set(xlabel="Time (s)", ylabel="Sole depth (mm)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "load-cycle.png", dpi=170)
    plt.close(fig)
    before, after = read(output / "compound-before.json"), read(output / "compound-after.json")
    labels = ("Downward translation", "Horizontal translation", "Resting bearing")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(3)
    ax.bar(
        x - 0.18,
        [row["total_force_N"][2] for row in before["cases"]],
        width=0.36,
        label="Foot and calf treated separately",
    )
    ax.bar(
        x + 0.18, [row["total_force_N"][2] for row in after["cases"]], width=0.36, label="Corrected fixed-link boundary"
    )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Total vertical force (N)",
        title="Same Go2 pose, geometry, material and motion",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "fixed-link-boundary.png", dpi=170)
    plt.close(fig)
    rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td><td>{row['final_depth_mm']:.2f} mm</td>"
        f"<td>{row['insertion_strength_ratio']:.3f}</td></tr>"
        for row in comparison
    )
    motion = read(output / "fixed-union-motion.json")
    selected = next(row for row in motion["results"] if row["candidate"] == "selected")
    performance = read(output / "fixed-union-performance.json")
    accuracy_rows = "".join(
        f"<tr><td>{label}</td><td>{100 * selected[key]['force']['pooled']:.2f}%</td>"
        f"<td>{100 * selected[key]['torque']['pooled']:.2f}%</td>"
        f"<td>{100 * selected[key]['force']['p95']:.2f}%</td></tr>"
        for label, key in (("Feet", "feet"), ("Calves", "calves"))
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Go2 — contact model validation</title>
<style>body{{
font:17px/1.6 system-ui;
max-width:1100px;
margin:36px auto;
padding:0 24px;
background:#f4f6f9;
color:#172033}}
section{{
background:white;
margin:22px 0;
padding:24px;
border-radius:12px}}
h1,h2{{
line-height:1.25}}
img{{
width:100%}}
table{{
border-collapse:collapse;
width:100%}}
td,th{{
padding:9px;
text-align:left;
border-bottom:1px solid #ddd}}
a{{
color:#0369a1}}
.status{{
padding:18px;
background:#fff3cd;
border-radius:8px}}
</style>
</head>
<body>
<a href="../index.html">Contact-resolution report</a> · <a href="../../go2-rft-mpm/report/">Go2 policy
report</a>
<h1>Go2 contact model validation</h1>
<p>13 November 2026. This update follows the GM-Loco comparison. The feet retain 128 triangles over each
complete sphere, with curved quadrature; lower-leg colliders remain coupled.</p>
<p class="status">
<strong>Geometry correction verified. New training has not started.</strong>
<a href="static-contact/">Latest: articulated RFT reaction solve, support video and MPM loading checks</a>.
The newer mode passes stationary support checks; material matching remains open. G1 remains on hold.</p>
<section>
<h2>Fixed foot/calf overlaps were adding internal forces</h2>
<p>The foot sphere and calf cylinders are separate bodies connected by a fixed joint. Their overlapping
internal surfaces were counted as exposed to sand. The correction groups enabled fixed links for containment
checks, using each collider's owning-body frame. It preserves the collision shapes, triangle counts, masses
and inertias.</p>
<img src="fixed-link-section.png" alt="Go2 foot and calf section showing their overlap">
<p>The red dashed calf boundary lies inside the foot collider. It belongs to the interior of the fixed assembly.</p>
<img src="fixed-link-boundary.png" alt="Vertical loads before and after excluding internal fixed-link
surfaces">
<p>In this prescribed buried pose, resting vertical support decreases from 186.72 to 132.70 N: 54.02 N was
supplied by internal patches. This is a controlled geometry comparison, not a new policy result. The
regression fails before the fix and passes after it for planar and curved integration. All 23 Newton RFT cases
and six Isaac Lab geometry/native-coupling cases pass.</p>
</section>
<section>
<h2>Force accuracy and cost after correcting the boundary</h2>
<p>The final profile retains 128 triangles per foot and uses order-six curved quadrature on the selected
foot and distal-calf surfaces. Other surfaces retain their previous resolution. On 64 recorded Go2
poses/twists crossed with 16 field cases, comparison with order-24 integration gives:</p>
<table><tr><th>Region</th><th>Pooled force difference</th><th>Pooled torque difference</th>
<th>Force difference, 95th percentile</th></tr>{accuracy_rows}</table>
<p>Relative differences use 1 N and 0.01 Nm denominator floors. These are numerical integration checks;
percentiles pool the recorded cases. Individual and per-case outliers, and material-response accuracy,
require separate assessment.</p>
<p>At 4096 environments the profile measures {performance["environment_steps_per_second"]:,.0f} transitions/s
with zero actions. Direct RFT evaluations of common states take
{1000 * performance["common_motion_rft_evaluation_seconds"]["shallow"]:.1f} ms in the shallow case and
{1000 * performance["common_motion_rft_evaluation_seconds"]["deep"]:.1f} ms in the deep case.
This measures simulation throughput; PPO training has not run.</p>
</section>
<section>
<h2>MPM provides sustained support, with measurable substrate influence</h2>
<p>A 44 mm-diameter spherical foot is inserted slowly to 52.3 mm sole depth, held briefly, then released under
an 18.42 N load. The probe has one vertical DOF and 1 kg inertial mass; load is applied independently. It
isolates the foot, without a calf or robot controller. MPM uses density 1600 kg/m³, internal and interface
friction 0.4, APIC transfer, zero cohesion/dilatancy and the stiff elastic limit.</p>
<img src="mpm-reference.png" alt="MPM settling at different grid sizes and bed depths, with prescribed
insertion forces">
<table>
<tr>
<th>MPM configuration</th>
<th>Final sole depth</th>
<th>Insertion-only RFT scale</th>
</tr>{rows}</table>
<p>The shallow 10 mm cases ran about ten seconds after release; the finer and deep-bed cases ran three seconds
and reached negligible final speed. All five hold the load without rigid floor contact. The deep-bed delivered
force balances the rigid foot's vertical momentum to a 95th-percentile force error of 1.4 micro-Newtons.</p>
<p>Increasing coupling passes changes depth by only 0.17 mm. Refining cells from 10 to 5 mm changes depth by
1.88 mm and the fitted insertion scale by about 9%. Doubling the MPM iteration cap from 200 to 400 on the
5 mm grid changes settled depth by about 0.08 mm and the insertion-fit scale from 0.376 to 0.373.
These results provide a numerical reference with remaining
grid uncertainty. The 236 mm bed produces greater settling, so clipping an RFT layer at its bottom does not
reproduce confinement by itself.</p>
<p>A separate 36.83 N coarse-grid probe reaches 109.43 mm sole depth after three seconds and is still settling
at 0.33 mm/s. Its 8.6 mm substrate clearance is smaller than one cell. It does not certify a converged
nominal-load hold.</p>
</section>
<section>
<h2>Equal material inputs have not matched the force response</h2>
<p>For insertion from 15 to 50 mm depth, nominal published RFT needs a fitted strength multiplier of about
0.34–0.38 in the shallow MPM cases and 0.28 in the coarse deep bed. The finer shallow fit uses 0.376 with 4.7%
normalized fitting error. This is a single-geometry, single-direction calibration result; it does not
establish a universal coefficient. Tangential sliding, unloading, lower-leg geometry and further reference
checks remain necessary before choosing the training distribution.</p>
</section>
<section>
<h2>The explicit update causes a separate numerical sinking error</h2>
<p>The RFT sphere starts at rest at an admissible depth. Explicit forces overshoot as velocity changes sign,
causing repeated velocity reversals and net sinking. In a scalar diagnostic, selecting a reaction inside the
two directional RFT limits satisfies momentum balance and holds an admissible load.</p>
<img src="support-integration.png" alt="Explicit RFT sinking compared with the scalar implicit reaction">
<p>The implicit 1D case has zero drift and zero positive granular work in these holds at four and eight
substeps. It uses an unbounded bed. The 118 mm line is only a reference. It is not an articulated solver or an
MPM validation.</p>
<img src="load-cycle.png" alt="Load removal, reversal and overload in the scalar reaction diagnostic">
</section>
<section>
<h2>Implementation and training status</h2>
<p>The fixed-link boundary bug is corrected and regression-tested. The newer
<a href="static-contact/">projected articulated reaction mode</a> is implemented and passes tested stationary
support checks. Material matching remains open. Published RFT coefficients are retained; the new support
video uses fixed PD joint targets. No replacement walking policy has been trained.</p>
<p>The next policy study remains one 4096-environment run, full forward commands through 1.0 m/s, reset
randomization and a material-only curriculum. Physics acceptance comes first. Short observation history and
wider calibrated material variation follow that acceptance.</p>
<p>
<a href="summary.json">Numerical summary</a> · <a href="manifest.json">Artifact hashes</a>
</p>
</section>
</body>
</html>"""
    (output / "index.html").write_text(page.rstrip() + "\n")
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                mpm=comparison,
                mpm_revisions={row["case"]: row["newton_revision"] for row in comparison},
                newton_revision=args.newton_revision,
                newton_kernel_sha256=hashlib.sha256(
                    (Path(newton.__file__).parent / "_src/solvers/rft/kernels.py").read_bytes()
                ).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training_ready=False,
            ),
            indent=2,
        )
        + "\n"
    )
    files = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("logs/go2_granular/go2/low-speed"))
    parser.add_argument("--output", type=Path, default=Path("results/go2-contact-refinement/low-speed"))
    parser.add_argument("--newton_revision", required=True)
    build(parser.parse_args())
