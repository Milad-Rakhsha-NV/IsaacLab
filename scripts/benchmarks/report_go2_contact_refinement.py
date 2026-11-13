# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the Go2 contact-resolution investigation with portable numerical records."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read(path):
    """Read an existing numerical record without inventing missing results."""
    return json.loads(path.read_text())


def geometry_plot(output, profile, inventory, recorded_pose):
    """Draw one leg's actual parameter meshes and preserve a reproducible geometry export."""
    builder = newton.ModelBuilder()
    body = builder.add_body(label="/World/envs/env_0/Robot/FL_calf")
    for row in inventory:
        if not row["body_label"].endswith("/FL_calf"):
            continue
        p = row["local_pose"]
        builder.add_shape_cylinder(
            body,
            radius=row["scale"][0],
            half_height=row["scale"][1],
            xform=wp.transform(wp.vec3(p[:3]), wp.quat(p[3:])),
            label=row["label"],
        )
    foot = builder.add_body(label="/World/envs/env_0/Robot/FL_foot")
    row = next(r for r in inventory if r["body_label"].endswith("/FL_foot"))
    calf_row = next(r for r in inventory if r["body_label"].endswith("/FL_calf"))

    def transform(p):
        return wp.transform(wp.vec3(p[:3]), wp.quat(p[3:]))

    relative = wp.mul(
        wp.transform_inverse(transform(recorded_pose[calf_row["body"]])), transform(recorded_pose[row["body"]])
    )
    p = row["local_pose"]
    # Place the fixed sphere in the calf frame for a single-leg diagram.
    builder.add_shape_sphere(foot, radius=row["scale"][0], xform=wp.mul(relative, transform(p)), label=row["label"])
    model = builder.finalize(device="cpu")
    surfaces, rows = collision_surfaces(model, ".*", 12, resolutions=profile["solver"]["surface_resolution"])
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121, projection="3d")
    colors = ("#cbd5e1", "#38bdf8", "#0284c7", "#f59e0b")
    arrays = {}
    for i, ((_, mesh), row) in enumerate(zip(surfaces, rows, strict=True)):
        triangles = mesh.vertices[mesh.indices.reshape(-1, 3)]
        arrays[f"triangles_{i}"] = triangles
        ax.add_collection3d(
            Poly3DCollection(triangles * 1000, facecolor=colors[i], edgecolor="#334155", linewidth=0.35, alpha=0.7)
        )
    ax.set(
        xlim=(-30, 50),
        ylim=(-35, 35),
        zlim=(-245, 0),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="Calf-local z [mm]",
        title="One Go2 lower leg",
    )
    ax.set_box_aspect((80, 70, 245))
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_yticks([-20, 0, 20])
    ax.set_zlabel("Calf-local z [mm]", labelpad=12)
    ax.view_init(elev=8, azim=-65)
    axis = fig.add_subplot(122)
    axis.axis("off")
    axis.text(
        0,
        0.9,
        "128 triangles over the entire spherical foot\n64 on its lower hemisphere\nRadius: 22 mm\n\n"
        "Lower calf cylinders: 128 and 96 triangles\nUpper calf cylinder: 48 triangles, unchanged\n\n"
        "All leg and thigh colliders remain coupled.\nOther surfaces retain their old resolution.\n\n"
        "Triangles define integration patches.\nAdditional Gauss points use exact curved\n"
        "positions, normals and area Jacobians.\n\n"
        "The drawing uses the recorded fixed foot\ntransform relative to the calf link.",
        va="top",
        fontsize=11,
        linespacing=1.6,
    )
    fig.tight_layout()
    fig.savefig(output / "contact-geometry.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    np.savez_compressed(output / "contact-geometry.npz", **arrays)


def plots(output, motion, legacy, refined, support):
    """Separate integration accuracy, computational cost and prolonged support."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    selected = [r for r in motion["results"] if r["candidate"] in ("legacy", "faceted128", "selected")]
    for ax, component in zip(axes, ("force", "torque"), strict=True):
        for row, label in zip(selected, ("Previous surface", "128 flat facets", "128 curved patches"), strict=True):
            heights = sorted({r["height"] for r in row["by_height"]})
            values = [max(r["feet"][component]["p95"] for r in row["by_height"] if r["height"] == h) for h in heights]
            ax.plot(np.array(heights) * 1000, np.array(values) * 100, "o-", label=label)
        ax.axhline(2 if component == "force" else 5, color="#334155", linestyle=":", label="Proposed percentile target")
        ax.set(
            xlabel="Layer height above floor [mm]",
            ylabel=f"Foot {component} difference [%]",
            title="Worst material-case 95th percentile",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Identical recorded Go2 motion at 0.5 and 1.0 m/s; higher-order curved reference")
    fig.tight_layout()
    fig.savefig(output / "contact-accuracy.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row, label in ((legacy, "Previous"), (refined, "Refined")):
        times = row["common_motion_rft_evaluation_seconds"]
        x = np.array([0, 1]) + (-0.18 if label == "Previous" else 0.18)
        axes[0].bar(x, [times[k] * 1000 for k in ("shallow", "deep")], width=0.36, label=label)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["28 mm nominal burial", "118 mm nominal burial"],
        ylabel="4096-world RFT evaluation [ms]",
        title="Common recorded poses and twists",
    )
    axes[0].legend()
    axes[1].bar(["Previous", "Refined"], [r["environment_steps_per_second"] for r in (legacy, refined)])
    axes[1].set(ylabel="Environment transitions / wall second", title="Native Go2 stepping; zero actions")
    fig.tight_layout()
    fig.savefig(output / "contact-cost.png", dpi=170)
    plt.close(fig)
    data = np.load(output / "static-support.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, load_index in zip(axes, (0, 1), strict=True):
        for row in support["results"]:
            history = data[f"substeps_{row['substeps']}"]
            ax.plot(history[:, 0], 1000 * history[:, load_index + 1], label=f"{row['substeps']} substeps")
        ax.axhline(118, linestyle="--", color="#334155", label="Rigid substrate")
        ax.set(
            xlabel="Time [s]",
            ylabel="Sole penetration [mm]",
            title=f"Applied vertical load: {support['loads_N'][load_index]:.2f} N",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("128-triangle Go2 sphere: the static-support gate still fails")
    fig.tight_layout()
    fig.savefig(output / "static-support.png", dpi=170)
    plt.close(fig)


def main(args):
    """Archive selected measurements and publish an honest, self-contained status report."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    files = [
        "recorded-motion.json",
        "recorded-motion.npz",
        "performance-legacy.json",
        "performance-contact128.json",
        "static-support.json",
        "static-support.npz",
        "validation-status.json",
    ]
    if (args.source / "recorded-motion-reference.json").exists():
        files += ["recorded-motion-reference.json", "recorded-motion-reference.npz"]
    for name in files:
        shutil.copy2(args.source / name, output / name)
    for suffix in ("json", "npz"):
        shutil.copy2(args.source / f"curved-sphere/convergence.{suffix}", output / f"sphere-convergence.{suffix}")
    for speed in ("0.5", "1"):
        shutil.copy2(args.source / f"motion-rigid-{speed}/rollout.npz", output / f"motion-rigid-{speed}.npz")
        summary = (args.source / f"motion-rigid-{speed}/summary.json").read_text()
        (output / f"motion-rigid-{speed}-summary.json").write_text(summary.rstrip() + "\n")
    profile = read(args.profile)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    source = "recorded-motion-reference.json" if "recorded-motion-reference.json" in files else "recorded-motion.json"
    motion, legacy, refined, support = [
        read(output / name)
        for name in (source, "performance-legacy.json", "performance-contact128.json", "static-support.json")
    ]
    poses = np.load(output / source.replace(".json", ".npz"))["body_pose"]
    geometry_plot(output, profile, motion["source_inventory"], poses[0])
    plots(output, motion, legacy, refined, support)
    selected = next(r for r in motion["results"] if r["candidate"] == "selected")
    accuracy_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{row['feet']['force']['pooled']:.2%}</td>"
        f"<td>{row['feet']['torque']['pooled']:.2%}</td>"
        f"<td>{row['feet']['force']['maximum']:.1%}</td>"
        f"<td>{row['feet']['torque']['maximum']:.1%}</td></tr>"
        for row, label in zip(
            [r for r in motion["results"] if r["candidate"] in ("legacy", "faceted128", "selected")],
            ("Previous surface", "128 flat facets", "128 curved patches"),
            strict=True,
        )
    )
    note = (
        f"Selected foot integration: {selected['feet']['force']['pooled']:.2%} pooled force and "
        f"{selected['feet']['torque']['pooled']:.2%} pooled torque difference from the reference. "
        "Individual outliers remain; these are numerical differences, not material-model accuracy."
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Go2 contact-resolution follow-up</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}}
section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}img{{width:100%}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:9px;border-bottom:1px solid #cbd5e1;text-align:left}}
.status{{padding:18px;background:#fef3c7}}a{{color:#1d4ed8}}
code{{overflow-wrap:anywhere}}.table{{overflow-x:auto}}</style>
<h1>Go2 contact-resolution follow-up</h1>
<p><a href="low-speed/">Latest: fixed-link boundary correction and matched MPM foot validation</a></p>
<p><a href="../go2-rft-mpm/report/">Original policy comparison</a> ·
<a href="README.md">Methods and reproduction</a></p>
<p class="status"><strong>Geometry and numerical implementation complete. New training has not started.</strong>
The long-hold fixture still creeps or reaches the rigid floor. Stable near-rest reaction selection and material
matching remain required before another training run. G1 work is paused.</p>
<section><h2>Contact geometry</h2><p>The entire spherical foot has <strong>128 triangles</strong>, including 64 on
its lower hemisphere. The two lower calf cylinders have 128 and 96 triangles each. The upper calf, thighs,
hips and torso keep their previous resolution. All selected leg and thigh surfaces remain coupled.</p>
<p>Collision dimensions, poses, mass and inertia are preserved. The robot has {refined["triangles_per_robot"]:,}
retained integration triangles, compared with {legacy["triangles_per_robot"]:,} previously. Flat surface
subdivision remains separate from curved-surface accuracy.</p>
<img src="contact-geometry.png" alt="Go2 spherical foot and lower calf integration patches">
<p>The selected rule uses 36 Gauss points per foot triangle and nine per refined calf triangle. These points
lie on the analytic curved collider and use its true normal and area Jacobian. Shared templates avoid
duplicating point geometry across 4,096 environments. Triangle count and integration-point count are distinct.</p>
</section>
<section><h2>Accuracy on recorded walking motion</h2><p>{html.escape(note)}</p>
<p>Both rigid-control recordings last ten seconds, with commands 0.5 and 1.0 m/s. The audit selects 32 frames
from each and uses the exact recorded world-space COM twists. It crosses four heights (3, 30, 60, 118 mm),
two bulk-friction values and two interface ratios. Density is 1,800 kg/m³ and strength multiplier 1.2.
The same poses, collider unions and material cases are used for every candidate.</p>
<div class="table"><table><tr><th>Foot integration</th><th>Pooled force</th><th>Pooled torque</th>
<th>Maximum normalized force</th><th>Maximum normalized torque</th></tr>{accuracy_rows}</table></div>
<p>Errors use denominator floors of 1 N and 0.01 N·m. Percentiles and maxima serve different purposes:
large percentages on tiny torques remain visible in the raw records. Reference integration refines the
same curved parameter patches; unselected upper surfaces remain coarse. This audit supplies no independent
physical validation of sand or policy robustness. Union rejection covers only colliders of the same body;
it does not resolve shielding between articulated links.</p>
<p>Selected calf integration has {selected["calves"]["force"]["pooled"]:.2%} pooled force and
{selected["calves"]["torque"]["pooled"]:.2%} pooled torque difference. Its maximum normalized differences are
{selected["calves"]["force"]["maximum"]:.1%} and {selected["calves"]["torque"]["maximum"]:.1%}.</p>
<p>The worst material-case calf force percentile is about 3.2%, above the proposed 2% target.
The profile therefore remains a candidate. Foot percentile accuracy improves, while strict pointwise
accuracy, calf resolution, cost and static support still require work.</p>
<img src="contact-accuracy.png" alt="Foot force and torque integration errors by layer height"></section>
<section><h2>Cost at 4,096 environments</h2><p>Both profiles run through the native Go2 environment on one
RTX PRO 4500 GPU. The direct RFT timings use identical recorded poses and twists at two immersion offsets.
The end-to-end measurements step zero actions and can follow different trajectories; they exclude PPO updates.</p>
<p>Measured transitions per second: previous {legacy["environment_steps_per_second"]:,.0f};
refined {refined["environment_steps_per_second"]:,.0f}. Reported total GPU use is
{refined["gpu_used_bytes"] / 2**30:.2f} GiB, including the CUDA context and other processes.</p>
<img src="contact-cost.png" alt="RFT evaluation and native environment throughput at 4096 worlds"></section>
<section><h2>Static support remains unresolved</h2><p>A vertical prismatic fixture applies 18.42, 36.83 and
55.25 N to separate 1 kg intruders. The nominal load is one quarter of Go2's weight. Each foot uses the new
128-triangle curved integration, in a 118 mm layer, for ten seconds at three timesteps. The two supported
loads start at rest at a depth with 110% of the required bearing capacity, independently checked against
the analytic sphere integral. This removes landing impact from the support test.</p>
<p>The nominal-load foot reaches the rigid substrate. The half-load foot still creeps more than the proposed
2 mm allowance from two to ten seconds. The overload is retained as a diagnostic. No matched Go2 MPM hold
reference is claimed here. More quadrature does not resolve the explicit near-rest reaction selection.</p>
<img src="static-support.png" alt="Load-controlled Go2 foot settling versus timestep">
<p>The next solver stage is a bounded near-rest reaction treatment coupled to the articulated dynamics,
followed by prolonged holds, load/unload work, timestep convergence and MPM force matching. Reset randomization
and new Go2 training follow those gates. The existing videos retain their original physics and policy weights.</p>
</section>
<section><h2>Verification and records</h2><p>Newton: 21 CPU/GPU RFT cases. Isaac Lab: six geometry and native
coupling cases, including masked reset, all-leg binding, granular impulse balance and the rigid zero-layer limit.
Disabling curved integration makes the independent analytic-load regression fail. Both repositories pass their
formatting and lint checks.</p><p><a href="recorded-motion.json">Motion audit</a> ·
<a href="static-support.json">Support fixture</a> · <a href="profile.json">Candidate profile</a> ·
<a href="manifest.json">Artifact checksums</a></p></section></html>"""
    (output / "index.html").write_text(page + "\n")
    readme = f"""# Go2 contact-resolution follow-up

{note}

The profile is a numerical candidate, with 128 triangles over each spherical foot and selective lower-calf
refinement. It is not cleared for training: prolonged support fails. G1 remains on hold. The original Go2
policy recordings retain their original implementation and geometry.

Open `index.html` alongside its PNGs. JSON and NPZ files retain the measurements, common motion samples,
material settings, source hashes and geometry. `manifest.json` verifies every archived artifact.

Run follow-up audits from the repository root using the current Newton pin. The archived measurements
above retain their original revision; these commands use a separate runtime:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/rft-study/go2-fixed-union-venv"
uv sync --locked
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \\
    --mode performance --legacy --performance_motion results/go2-contact-refinement/motion-rigid-1.npz \\
    --output /tmp/legacy.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \\
    --mode performance --performance_motion results/go2-contact-refinement/motion-rigid-1.npz \\
    --output /tmp/refined.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \\
    --samples 32 --orders 16 24 \\
    --trajectories results/go2-contact-refinement/motion-rigid-0.5.npz \\
                   results/go2-contact-refinement/motion-rigid-1.npz \\
    --output /tmp/motion.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_rft_support.py --output /tmp/support.json
```

The archive includes both complete ten-second motion recordings and their summaries (including body COM
locations), plus the 64 selected exact poses and twists. Source paths and SHA-256 hashes are in the JSON.
The checkpoint is not required to reproduce the prescribed-motion audit. The robot asset and numerical
profile remain fixed during that audit; no policy inference occurs in the force comparison.
"""
    (output / "README.md").write_text(readme)
    manifest = {
        p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("logs/go2_granular/go2/refinement"))
    parser.add_argument("--profile", type=Path, default=Path("scripts/benchmarks/configs/go2_rft_contact_128.json"))
    parser.add_argument("--output", type=Path, default=Path("results/go2-contact-refinement"))
    main(parser.parse_args())
