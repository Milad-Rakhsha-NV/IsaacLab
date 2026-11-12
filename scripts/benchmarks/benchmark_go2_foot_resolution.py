# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolate curved Go2 foot geometry and RFT quadrature on prescribed motion."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import matplotlib
import newton

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import warp as wp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from newton.solvers import SolverRFT


def plot_resolution(output, meshes, results):
    """Show the actual curved contact surfaces and their force/torque convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "3d"})
    for ax, key in zip(axes, (("uv", 12), ("icosphere", 3), ("icosphere", 4)), strict=True):
        triangles = meshes[key]
        lower = triangles[triangles.mean(axis=1)[:, 2] < 0] * 1000
        row = next(r for r in results if (r["kind"], r["resolution"]) == key)
        ax.add_collection3d(Poly3DCollection(lower, facecolor="#dbeafe", edgecolor="#1d4ed8", linewidth=0.3))
        ax.set(
            xlim=(-24, 24),
            ylim=(-24, 24),
            zlim=(-24, 2),
            title=f"{row['label']} · {1000 * row['maximum_edge_m']:.1f} mm edge\n{len(lower)} lower-half triangles",
            xlabel="x [mm]",
            ylabel="y [mm]",
            zlabel="z [mm]",
        )
        ax.set_box_aspect((48, 48, 26))
        ax.view_init(elev=-20, azim=-60)
    fig.suptitle("Go2 spherical foot: lower contact surface, radius 22 mm")
    fig.tight_layout()
    fig.savefig(output.with_name("go2-foot-meshes.png"), dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, component in zip(axes, ("force", "torque"), strict=True):
        for kind in ("uv", "icosphere"):
            rows = [r for r in results[:-1] if r["kind"] == kind]
            x = [1000 * r["maximum_edge_m"] for r in rows]
            ax.plot(x, [100 * r[f"{component}_relative_error"] for r in rows], "o-", label=f"{kind}: pooled L2")
            ax.plot(x, [100 * r[f"{component}_error_p95"] for r in rows], "s--", label=f"{kind}: 95th percentile")
        ax.set(xlabel="Maximum edge [mm]", ylabel=f"{component.title()} difference [%]")
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Go2 foot: geometry and quadrature self-convergence against 384 UV segments")
    fig.tight_layout()
    fig.savefig(output.with_name("go2-foot-convergence.png"), dpi=160)
    plt.close(fig)


def main(args):
    """Compare sphere tessellations without integrating a robot or changing material history."""
    radius = 0.022
    depths = (0.001, 0.003, 0.01, 0.025, 0.044)
    velocities = ((0, 0, 0), (0, 0, -0.1), (0.3, 0, -0.02), (1, 0, -0.1), (0, 0, 0.1))
    spins = ((0, 0, 0), (0, 8, 0))
    orientations = ((0, 0, 0), (0.3, 0.2, 0.1), (0.7, 0.4, 0.3))
    states = list(itertools.product(depths, velocities, spins, orientations))
    builder = newton.ModelBuilder()
    poses, twists = [], []
    for i, (depth, velocity, spin, orientation) in enumerate(states):
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label=f"foot_probe_{i}")
        poses.append([0, 0, radius - depth, *np.asarray(wp.quat_rpy(*map(float, orientation)))])
        twists.append([*velocity, *spin])
    model = builder.finalize(device=args.device)
    state = model.state()
    state.body_q.assign(np.array(poses, np.float32))
    state.body_qd.assign(np.array(twists, np.float32))
    results, meshes = [], {}
    materials = list(itertools.product((0.3, 0.5), (0.6, 1.0)))
    start = time.monotonic()
    candidates = [("uv", s) for s in (12, 24, 32, 48, 64, 96, 192)]
    candidates += [("icosphere", level) for level in (2, 3, 4, 5)] + [("uv", 384)]
    for kind, resolution in candidates:
        # Match the task's unit-sphere generation followed by collider scaling.
        if kind == "uv":
            unit = newton.Mesh.create_sphere(
                1.0, num_latitudes=resolution // 2, num_longitudes=resolution, compute_inertia=False
            )
            mesh = newton.Mesh(unit.vertices * np.float32(radius), unit.indices, compute_inertia=False)
        else:
            source = trimesh.creation.icosphere(subdivisions=resolution, radius=radius)
            mesh = newton.Mesh(source.vertices, source.faces.reshape(-1), compute_inertia=False)
        probe = SolverRFT(
            model,
            [(i, mesh) for i in range(len(states))],
            surface_height=0.0,
            config=SolverRFT.Config(bearing=0.87574, velocity_regularization=0.04),
        )
        count = probe.triangle_count // len(states)
        triangles = probe.triangle_vertices.numpy()[:count]
        if (kind, resolution) in (("uv", 12), ("icosphere", 3), ("icosphere", 4)):
            meshes[kind, resolution] = triangles
        area = (
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1) / 2
        )
        wrenches = []
        material = probe.material.numpy()
        for friction, roughness in materials:
            material[:, 1] = friction
            material[:, 2] = friction * roughness
            probe.material.assign(material)
            wrenches.append(probe.evaluate(state).numpy().copy())
        results.append(
            dict(
                kind=kind,
                resolution=resolution,
                label=f"{kind} {resolution}",
                raw_triangles_per_foot=len(mesh.indices) // 3,
                integrated_triangles_per_foot=count,
                lower_hemisphere_triangles=int((triangles.mean(axis=1)[:, 2] < 0).sum()),
                maximum_edge_m=float(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=-1).max()),
                sphere_area_relative_error=float(1 - area.sum() / (4 * np.pi * radius**2)),
                wrench=np.array(wrenches),
            )
        )
        del probe
    reference = results[-1]["wrench"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **{f"wrench_{i}": row["wrench"] for i, row in enumerate(results)},
        body_pose=np.array(poses, np.float32),
        body_twist=np.array(twists, np.float32),
    )
    depth_index = np.array([s[0] for s in states])
    for row in results:
        wrench = row.pop("wrench")
        delta = wrench - reference
        force = np.linalg.norm(delta[..., :3], axis=-1) / np.maximum(np.linalg.norm(reference[..., :3], axis=-1), 1)
        torque = np.linalg.norm(delta[..., 3:], axis=-1) / np.maximum(np.linalg.norm(reference[..., 3:], axis=-1), 0.01)
        row.update(
            force_relative_error=float(np.linalg.norm(delta[..., :3]) / np.linalg.norm(reference[..., :3])),
            torque_relative_error=float(np.linalg.norm(delta[..., 3:]) / np.linalg.norm(reference[..., 3:])),
            force_error_p95=float(np.percentile(force, 95)),
            torque_error_p95=float(np.percentile(torque, 95)),
            per_depth=[
                dict(
                    depth_m=d,
                    force_error_p95=float(np.percentile(force[:, depth_index == d], 95)),
                    torque_error_p95=float(np.percentile(torque[:, depth_index == d], 95)),
                )
                for d in depths
            ],
        )
    kernel = Path(newton.__file__).parent / "_src/solvers/rft/kernels.py"
    record = dict(
        results=results,
        radius_m=radius,
        depths_m=depths,
        velocities_m_s=velocities,
        angular_velocities_rad_s=spins,
        orientations_rpy_rad=orientations,
        materials=[dict(internal_friction=mu, roughness=rough) for mu, rough in materials],
        density_kg_m3=1600,
        bearing=0.87574,
        velocity_regularization_m_s=0.04,
        prescribed_states_per_material=len(states),
        reference_segments=384,
        wall_seconds=time.monotonic() - start,
        rft_kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="Prescribed centered sphere probes, not robot-policy validation or material validation. "
        "Refinement changes curved geometry and its quadrature together. Torques are about the sphere center. "
        "Percentiles use 1 N / 0.01 Nm denominator floors. Near-zero-area pole triangles remain in solver counts.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    plot_resolution(args.output, meshes, results)
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
