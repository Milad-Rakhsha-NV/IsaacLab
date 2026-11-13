# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure MPM convergence and RFT agreement on prescribed foot and plate trajectories.

The small frictionless-wall container keeps the bed level during preparation.
Domain-size sensitivity is measured separately. These are numerical comparisons,
not measurements of physical sand. Generated records belong under ``logs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM, SolverRFT

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@wp.kernel
def prescribe_probe(
    clock: wp.array[float],
    settle: float,
    depth: float,
    speed: float,
    half_height: float,
    drag: int,
    unload: int,
    q0: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    q1: wp.array[wp.transform],
    v1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    insertion_speed = wp.where(drag == 1, 0.1, speed)
    insertion_time = (depth + 0.01) / insertion_speed
    progress = wp.clamp(t - settle, 0.0, insertion_time)
    z = half_height + 0.01 - insertion_speed * progress
    vz = wp.where(t >= settle and t < settle + insertion_time, -insertion_speed, 0.0)
    x = float(0.0)
    vx = float(0.0)
    if drag == 1:
        start = settle + insertion_time + 0.1
        x = -0.02 + wp.clamp((t - start) * speed, 0.0, 0.04)
        vx = wp.where(t >= start and t < start + 0.04 / speed, speed, 0.0)
    elif unload == 1:
        start = settle + insertion_time + 0.1
        if t >= start:
            z = half_height - depth + speed * wp.clamp(t - start, 0.0, insertion_time)
            vz = wp.where(t < start + insertion_time, speed, 0.0)
    pose = wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())
    velocity = wp.spatial_vector(vx, 0.0, vz, 0.0, 0.0, 0.0)
    q0[0] = pose
    q1[0] = pose
    v0[0] = velocity
    v1[0] = velocity


@wp.kernel
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def accumulate_impulses(
    impulses: wp.array[wp.vec3],
    collider_ids: wp.array[int],
    collider_bodies: wp.array[int],
    total: wp.array[wp.vec3],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider >= 0 and collider < collider_bodies.shape[0]:
        slot = wp.where(collider_bodies[collider] >= 0, 0, 1)
        wp.atomic_add(total, slot, impulses[i])


@wp.kernel
def particle_statistics(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    masses: wp.array[float],
    probe_q: wp.array[wp.transform],
    radius: float,
    plate: int,
    bottom: float,
    out: wp.array[float],
):
    i = wp.tid()
    p = positions[i]
    v = velocities[i]
    m = masses[i]
    wp.atomic_add(out, 0, m * v[0])
    wp.atomic_add(out, 1, m * v[1])
    wp.atomic_add(out, 2, m * v[2])
    wp.atomic_add(out, 3, m * wp.dot(v, v))
    wp.atomic_max(out, 4, wp.max(0.0, bottom - p[2]))
    delta = p - wp.transform_get_translation(probe_q[0])
    penetration = radius - wp.length(delta)
    if plate == 1:
        b = wp.abs(delta) - wp.vec3(0.025, 0.025, 0.0025)
        penetration = -(wp.length(wp.max(b, wp.vec3(0.0))) + wp.min(wp.max(b[0], wp.max(b[1], b[2])), 0.0))
    wp.atomic_max(out, 5, wp.max(0.0, penetration))
    wp.atomic_add(out, 6, m * p[2])


def make_model(args, probe_mesh=None, probe_mass=None):
    """Build an exact-volume particle grid, a prescribed intruder, and a backing floor."""
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    if probe_mass is None:
        probe = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()), is_kinematic=True, label="probe"
        )
    else:
        if probe_mesh is not None:
            low, high = probe_mesh.vertices.min(0), probe_mesh.vertices.max(0)
        elif args.geometry == "sphere":
            low, high = np.full(3, -0.022), np.full(3, 0.022)
        else:
            low, high = -np.array([0.025, 0.025, 0.0025]), np.array([0.025, 0.025, 0.0025])
        center, size = (low + high) / 2, high - low
        inertia = (
            probe_mass
            / 12
            * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
        )
        pose = wp.transform(wp.vec3(-float(center[0]), -float(center[1]), -float(low[2]) + 0.01), wp.quat_identity())
        probe = builder.add_link(
            xform=pose,
            mass=probe_mass,
            com=wp.vec3(*center),
            inertia=wp.mat33(np.diag(inertia)),
            lock_inertia=True,
            label="probe",
        )
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=probe,
            parent_xform=pose,
            axis=(0.0, 0.0, 1.0),
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
            limit_lower=-1.0,
            limit_upper=1.0,
        )
        builder.add_articulation([joint])
    material = newton.ModelBuilder.ShapeConfig(mu=args.friction * args.roughness, density=0.0)
    if probe_mesh is not None:
        builder.add_shape_mesh(probe, mesh=probe_mesh, cfg=material)
    elif args.geometry == "sphere":
        builder.add_shape_sphere(probe, radius=0.022, cfg=material)
    else:
        builder.add_shape_box(probe, hx=0.025, hy=0.025, hz=0.0025, cfg=material)
    length, width = 0.4 * args.domain_scale, 0.3 * args.domain_scale
    floor = builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -args.bed_depth - 0.02), wp.quat_identity()),
        hx=length / 2 + 0.05,
        hy=width / 2 + 0.05,
        hz=0.02,
        cfg=material,
    )
    wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        position = [0.0, 0.0, -args.bed_depth / 2]
        position[axis] = sign * ((length if axis == 0 else width) / 2 + 0.01)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            hx=0.01 if axis == 0 else length / 2 + 0.02,
            hy=0.01 if axis == 1 else width / 2 + 0.02,
            hz=args.bed_depth / 2 + 0.04,
            cfg=wall_cfg,
        )
    extent = np.array([length, width, args.bed_depth])
    divisions = getattr(args, "particles_per_cell", 2)
    counts = np.ceil(extent / (args.voxel_size / divisions) - 1e-9).astype(int)
    spacing = extent / counts
    lower = np.array([-length / 2, -width / 2, -args.bed_depth])
    axes = [lower[i] + (np.arange(counts[i]) + 0.5) * spacing[i] for i in range(3)]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    count = len(positions)
    mass = float(np.prod(extent) * args.density / count)
    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros_like(positions).tolist(),
        mass=[mass] * count,
        radius=[float(np.cbrt(np.prod(spacing))) / 2] * count,
    )
    model = builder.finalize(device=args.device)
    # Native MPM uses (2r)^3, including for cells with unequal axis spacing.
    particle_volume = (2 * model.particle_radius.numpy()) ** 3
    np.testing.assert_allclose(particle_volume.sum(), np.prod(extent), rtol=1e-6)
    np.testing.assert_allclose(model.particle_mass.numpy() / particle_volume, args.density, rtol=1e-6)
    for key, value in dict(
        young_modulus=args.modulus,
        poisson_ratio=0.3,
        friction=args.friction,
        yield_pressure=1e15,
        tensile_yield_ratio=0.0,
        yield_stress=0.0,
        viscosity=0.0,
        hardening=0.0,
        dilatancy=0.0,
    ).items():
        getattr(model.mpm, key).fill_(value)
    return model, floor, count * mass


def rft_reference(args):
    """Use the training closure and quadrature on exactly the same prescribed pose."""
    builder = newton.ModelBuilder()
    builder.add_body(mass=1.0)
    model = builder.finalize(device=args.device)
    mesh = (
        newton.Mesh.create_sphere(
            0.022, num_latitudes=args.rft_segments // 2, num_longitudes=args.rft_segments, compute_inertia=False
        )
        if args.geometry == "sphere"
        else newton.Mesh.create_box(0.025, 0.025, 0.0025, compute_inertia=False)
    )
    solver = SolverRFT(
        model,
        [(0, mesh)],
        surface_height=0.0,
        surface_bottom=-args.bed_depth,
        config=SolverRFT.Config(
            density=args.density,
            internal_friction=args.friction,
            surface_friction=args.friction * args.roughness,
            velocity_regularization=0.04,
            bearing=0.87574,
        ),
    )
    return solver, model.state()


def run(args):
    """Run one isolated case and retain its force and conservation traces."""
    model, _, total_mass = make_model(args)
    config = SolverImplicitMPM.Config(
        voxel_size=args.voxel_size,
        grid_type="sparse",
        max_active_cell_count=1 << 18,
        max_leaf_node_count=1 << 15,
        max_lower_node_count=1 << 11,
        max_upper_node_count=1 << 9,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        collider_basis="pic27",
        strain_basis="P0",
        transfer_scheme=args.transfer_scheme,
        air_drag=1.0,
    )
    solver = SolverImplicitMPM(model, config=config)
    states = [model.state(), model.state()]
    clock = wp.zeros(1, device=args.device)
    impulse = wp.zeros(2, dtype=wp.vec3, device=args.device)
    statistics = wp.zeros(7, device=args.device)
    frame_dt = 0.005
    dt = frame_dt / args.substeps
    plate = int(args.geometry == "plate")

    def advance():
        impulse.zero_()
        for k in range(args.substeps):
            before, after = states[k % 2], states[1 - k % 2]
            wp.launch(
                prescribe_probe,
                dim=1,
                inputs=[
                    clock,
                    args.settle,
                    args.depth,
                    args.speed,
                    0.0025 if plate else 0.022,
                    int(args.mode == "drag"),
                    int(args.unload),
                    before.body_q,
                    before.body_qd,
                    after.body_q,
                    after.body_qd,
                ],
                device=args.device,
            )
            solver.step(before, after, None, None, dt)
            if args.project:
                solver.project_outside(after, after, dt)
            values, _, ids = solver.collect_collider_impulses(after)
            wp.launch(accumulate_impulses, dim=ids.shape[0], inputs=[values, ids, solver.collider_body_index, impulse])
            wp.launch(advance_clock, dim=1, inputs=[clock, dt])
        statistics.zero_()
        wp.launch(
            particle_statistics,
            dim=model.particle_count,
            inputs=[
                states[0].particle_q,
                states[0].particle_qd,
                model.particle_mass,
                states[0].body_q,
                0.022,
                plate,
                -args.bed_depth,
                statistics,
            ],
        )

    # Compile and create persistent sparse-grid buffers before capture.
    advance()
    with wp.ScopedCapture(device=args.device) as capture:
        advance()
    reference, rft_state = rft_reference(args)
    insertion = (args.depth + 0.01) / (0.1 if args.mode == "drag" else args.speed)
    duration = args.settle + insertion + (0.1 + 0.04 / args.speed if args.mode == "drag" else 0.1)
    if args.unload:
        duration += insertion
    rows = []
    previous_p = np.zeros(3)
    start = time.monotonic()
    for frame in range(math.ceil(duration / frame_dt)):
        wp.capture_launch(capture.graph)
        t = float(clock.numpy()[0])
        stats = statistics.numpy()
        impulses = impulse.numpy()
        p = stats[:3]
        balance = p - previous_p + impulses.sum(axis=0) - np.array([0.0, 0.0, -total_mass * 9.81 * frame_dt])
        previous_p = p.copy()
        rft_state.body_q.assign(states[0].body_q)
        rft_state.body_qd.assign(states[0].body_qd)
        rft = reference.evaluate(rft_state).numpy()[0, :3]
        rows.append(
            np.r_[
                t,
                states[0].body_q.numpy()[0, :3],
                impulses[0] / frame_dt,
                rft,
                impulses[1] / frame_dt,
                np.sqrt(stats[3] / total_mass),
                stats[4:6],
                balance,
            ]
        )
        solver.check_sparse_grid_rebuild_status()
        if not np.isfinite(rows[-1]).all():
            raise RuntimeError("Nonfinite probe state or force")
    rows = np.asarray(rows)
    insertion_end = args.settle + insertion
    if args.mode == "drag":
        mask = (rows[:, 0] > insertion_end + 0.1 + 0.01 / args.speed) & (rows[:, 0] < duration - 0.005)
    else:
        penetration = (0.0025 if plate else 0.022) - rows[:, 3]
        mask = (penetration > 0.005) & (rows[:, 0] < insertion_end)
    settled = rows[(rows[:, 0] > args.settle - 0.1) & (rows[:, 0] < args.settle)]
    summary = dict(
        settings=vars(args) | {"output": str(args.output)},
        simulation_source_sha256=SOURCE_SHA256,
        particle_count=model.particle_count,
        total_mass_kg=total_mass,
        wall_seconds=time.monotonic() - start,
        mean_force_N=rows[mask, 4:7].mean(axis=0).tolist(),
        mean_rft_force_N=rows[mask, 7:10].mean(axis=0).tolist(),
        peak_force_N=np.linalg.norm(rows[mask, 4:7], axis=1).max().item(),
        settled_rms_speed_m_s=float(settled[:, 13].mean()),
        maximum_floor_penetration_m=float(rows[:, 14].max()),
        maximum_probe_penetration_m=float(rows[:, 15].max()),
        momentum_balance_mean_relative=float(
            np.linalg.norm(rows[1:, 16:19], axis=1).mean() / (total_mass * 9.81 * frame_dt)
        ),
        container="0.4 x 0.3 m scaled horizontally, frictionless sidewalls, rigid bottom",
        force_sampling="sum of all collider impulses over each 5 ms frame divided by frame duration",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "trace.npz", trace=rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("vertical", "drag"), default="vertical")
    parser.add_argument("--geometry", choices=("sphere", "plate"), default="sphere")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--depth", type=float, default=0.04)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--bed_depth", type=float, default=0.118)
    parser.add_argument("--density", type=float, default=1600.0)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--modulus", type=float, default=1e15)
    parser.add_argument("--domain_scale", type=float, default=1.0)
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--transfer_scheme", choices=("pic", "apic"), default="pic")
    parser.add_argument("--rft_segments", type=int, default=12)
    parser.add_argument("--unload", action="store_true")
    run(parser.parse_args())
