# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare G1 collision geometry in MPM and RFT along identical prescribed motions.

Forces and ankle-origin torques are integrated over every substep. There is no
policy or rigid-body integrator. The backing floor and container react against
particles only; no direct floor reaction is added to the measured foot wrench.
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
from investigate_go2_granular import make_model
from newton.solvers import SolverImplicitMPM, SolverRFT
from scipy.spatial.transform import Rotation


@wp.kernel
def prescribe(
    tick: wp.array[int],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    q0: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    q1: wp.array[wp.transform],
    v1: wp.array[wp.spatial_vector],
):
    i = wp.min(tick[0], poses.shape[0] - 2)
    q0[0] = poses[i]
    q1[0] = poses[i + 1]
    v0[0] = velocities[i]
    v1[0] = velocities[i + 1]


@wp.kernel
def advance_tick(tick: wp.array[int]):
    tick[0] += 1


@wp.kernel
def wrench_impulse(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    ids: wp.array[int],
    bodies: wp.array[int],
    pose: wp.array[wp.transform],
    out: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    collider = ids[i]
    if collider >= 0 and collider < bodies.shape[0]:
        slot = wp.where(bodies[collider] == 0, 0, 1)
        r = positions[i] - wp.transform_get_translation(pose[0])
        wp.atomic_add(out, slot, wp.spatial_vector(impulses[i], wp.cross(r, impulses[i])))


@wp.kernel
def integrate_rft(force: wp.array[wp.spatial_vector], dt: float, total: wp.array[wp.spatial_vector]):
    total[0] += dt * force[0]


@wp.kernel
def statistics(
    q: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    mass: wp.array[float],
    bottom: float,
    out: wp.array[float],
):
    i = wp.tid()
    p, velocity = q[i], v[i]
    for axis in range(3):
        wp.atomic_add(out, axis, mass[i] * velocity[axis])
        if not wp.isfinite(p[axis]) or not wp.isfinite(velocity[axis]):
            wp.atomic_add(out, 5, 1.0)
    wp.atomic_add(out, 3, mass[i] * wp.dot(velocity, velocity))
    wp.atomic_max(out, 4, wp.max(0.0, bottom - p[2]))


def trajectory(args, bounds):
    """Return exact shared poses, COM-at-ankle twists and named phase intervals."""
    phases = [("preparation", args.settle), ("insertion", (args.depth + 0.01) / args.speed), ("hold", 0.35)]
    if args.motion == "cycle":
        phases += [("drag", 0.04 / args.speed), ("drag_hold", 0.25), ("pitch", 0.4), ("roll", 0.4)]
    phases += [("withdrawal", (args.depth + 0.01) / args.speed), ("released", 0.15)]
    ends = np.cumsum([seconds for _, seconds in phases])
    dt = 0.005 / args.substeps
    t = np.arange(math.ceil(ends[-1] / dt) + 4) * dt
    xyz = np.zeros((len(t), 3))
    minimum_z = bounds[0][2]
    initial_x = -(bounds[0][0] + bounds[1][0]) / 2 - 0.02
    xyz[:, 0] = initial_x
    xyz[:, 2] = -minimum_z + 0.01
    angles = np.zeros((len(t), 3))
    phase_id = np.zeros(len(t), dtype=int)
    start = 0.0
    for i, (name, seconds) in enumerate(phases):
        local = np.clip((t - start) / seconds, 0, 1)
        mask = t >= start
        phase_id[mask] = i
        if name == "insertion":
            xyz[mask, 2] = -minimum_z + 0.01 - (args.depth + 0.01) * local[mask]
        elif name == "drag":
            xyz[mask, 0] = initial_x + 0.04 * local[mask]
        elif name in ("pitch", "roll"):
            axis = 1 if name == "pitch" else 0
            angles[:, axis] = args.angle * np.sin(np.pi * local) ** 2
        elif name == "withdrawal":
            xyz[mask, 2] = -minimum_z - args.depth + (args.depth + 0.01) * local[mask]
        start += seconds
    rotations = Rotation.from_euler("xyz", angles)
    q = np.c_[xyz, rotations.as_quat()].astype(np.float32)
    velocity = np.gradient(xyz, dt, axis=0)
    angular = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    twist = np.c_[velocity, np.vstack([angular, angular[-1]])].astype(np.float32)
    return (
        q,
        twist,
        phase_id,
        [
            dict(name=name, start=float(end - seconds), end=float(end))
            for (name, seconds), end in zip(phases, ends, strict=True)
        ],
    )


def run(args):
    """Save interval wrenches, particle balance and the full prescribed trajectory."""
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(args.geometry_file.read_text())
    entry = next(row for row in geometry["surfaces"] if row["body_label"].endswith("/left_ankle_roll_link"))
    mesh = newton.Mesh(
        np.array(entry["vertices"], np.float32), np.array(entry["indices"], np.int32), compute_inertia=False
    )
    model, _, total_mass = make_model(args, probe_mesh=mesh)
    config = SolverImplicitMPM.Config(
        voxel_size=args.voxel_size,
        grid_type="sparse",
        max_active_cell_count=1 << 19,
        max_leaf_node_count=1 << 15,
        max_lower_node_count=1 << 11,
        max_upper_node_count=1 << 9,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        collider_basis=args.collider_basis,
        strain_basis="P0",
        transfer_scheme=args.transfer_scheme,
        air_drag=1.0,
    )
    solver = SolverImplicitMPM(model, config=config)
    states = [model.state(), model.state()]
    rb = newton.ModelBuilder()
    rb.add_body(mass=1.0)
    rmodel = rb.finalize(device=args.device)
    rft = SolverRFT(
        rmodel,
        [(0, mesh)],
        surface_height=0.0,
        surface_bottom=-args.bed_depth,
        max_edge_length=args.rft_edge,
        config=SolverRFT.Config(
            density=args.density,
            internal_friction=args.friction,
            surface_friction=args.friction * args.roughness,
            bearing=args.bearing,
            velocity_regularization=args.regularization,
        ),
    )
    rstate = rmodel.state()
    q, v, phase, intervals = trajectory(args, entry["bounds_m"])
    poses = wp.array(q, dtype=wp.transform, device=args.device)
    velocities = wp.array(v, dtype=wp.spatial_vector, device=args.device)
    tick = wp.zeros(1, dtype=int, device=args.device)
    impulse = wp.zeros(2, dtype=wp.spatial_vector, device=args.device)
    rft_impulse = wp.zeros(1, dtype=wp.spatial_vector, device=args.device)
    stats = wp.zeros(6, device=args.device)
    frame_dt = 0.005
    dt = frame_dt / args.substeps

    def advance():
        impulse.zero_()
        rft_impulse.zero_()
        for k in range(args.substeps):
            before, after = states[k % 2], states[1 - k % 2]
            wp.launch(
                prescribe,
                dim=1,
                inputs=[tick, poses, velocities, before.body_q, before.body_qd, after.body_q, after.body_qd],
            )
            rstate.body_q, rstate.body_qd = before.body_q, before.body_qd
            wp.launch(integrate_rft, dim=1, inputs=[rft.evaluate(rstate), dt, rft_impulse])
            solver.step(before, after, None, None, dt)
            values, positions, ids = solver.collect_collider_impulses(after)
            wp.launch(
                wrench_impulse,
                dim=ids.shape[0],
                inputs=[values, positions, ids, solver.collider_body_index, before.body_q, impulse],
            )
            wp.launch(advance_tick, dim=1, inputs=[tick])
        stats.zero_()
        wp.launch(
            statistics,
            dim=model.particle_count,
            inputs=[states[0].particle_q, states[0].particle_qd, model.particle_mass, -args.bed_depth, stats],
        )

    advance()
    with wp.ScopedCapture(device=args.device) as capture:
        advance()
    duration = intervals[-1]["end"]
    rows, mpm_wrench, rft_wrench, balances, kinematics, labels = [], [], [], [], [], []
    previous = stats.numpy()[:3].copy()
    start = time.monotonic()
    for frame in range(math.ceil(duration / frame_dt) - 2):
        wp.capture_launch(capture.graph)
        i = int(tick.numpy()[0])
        s, impulses = stats.numpy(), impulse.numpy()
        assert s[5] == 0 and np.isfinite(s).all(), "Nonfinite material state"
        balance = s[:3] - previous + impulses[:, :3].sum(0) - np.array([0, 0, -total_mass * 9.81 * frame_dt])
        previous = s[:3].copy()
        rows.append([i * dt, np.sqrt(s[3] / total_mass), s[4]])
        mpm_wrench.append(impulses[0] / frame_dt)
        rft_wrench.append(rft_impulse.numpy()[0] / frame_dt)
        balances.append(balance)
        mid = i - args.substeps // 2
        kinematics.append(np.r_[q[mid], v[mid]])
        labels.append(phase[mid])
        solver.check_sparse_grid_rebuild_status()
        if frame % 50 == 0:
            temporary = args.output / "progress.tmp"
            temporary.write_text(json.dumps(dict(simulated_seconds=i * dt, total_seconds=duration)))
            temporary.replace(args.output / "progress.json")
    rows, mpm_wrench, rft_wrench, balances = map(np.asarray, (rows, mpm_wrench, rft_wrench, balances))
    assert np.isfinite(mpm_wrench).all() and np.isfinite(rft_wrench).all()
    settled = (rows[:, 0] >= args.settle - 0.1) & (rows[:, 0] < args.settle)
    summary = dict(
        settings=vars(args) | dict(output=str(args.output), geometry_file=str(args.geometry_file)),
        phases=intervals,
        particle_count=model.particle_count,
        particles_finite=True,
        wall_seconds=time.monotonic() - start,
        max_floor_penetration_m=float(rows[:, 2].max()),
        settled_rms_speed_m_s=float(rows[settled, 1].mean()),
        momentum_balance_relative=float(np.linalg.norm(balances, axis=1).mean() / (total_mass * 9.81 * frame_dt)),
        geometry_sha256=hashlib.sha256(args.geometry_file.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wrench_frame="world force XYZ, world torque XYZ about prescribed ankle origin",
        force_sampling="all collider impulses over every substep of each 5 ms frame",
        warmup_seconds=0.01,
    )
    np.savez_compressed(
        args.output / "trace.npz",
        statistics=rows,
        mpm_wrench=mpm_wrench,
        rft_wrench=rft_wrench,
        balance=balances,
        kinematics=kinematics,
        phase=labels,
        poses=q,
        velocities=v,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry_file", type=Path, required=True)
    parser.add_argument("--geometry", default="g1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--depth", type=float, default=0.04)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--bed_depth", type=float, default=0.118)
    parser.add_argument("--density", type=float, default=1600.0)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--modulus", type=float, default=1e15)
    parser.add_argument("--domain_scale", type=float, default=1.0)
    parser.add_argument("--transfer_scheme", choices=("pic", "apic"), default="apic")
    parser.add_argument("--collider_basis", default="pic27")
    parser.add_argument("--motion", choices=("cycle", "vertical"), default="cycle")
    parser.add_argument("--angle", type=float, default=0.15)
    parser.add_argument("--rft_edge", type=float, default=0.02)
    parser.add_argument("--bearing", type=float, default=0.87574)
    parser.add_argument("--regularization", type=float, default=0.04)
    run(parser.parse_args())
