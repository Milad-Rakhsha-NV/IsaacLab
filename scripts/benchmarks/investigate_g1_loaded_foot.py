# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare load-controlled Go2/G1 foot settling and proxy-coupling sensitivity.

A vertical prismatic joint fixes lateral translation and orientation. The load
is independent of the chosen inertial mass. Static support is measured under
load, separately from a prescribed-velocity fixture's reaction wrench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from investigate_go2_granular import accumulate_impulses, make_model
from newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverRFT
from newton.solvers.experimental.coupled import SolverCoupledProxy

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@wp.kernel
def apply_load(
    body_f: wp.array[wp.spatial_vector],
    clock: wp.array[float],
    mass: float,
    load: float,
    load_after: float,
    switch_time: float,
    lateral: float,
    lateral_start: float,
    dt: float,
    external: wp.array[float],
    applied_impulse: wp.array[wp.float64],
):
    current_load = wp.where(clock[0] >= switch_time, load_after, load)
    horizontal = wp.where(clock[0] >= lateral_start, lateral, 0.0)
    extra = mass * 9.81 - current_load
    external[0] = extra
    external[1] = horizontal
    body_f[0] += wp.spatial_vector(horizontal, 0.0, extra, 0.0, 0.0, 0.0)
    applied_impulse[0] += wp.float64(dt) * wp.float64(current_load)
    applied_impulse[1] += wp.float64(dt) * wp.float64(horizontal)
    clock[0] += dt


@wp.kernel
def record_delivered_impulse(
    forces: wp.array[wp.spatial_vector], external: wp.array[float], dt: float, impulse: wp.array[wp.float64]
):
    # Coupling iterations are alternative solves of one interval. Retain the
    # final source solve, rather than summing impulses from all trials.
    impulse[0] = wp.float64(dt) * wp.float64(forces[0][2] - external[0])
    impulse[1] = wp.float64(dt) * wp.float64(forces[0][0] - external[1])


@wp.kernel
def accumulate_delivered_impulse(substep: wp.array[wp.float64], frame: wp.array[wp.float64]):
    i = wp.tid()
    frame[i] += substep[i]


class _LoadedFootMuJoCo(SolverMuJoCo):
    """Measure the granular force actually received by the rigid solver."""

    def __init__(self, model, *, delivered_impulse, external, **kwargs):
        super().__init__(model, **kwargs)
        self.delivered_impulse = delivered_impulse
        self.external = external

    def step(self, state_in, state_out, control, contacts, dt):
        wp.launch(
            record_delivered_impulse,
            dim=1,
            inputs=[state_in.body_f, self.external, dt, self.delivered_impulse],
            device=self.model.device,
        )
        super().step(state_in, state_out, control, contacts, dt)


@wp.kernel
def prescribe_preload(
    clock: wp.array[float],
    depth: float,
    speed: float,
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    travel = depth + 0.01
    displacement = wp.min(speed * clock[0], travel)
    velocity = wp.where(speed * clock[0] < travel, -speed, 0.0)
    body_q[0] = wp.transform(wp.vec3(0.0, 0.0, 0.032 - displacement), wp.quat_identity())
    body_qd[0] = wp.spatial_vector(0.0, 0.0, velocity, 0.0, 0.0, 0.0)
    joint_q[0] = -displacement
    joint_qd[0] = velocity
    clock[0] += dt


def run(args):
    """Prepare the bed, release the foot and retain support and settling histories."""
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = None
    if args.geometry == "g1":
        if args.geometry_file is None:
            raise ValueError("G1 probes require --geometry_file")
        geometry = json.loads(args.geometry_file.read_text())
        foot = next(row for row in geometry["surfaces"] if row["body_label"].endswith("/left_ankle_roll_link"))
        mesh = newton.Mesh(
            np.array(foot["vertices"], np.float32), np.array(foot["indices"], np.int32), compute_inertia=False
        )
    model, _, _ = make_model(args, probe_mesh=mesh, probe_mass=args.mass, probe_lateral=args.free_lateral)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    delivered_impulse = wp.zeros(2, dtype=wp.float64, device=model.device)
    delivered_substep = wp.zeros_like(delivered_impulse)
    external = wp.zeros(2, device=model.device)
    load_clock = wp.zeros(1, device=model.device)
    applied_impulse = wp.zeros(2, dtype=wp.float64, device=model.device)
    dt = 0.005 / args.substeps
    mpm = None
    preload_rows = []
    minimum_local_z = (
        -0.022
        if mesh is None and args.geometry == "sphere"
        else (-0.0025 if mesh is None else float(mesh.vertices[:, 2].min()))
    )
    if args.field == "mpm":
        config = SolverImplicitMPM.Config(
            voxel_size=args.voxel_size,
            grid_type="sparse",
            max_active_cell_count=1 << 19,
            max_leaf_node_count=1 << 15,
            max_lower_node_count=1 << 11,
            max_upper_node_count=1 << 9,
            max_iterations=args.iterations,
            tolerance=1e-4,
            collider_basis="pic27",
            strain_basis="P0",
            transfer_scheme="apic",
        )
        solver = SolverCoupledProxy(
            model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="rigid",
                    solver=lambda m: _LoadedFootMuJoCo(
                        m,
                        delivered_impulse=delivered_substep,
                        external=external,
                        use_mujoco_contacts=True,
                        integrator="implicitfast",
                        njmax=64,
                        nconmax=32,
                    ),
                    bodies=[0],
                    joints=[0],
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda m: SolverImplicitMPM(m, config=config),
                    particles=list(range(model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="rigid",
                        destination="mpm",
                        bodies=[0],
                        mode="lagged",
                        proxy_relaxation=args.relaxation,
                        collision_pipeline=lambda m: None,
                    )
                ],
                iterations=args.passes,
            ),
        )
        mpm = solver.solver("mpm")
        if args.preload_depth is not None:
            if args.geometry != "sphere":
                raise ValueError("The prescribed preload currently uses the Go2 sphere")
            mpm.setup_collider(
                body_mass=wp.zeros_like(mpm.model.body_mass),
                body_inv_inertia=wp.zeros_like(mpm.model.body_inv_inertia),
                body_q=state.body_q,
            )

        def prepare():
            for _ in range(args.substeps):
                mpm.step(state, state, None, None, dt)

        initial_pose = state.body_q.numpy().copy()
        prepare()
        with wp.ScopedCapture(device=args.device) as cap:
            prepare()
        # CUDA capture records this frame without executing it. The eager
        # warmup above advances one frame, not two.
        for _ in range(max(0, round(args.settle / 0.005) - 1)):
            wp.capture_launch(cap.graph)
        mpm.check_sparse_grid_rebuild_status()
        np.testing.assert_array_equal(state.body_q.numpy(), initial_pose)
        if args.preload_depth is not None:
            clock = wp.zeros(1, device=model.device)
            preload_impulse = wp.zeros(2, dtype=wp.vec3, device=model.device)

            def preload():
                preload_impulse.zero_()
                for _ in range(args.substeps):
                    wp.launch(
                        prescribe_preload,
                        dim=1,
                        inputs=[
                            clock,
                            args.preload_depth,
                            args.preload_speed,
                            dt,
                            state.body_q,
                            state.body_qd,
                            state.joint_q,
                            state.joint_qd,
                        ],
                        device=model.device,
                    )
                    mpm.step(state, state, None, None, dt)
                    impulses, _, ids = mpm.collect_collider_impulses(state)
                    wp.launch(
                        accumulate_impulses,
                        dim=ids.shape[0],
                        inputs=[impulses, ids, mpm.collider_body_index, preload_impulse],
                        device=model.device,
                    )

            preload()
            with wp.ScopedCapture(device=args.device) as cap_preload:
                preload()
            total = (args.preload_depth + 0.01) / args.preload_speed + args.preload_hold
            for frame in range(max(0, round(total / 0.005) - 1)):
                wp.capture_launch(cap_preload.graph)
                mpm.check_sparse_grid_rebuild_status()
                if frame % 10 == 0:
                    preload_rows.append(
                        [
                            float(clock.numpy()[0]),
                            float(state.body_q.numpy()[0, 2]),
                            float(preload_impulse.numpy()[0, 2]) / 0.005,
                        ]
                    )
            mpm.setup_collider(
                body_mass=mpm.model.body_mass, body_inv_inertia=mpm.model.body_inv_inertia, body_q=state.body_q
            )
    else:
        if args.preload_depth is not None:
            if args.geometry != "sphere":
                raise ValueError("The prescribed preload currently uses the Go2 sphere")
            displacement = state.joint_q.numpy()
            displacement[0] = -(args.preload_depth + 0.01)
            state.joint_q.assign(displacement)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        rigid = _LoadedFootMuJoCo(
            model,
            delivered_impulse=delivered_substep,
            external=external,
            use_mujoco_contacts=True,
            integrator="implicitfast",
            njmax=64,
            nconmax=32,
        )
        quadrature = {}
        surfaces = [(0, mesh)]
        if mesh is None:
            from isaaclab_newton.physics.rft_solver import collision_surfaces

            rules = [dict(shape_pattern=".*", sphere_subdivisions=2)] if args.geometry == "sphere" else []
            surfaces, inventory = collision_surfaces(model, "probe", 12, resolutions=rules)
            quadrature = dict(surface_shape_ids=[r["shape"] for r in inventory])
            if args.geometry == "sphere":
                quadrature["surface_quadrature_orders"] = [6]
        solver = SolverRFT(
            model,
            surfaces,
            rigid_solver=rigid,
            surface_height=0.0,
            surface_bottom=-args.bed_depth,
            max_edge_length=0.02 if mesh is not None else None,
            **quadrature,
            config=SolverRFT.Config(
                density=args.density,
                internal_friction=args.friction,
                surface_friction=args.friction * args.roughness,
                strength=args.density
                * 9.81
                * (894 * args.friction**3 - 386 * args.friction**2 + 89 * args.friction)
                * args.strength_multiplier,
                velocity_regularization=args.velocity_regularization,
                bearing=args.bearing,
                static_contact_iterations=args.static_contact_iterations,
                static_contact_speed=args.static_contact_speed,
                static_contact_strength=args.static_contact_strength,
            ),
        )
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()

    def advance():
        delivered_impulse.zero_()
        applied_impulse.zero_()
        for _ in range(args.substeps):
            state.clear_forces()
            wp.launch(
                apply_load,
                dim=1,
                inputs=[
                    state.body_f,
                    load_clock,
                    args.mass,
                    args.load,
                    args.load if args.load_after is None else args.load_after,
                    args.load_switch_time,
                    args.lateral_force,
                    args.lateral_start_time,
                    dt,
                    external,
                    applied_impulse,
                ],
                device=model.device,
            )
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, dt)
            wp.launch(
                accumulate_delivered_impulse, dim=2, inputs=[delivered_substep, delivered_impulse], device=model.device
            )

    previous = float(state.body_qd.numpy()[0, 2])
    initial_pose = state.body_q.numpy()[0].copy()
    advance()
    first_pose, first_velocity = state.body_q.numpy()[0], state.body_qd.numpy()[0]
    rows = [
        [
            0.005,
            float(first_pose[2]),
            float(first_velocity[2]),
            args.mass * (float(first_velocity[2]) - previous) / 0.005 + float(applied_impulse.numpy()[0]) / 0.005,
            -(float(first_pose[2]) + minimum_local_z),
            float(delivered_impulse.numpy()[0]) / 0.005,
            float(first_pose[0]),
            float(first_velocity[0]),
            float(applied_impulse.numpy()[1]) / 0.005,
            float(applied_impulse.numpy()[0]) / 0.005,
            float(delivered_impulse.numpy()[1]) / 0.005,
        ]
    ]
    with wp.ScopedCapture(device=args.device) as cap:
        advance()
    previous = float(state.body_qd.numpy()[0, 2])
    start = time.monotonic()
    for frame in range(1, round(args.seconds / 0.005)):
        wp.capture_launch(cap.graph)
        q, v = state.body_q.numpy()[0], state.body_qd.numpy()[0]
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError("Nonfinite loaded-foot state")
        support = args.mass * (v[2] - previous) / 0.005 + float(applied_impulse.numpy()[0]) / 0.005
        previous = float(v[2])
        depth = -(float(q[2]) + minimum_local_z)
        rows.append(
            [
                (frame + 1) * 0.005,
                float(q[2]),
                float(v[2]),
                support,
                depth,
                float(delivered_impulse.numpy()[0]) / 0.005,
                float(q[0]),
                float(v[0]),
                float(applied_impulse.numpy()[1]) / 0.005,
                float(applied_impulse.numpy()[0]) / 0.005,
                float(delivered_impulse.numpy()[1]) / 0.005,
            ]
        )
        if mpm is not None:
            mpm.check_sparse_grid_rebuild_status()
        if frame % 50 == 0:
            (args.output / "progress.json").write_text(
                json.dumps(
                    dict(
                        simulated_seconds=(frame + 1) * 0.005,
                        total_seconds=args.seconds,
                        sole_depth_m=depth,
                        vertical_speed_m_s=float(v[2]),
                    )
                )
            )
            np.savez_compressed(
                args.output / "partial-trace.npz", trace=np.asarray(rows), preload=np.asarray(preload_rows)
            )
    rows = np.array(rows)
    particles_finite = bool(
        np.isfinite(state.particle_q.numpy()).all() and np.isfinite(state.particle_qd.numpy()).all()
    )
    if not particles_finite:
        raise RuntimeError("Nonfinite loaded-foot material state")
    after = rows[:, 0] >= max(0.5, args.seconds - 0.5)
    summary = dict(
        settings=vars(args) | dict(output=str(args.output), geometry_file=str(args.geometry_file)),
        wall_seconds=time.monotonic() - start,
        particles_finite=particles_finite,
        warmup_seconds=0.005,
        warmup_included_in_trace=True,
        source_sha256=SOURCE_SHA256,
        initial_release_pose=initial_pose.tolist(),
        trace_columns=[
            "time_s",
            "center_z_m",
            "vertical_speed_m_s",
            "total_support_N",
            "sole_depth_m",
            "delivered_granular_support_N",
            "center_x_m",
            "horizontal_speed_m_s",
            "applied_horizontal_force_N",
            "applied_downward_load_N",
            "delivered_horizontal_granular_force_N",
        ],
        preload_columns=["time_s", "center_z_m", "mpm_support_N"],
        final_sole_depth_m=float(rows[-1, 4]),
        mean_final_speed_m_s=float(abs(rows[after, 2]).mean()),
        mean_final_total_support_N=float(rows[after, 3].mean()),
        mean_final_delivered_granular_support_N=float(rows[after, 5].mean()),
        free_support_balance_error_p95_N=float(
            np.quantile(
                abs(rows[rows[:, 4] < args.bed_depth - 0.002, 3] - rows[rows[:, 4] < args.bed_depth - 0.002, 5]), 0.95
            )
        )
        if (rows[:, 4] < args.bed_depth - 0.002).any()
        else None,
        backing_floor_proximity=bool(rows[:, 4].max() >= args.bed_depth - 0.002),
        interpretation=(
            "Total support inferred from momentum; includes rigid bottom contact if reached. "
            f"{'Vertical and lateral DOFs' if args.free_lateral else 'One vertical DOF'}; "
            "load is independent of inertial mass."
        ),
    )
    np.savez_compressed(args.output / "trace.npz", trace=rows, preload=np.asarray(preload_rows))
    (args.output / "progress.json").write_text(
        json.dumps(
            dict(
                simulated_seconds=float(rows[-1, 0]),
                total_seconds=args.seconds,
                sole_depth_m=float(rows[-1, 4]),
                vertical_speed_m_s=float(rows[-1, 2]),
                complete=True,
            )
        )
        + "\n"
    )
    (args.output / "partial-trace.npz").unlink(missing_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry_file", type=Path)
    p.add_argument("--field", choices=("rft", "mpm"), default="mpm")
    p.add_argument("--mass", type=float, default=16.0)
    p.add_argument("--load", type=float, default=158.13)
    p.add_argument("--load_after", type=float)
    p.add_argument("--load_switch_time", type=float, default=1e30)
    p.add_argument("--free_lateral", action="store_true")
    p.add_argument("--lateral_force", type=float, default=0.0)
    p.add_argument("--lateral_start_time", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--relaxation", type=float, default=1.0)
    p.add_argument("--bearing", type=float, default=0.87574)
    p.add_argument("--static_contact_iterations", type=int, default=0)
    p.add_argument("--static_contact_speed", type=float, default=0.02)
    p.add_argument("--static_contact_strength", type=float, default=1.0)
    p.add_argument("--velocity_regularization", type=float, default=0.04)
    p.add_argument("--strength_multiplier", type=float, default=1.0)
    p.add_argument("--voxel_size", type=float, default=0.01)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--particles_per_cell", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--geometry", choices=("g1", "sphere", "plate"), default="g1")
    p.add_argument("--bed_depth", type=float, default=0.118)
    p.add_argument("--density", type=float, default=1600.0)
    p.add_argument("--friction", type=float, default=0.4)
    p.add_argument("--roughness", type=float, default=1.0)
    p.add_argument("--modulus", type=float, default=1e15)
    p.add_argument("--domain_scale", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=0.5)
    p.add_argument("--preload_depth", type=float)
    p.add_argument("--preload_speed", type=float, default=0.02)
    p.add_argument("--preload_hold", type=float, default=0.1)
    run(p.parse_args())
