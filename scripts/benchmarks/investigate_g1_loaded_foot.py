# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare load-controlled G1 foot settling and proxy-coupling sensitivity.

A vertical prismatic joint fixes lateral translation and orientation. The load
is independent of the chosen inertial mass. Static support is measured under
load, separately from a prescribed-velocity fixture's reaction wrench.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from investigate_go2_granular import make_model
from newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverRFT
from newton.solvers.experimental.coupled import SolverCoupledProxy


@wp.kernel
def apply_load(body_f: wp.array[wp.spatial_vector], extra: float):
    body_f[0] += wp.spatial_vector(0.0, 0.0, extra, 0.0, 0.0, 0.0)


def run(args):
    """Prepare the bed, release the foot and retain support and settling histories."""
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(args.geometry_file.read_text())
    foot = next(row for row in geometry["surfaces"] if row["body_label"].endswith("/left_ankle_roll_link"))
    mesh = newton.Mesh(
        np.array(foot["vertices"], np.float32), np.array(foot["indices"], np.int32), compute_inertia=False
    )
    model, _, _ = make_model(args, probe_mesh=mesh, probe_mass=args.mass)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    dt = 0.005 / args.substeps
    mpm = None
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
                    solver=lambda m: SolverMuJoCo(
                        m, use_mujoco_contacts=True, integrator="implicitfast", njmax=64, nconmax=32
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

        def prepare():
            for _ in range(args.substeps):
                mpm.step(state, state, None, None, dt)

        prepare()
        with wp.ScopedCapture(device=args.device) as cap:
            prepare()
        for _ in range(max(0, round(args.settle / 0.005) - 2)):
            wp.capture_launch(cap.graph)
        mpm.check_sparse_grid_rebuild_status()
    else:
        rigid = SolverMuJoCo(model, use_mujoco_contacts=True, integrator="implicitfast", njmax=64, nconmax=32)
        solver = SolverRFT(
            model,
            [(0, mesh)],
            rigid_solver=rigid,
            surface_height=0.0,
            surface_bottom=-args.bed_depth,
            max_edge_length=0.02,
            config=SolverRFT.Config(
                density=args.density,
                internal_friction=args.friction,
                surface_friction=args.friction * args.roughness,
                strength=args.density
                * 9.81
                * (894 * args.friction**3 - 386 * args.friction**2 + 89 * args.friction)
                * args.strength_multiplier,
                velocity_regularization=0.04,
                bearing=args.bearing,
            ),
        )
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()

    def advance():
        for _ in range(args.substeps):
            state.clear_forces()
            wp.launch(apply_load, dim=1, inputs=[state.body_f, args.mass * 9.81 - args.load])
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, dt)

    advance()
    with wp.ScopedCapture(device=args.device) as cap:
        advance()
    rows = []
    previous = float(state.body_qd.numpy()[0, 2])
    start = time.monotonic()
    for frame in range(round(args.seconds / 0.005) - 2):
        wp.capture_launch(cap.graph)
        q, v = state.body_q.numpy()[0], state.body_qd.numpy()[0]
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError("Nonfinite loaded-foot state")
        support = args.mass * (v[2] - previous) / 0.005 + args.load
        previous = float(v[2])
        depth = -(float(q[2]) + float(mesh.vertices[:, 2].min()))
        rows.append([(frame + 3) * 0.005, float(q[2]), float(v[2]), support, depth])
        if mpm is not None:
            mpm.check_sparse_grid_rebuild_status()
        if frame % 50 == 0:
            (args.output / "progress.json").write_text(
                json.dumps(dict(simulated_seconds=(frame + 3) * 0.005, total_seconds=args.seconds))
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
        warmup_seconds=0.01,
        final_sole_depth_m=float(rows[-1, 4]),
        mean_final_speed_m_s=float(abs(rows[after, 2]).mean()),
        mean_final_total_support_N=float(rows[after, 3].mean()),
        backing_floor_proximity=bool(rows[:, 4].max() >= args.bed_depth - 0.002),
        interpretation=(
            "Total support inferred from momentum; includes rigid bottom contact if reached. "
            "One vertical DOF; load is independent of inertial mass."
        ),
    )
    np.savez_compressed(args.output / "trace.npz", trace=rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry_file", type=Path, required=True)
    p.add_argument("--field", choices=("rft", "mpm"), default="mpm")
    p.add_argument("--mass", type=float, default=16.0)
    p.add_argument("--load", type=float, default=158.13)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--relaxation", type=float, default=1.0)
    p.add_argument("--bearing", type=float, default=0.87574)
    p.add_argument("--strength_multiplier", type=float, default=1.0)
    p.add_argument("--voxel_size", type=float, default=0.01)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--particles_per_cell", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--geometry", default="g1")
    p.add_argument("--bed_depth", type=float, default=0.118)
    p.add_argument("--density", type=float, default=1600.0)
    p.add_argument("--friction", type=float, default=0.4)
    p.add_argument("--roughness", type=float, default=1.0)
    p.add_argument("--modulus", type=float, default=1e15)
    p.add_argument("--domain_scale", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=0.5)
    run(p.parse_args())
