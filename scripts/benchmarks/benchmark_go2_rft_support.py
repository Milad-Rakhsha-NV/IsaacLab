# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check 10-second load-controlled Go2 sphere support before policy training."""

import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from newton.solvers import SolverMuJoCo, SolverRFT


@wp.kernel
def apply_load(forces: wp.array[wp.spatial_vector], loads: wp.array[float]):
    i = wp.tid()
    forces[i] = wp.spatial_vector(0.0, 0.0, 9.81 - loads[i], 0.0, 0.0, 0.0)


def main(args):
    """Distinguish a holding failure from geometric quadrature error and overload."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nominal = 147.33641052246094 / 4
    loads = np.array([0.5, 1.0, 1.5]) * nominal
    strength = 1600 * 9.81 * (894 * 0.4**3 - 386 * 0.4**2 + 89 * 0.4)
    # For a fully immersed sphere, C(d) = beta*S*pi*r^2*(2*d/3-r/6).
    # Start the two supported loads at 110% bearing capacity, without impact.
    initial_depths = np.minimum(0.116, 1.5 * 1.1 * loads / (0.87574 * strength * np.pi * 0.022**2) + 0.022 / 4)
    results, histories = [], {}
    for substeps in (4, 8, 16):
        builder = newton.ModelBuilder()
        for i in range(3):
            pose = wp.transform(wp.vec3(i * 0.1, 0, 0.022 - initial_depths[i]), wp.quat_identity())
            body = builder.add_link(
                xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, label=f"foot_{i}"
            )
            joint = builder.add_joint_prismatic(
                parent=-1,
                child=body,
                parent_xform=pose,
                axis=(0, 0, 1),
                target_ke=0,
                target_kd=0,
                damping=0,
                armature=0,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
            builder.add_articulation([joint])
            builder.add_shape_sphere(body, radius=0.022, label=f"foot_{i}/sphere", cfg=builder.ShapeConfig(density=0))
        builder.add_ground_plane(height=-0.118)
        model = builder.finalize(device="cuda:0")
        surfaces, inventory = collision_surfaces(
            model, "foot_.*", 12, resolutions=[dict(shape_pattern="foot_.*/sphere", sphere_subdivisions=2)]
        )
        rigid = SolverMuJoCo(model, use_mujoco_contacts=True, njmax=64, nconmax=32)
        solver = SolverRFT(
            model,
            surfaces,
            rigid_solver=rigid,
            surface_height=0,
            surface_bottom=-0.118,
            surface_shape_ids=[r["shape"] for r in inventory],
            surface_quadrature_orders=[6] * 3,
            config=SolverRFT.Config(bearing=0.87574, velocity_regularization=0.04),
        )
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        initial_capacity = solver.evaluate(state).numpy()[:, 2]
        np.testing.assert_allclose(initial_capacity[:2], 1.1 * loads[:2], rtol=5e-4)
        control = model.control()
        load_array = wp.array(loads, dtype=float, device=model.device)
        dt = 0.005 / substeps

        def advance():
            for _ in range(substeps):
                state.clear_forces()
                wp.launch(apply_load, dim=3, inputs=[state.body_f, load_array], device=model.device)
                solver.step(state, state, control, None, dt)

        advance()
        with wp.ScopedCapture(device=model.device) as capture:
            advance()
        rows = [[0.0, *initial_depths], [0.01, *(-state.body_q.numpy()[:, 2] + 0.022)]]
        for frame in range(2, 2000):
            wp.capture_launch(capture.graph)
            if (frame + 1) % 10 == 0:
                rows.append([(frame + 1) * 0.005, *(-state.body_q.numpy()[:, 2] + 0.022)])
        values = np.array(rows)
        histories[f"substeps_{substeps}"] = values
        initial = values[np.argmin(abs(values[:, 0] - 2)), 1:]
        final = values[-1, 1:]
        results.append(
            dict(
                substeps=substeps,
                depths_2s_m=initial.tolist(),
                depths_10s_m=final.tolist(),
                creep_2_to_10s_m=(final - initial).tolist(),
                substrate_proximity_10s=(final > 0.117).tolist(),
            )
        )
    np.savez_compressed(args.output.with_suffix(".npz"), **histories)
    args.output.write_text(
        json.dumps(
            dict(
                results=results,
                loads_N=loads.tolist(),
                initialization="At rest at an admissible preloaded depth; no landing impact",
                initial_depths_m=initial_depths.tolist(),
                initial_bearing_capacity_N=initial_capacity.tolist(),
                foot_radius_m=0.022,
                triangles_per_foot=128,
                quadrature_order=6,
                bed_depth_m=0.118,
                bearing=0.87574,
                velocity_regularization_m_s=0.04,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                newton_kernel_sha256=hashlib.sha256(
                    (Path(newton.__file__).parent / "_src/solvers/rft/kernels.py").read_bytes()
                ).hexdigest(),
                interpretation="Vertical prismatic 1 kg intruders with independent applied loads. "
                "Nominal load is one quarter of measured Go2 weight; this is a solver fixture, "
                "not articulated standing or a matched MPM validation.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
