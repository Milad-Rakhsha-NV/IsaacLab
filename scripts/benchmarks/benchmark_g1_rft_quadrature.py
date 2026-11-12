# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit sole and whole-foot RFT quadrature using recorded exact G1 motion."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from newton.solvers import SolverRFT

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.g1.granular_env_cfg import G1GranularRFTEnvCfg
from isaaclab_tasks.core.velocity.granular_env import GranularVelocityEnv


def material_cases(profile):
    """Cover height and angular endpoints; density and strength scale the traction."""
    ranges = profile["environment"]
    cases = [
        dict(
            height=0.118,
            density=1600.0,
            friction=0.4,
            roughness=0.3,
            strength=profile["solver"]["strength_multiplier"],
            bearing=profile["solver"]["bearing"],
        )
    ]
    for height, friction, roughness, bearing in itertools.product(
        (0.003, 0.01, 0.03, 0.06, 0.118), ranges["friction_range"], ranges["roughness_range"], ranges["bearing_range"]
    ):
        cases.append(
            dict(
                height=height,
                density=max(ranges["density_range"]),
                friction=friction,
                roughness=roughness,
                strength=max(ranges["strength_multiplier_range"]),
                bearing=bearing,
            )
        )
    return cases


def errors(force, reference):
    """Report pooled errors and loaded-sample errors with explicit normalization floors."""
    delta = force - reference
    f = np.linalg.norm(delta[..., :3], axis=-1) / np.maximum(np.linalg.norm(reference[..., :3], axis=-1), 20)
    t = np.linalg.norm(delta[..., 3:], axis=-1) / np.maximum(np.linalg.norm(reference[..., 3:], axis=-1), 1)
    loaded = np.linalg.norm(reference[..., :3], axis=-1) > 1
    return dict(
        force_relative_error=float(np.linalg.norm(delta[..., :3]) / max(np.linalg.norm(reference[..., :3]), 1e-12)),
        torque_relative_error=float(np.linalg.norm(delta[..., 3:]) / max(np.linalg.norm(reference[..., 3:]), 1e-12)),
        loaded_force_error_p95=float(np.percentile(f[loaded], 95)),
        loaded_torque_error_p95=float(np.percentile(t[loaded], 95)),
        maximum_force_error=float(f.max()),
        maximum_torque_error=float(t.max()),
    )


def main(args):
    """Batch independent prescribed foot states; do not integrate or change trained physics."""
    profile = json.loads(args.profile.read_text())
    cfg = G1GranularRFTEnvCfg()
    cfg.scene.num_envs = 1
    cfg.randomize_material = False
    cfg.curriculum_enabled = False
    cfg.sim.visualizer_cfgs = []
    for name, value in profile["solver"].items():
        setattr(cfg.sim.physics.solver_cfg, name, value)
    with launch_simulation(cfg):
        env = GranularVelocityEnv(cfg)
        try:
            env.reset()
            assert env.rft.config.inertial_coefficient == 0.0
            original = env.rft.model
            surfaces, inventory = collision_surfaces(original, cfg.sim.physics.solver_cfg.body_pattern, 12)
            feet = [(body, mesh) for body, mesh in surfaces if "ankle_roll" in original.body_label[body]]
            assert len(feet) == 2
            foot_ids = [body for body, _ in feet]
            rng = np.random.default_rng(42)
            poses, velocities, sources = [], [], []
            for trajectory in args.trajectories:
                with np.load(trajectory) as data:
                    pose, velocity = data["body_pose"], data["body_twist"]
                    assert velocity.shape == (*pose.shape[:2], 6)
                    indices = np.sort(
                        rng.choice(np.arange(5, len(pose)), min(args.samples, len(pose) - 5), replace=False)
                    )
                    poses.extend(pose[indices])
                    velocities.extend(velocity[indices])
                sources.append(
                    dict(
                        path=str(trajectory),
                        velocity_source="recorded exact world-space COM twists",
                        frames=indices.tolist(),
                        sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                    )
                )
            poses, velocities = np.array(poses), np.array(velocities)
            count = len(poses)
            builder = newton.ModelBuilder()
            com = original.body_com.numpy()
            batch_surfaces = []
            for sample in range(count):
                for body, mesh in feet:
                    index = builder.add_body(
                        com=wp.vec3(*com[body]),
                        mass=1.0,
                        inertia=wp.mat33(np.eye(3)),
                        label=f"sample_{sample}_{original.body_label[body]}",
                    )
                    batch_surfaces.append((index, mesh))
            model = builder.finalize(device=original.device)
            np.testing.assert_allclose(original.gravity.numpy()[0], original.gravity.numpy()[-1])
            model.gravity.fill_(wp.vec3(*original.gravity.numpy()[0]))
            state = model.state()
            state.body_q.assign(poses[:, foot_ids].reshape(-1, 7))
            state.body_qd.assign(velocities[:, foot_ids].reshape(-1, 6))
            cases = material_cases(profile)
            results, parity = [], []
            args.output.parent.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            for edge in (None, 0.04, 0.02, 0.01):
                probe = SolverRFT(
                    model,
                    batch_surfaces,
                    config=env.rft.config,
                    surface_height=0.118,
                    surface_bottom=0.0,
                    max_edge_length=edge,
                )
                per_foot = probe.triangle_count // (count * 2)
                vertices = probe.triangle_vertices.numpy().reshape(count, 2, per_foot, 3, 3)
                minimum = vertices[0, :, :, :, 2].min(axis=(1, 2))
                sole = np.all(np.isclose(vertices[0, :, :, :, 2], minimum[:, None, None], atol=1e-7), axis=-1)
                material = probe.material.numpy()
                whole, soles = [], []
                for case_id, case in enumerate(cases):
                    mu = case["friction"]
                    material[:, 0] = case["density"]
                    material[:, 1] = mu
                    material[:, 2] = mu * case["roughness"]
                    material[:, 3] = case["density"] * 9.81 * (894 * mu**3 - 386 * mu**2 + 89 * mu) * case["strength"]
                    material[:, 6] = case["bearing"]
                    probe.material.assign(material)
                    probe.surface_height.fill_(case["height"])
                    force = probe.evaluate(state).numpy().reshape(count, 2, 6)
                    triangle_force = probe.triangle_f.numpy().reshape(count, 2, per_foot, 6)
                    np.testing.assert_allclose(triangle_force.sum(axis=2), force, rtol=2e-4, atol=2e-3)
                    whole.append(force.copy())
                    soles.append((triangle_force * sole[None, :, :, None]).sum(axis=2))
                    if edge is None and case_id == 0:
                        original_state = original.state()
                        saved_material, saved_height = env.rft.material.numpy(), env.rft.surface_height.numpy()
                        env.rft.material.assign(np.tile(material[0], (len(saved_material), 1)))
                        env.rft.surface_height.fill_(case["height"])
                        selected = np.unique(
                            [0, count // 2, count - 1, np.linalg.norm(force[..., :3], axis=-1).argmax() // 2]
                        )
                        for sample in selected:
                            original_state.body_q.assign(poses[sample])
                            original_state.body_qd.assign(velocities[sample])
                            expected = env.rft.evaluate(original_state).numpy()[foot_ids]
                            np.testing.assert_allclose(force[sample], expected, rtol=2e-4, atol=2e-3)
                            parity.append(
                                dict(
                                    sample=int(sample),
                                    maximum_absolute_difference=float(abs(force[sample] - expected).max()),
                                )
                            )
                        env.rft.material.assign(saved_material)
                        env.rft.surface_height.assign(saved_height)
                    args.output.with_suffix(".progress.json").write_text(
                        json.dumps(
                            dict(
                                max_edge_length=edge,
                                completed_material_cases=case_id + 1,
                                total_material_cases=len(cases),
                                motion_samples=count,
                            )
                        )
                    )
                results.append(
                    dict(
                        max_edge_length=edge,
                        triangles_per_foot=per_foot,
                        sole_triangles_per_foot=sole.sum(axis=1).tolist(),
                        whole=np.array(whole),
                        sole=np.array(soles),
                    )
                )
                del probe
            reference = results[-1]
            rows = []
            for result in results:
                rows.append(
                    dict(
                        max_edge_length=result["max_edge_length"],
                        triangles_per_foot=result["triangles_per_foot"],
                        sole_triangles_per_foot=result["sole_triangles_per_foot"],
                        feet=[
                            dict(
                                body=original.body_label[body],
                                whole_foot=errors(result["whole"][:, :, foot], reference["whole"][:, :, foot]),
                                sole_only=errors(result["sole"][:, :, foot], reference["sole"][:, :, foot]),
                            )
                            for foot, body in enumerate(foot_ids)
                        ],
                    )
                )
            kernel = Path(newton.__file__).parent / "_src/solvers/rft/kernels.py"
            record = dict(
                results=rows,
                trajectories=sources,
                material_cases=cases,
                motion_samples=count,
                independent_foot_states=2 * count,
                original_model_parity=parity,
                foot_inventory=[row for row in inventory if row["body"] in foot_ids],
                wall_seconds=time.monotonic() - start,
                rft_kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
                torque_reference="world-space moment about the actual foot body COM",
                normalization=(
                    "Pooled L2 error; loaded p95 uses 20 N / 1 N m denominator floors and reference force >1 N."
                ),
                interpretation=(
                    "Surface quadrature self-convergence on exact current-policy poses/twists. "
                    "This is not physical validation. "
                    "Only feet are counted. Sole-only forces include bottom-face triangles; "
                    "whole-foot forces also include side/top faces. The 10 mm edge mesh is the numerical reference. "
                    "Density/strength scale traction with inertial correction disabled."
                ),
            )
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(rows))
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
