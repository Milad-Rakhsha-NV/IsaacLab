# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure selective Go2 contact integration on exact recorded motion or 4096 worlds."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from newton.solvers import SolverRFT

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.go2.rft_env import Go2RFTEnv
from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTCurriculumEnvCfg


def errors(values, reference):
    """Keep absolute floors visible when normalizing small contact loads."""
    delta = values - reference
    result = {}
    for name, components, floor in (("force", slice(0, 3), 1.0), ("torque", slice(3, 6), 0.01)):
        a, b = delta[..., components], reference[..., components]
        relative = np.linalg.norm(a, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), floor)
        result[name] = dict(
            pooled=float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12)),
            p95=float(np.percentile(relative, 95)),
            maximum=float(relative.max()),
            absolute_maximum=float(np.linalg.norm(a, axis=-1).max()),
            denominator_floor=floor,
        )
    return result


def copy_geometry(original, inventory, count):
    """Batch unchanged bodies and collider unions without articulation integration."""
    builder = newton.ModelBuilder()
    com = original.body_com.numpy()
    for sample in range(count):
        bodies = [
            builder.add_body(com=wp.vec3(c), mass=1.0, inertia=wp.mat33(np.eye(3)), label=f"/sample_{sample}{label}")
            for c, label in zip(com, original.body_label, strict=True)
        ]
        for row in inventory:
            position, quaternion = row["local_pose"][:3], row["local_pose"][3:]
            kwargs = dict(
                body=bodies[row["body"]],
                xform=wp.transform(wp.vec3(position), wp.quat(quaternion)),
                label=f"/sample_{sample}{row['label']}",
                cfg=builder.ShapeConfig(density=0),
            )
            x, y, z = row["scale"]
            if row["kind"] == "SPHERE":
                builder.add_shape_sphere(radius=x, **kwargs)
            elif row["kind"] == "CYLINDER":
                builder.add_shape_cylinder(radius=x, half_height=y, **kwargs)
            elif row["kind"] == "BOX":
                builder.add_shape_box(hx=x, hy=y, hz=z, **kwargs)
            else:
                raise ValueError(f"Unhandled Go2 collider: {row['kind']}")
    return builder.finalize(device=original.device)


def motion_audit(args, env, profile):
    """Compare integration rules on identical exact poses and COM twists."""
    original = env.rft.model
    _, inventory = collision_surfaces(original, ".*", 12)
    poses, twists, sources = [], [], []
    rng = np.random.default_rng(42)
    for path in args.trajectories:
        with np.load(path) as data:
            frames = np.sort(rng.choice(np.arange(5, len(data["body_pose"])), args.samples, replace=False))
            poses.extend(data["body_pose"][frames])
            twists.extend(data["body_twist"][frames])
        sources.append(
            dict(path=str(path), frames=frames.tolist(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    poses, twists = np.asarray(poses), np.asarray(twists)
    model = copy_geometry(original, inventory, len(poses))
    state = model.state()
    state.body_q.assign(poses.reshape(-1, 7))
    state.body_qd.assign(twists.reshape(-1, 6))
    labels = original.body_label
    feet = [i for i, label in enumerate(labels) if label.endswith("_foot")]
    calves = [i for i, label in enumerate(labels) if label.endswith("_calf")]
    candidates = [("legacy", 0), ("faceted128", 0), ("selected", None)] + [("curved", order) for order in args.orders]
    material_cases = list(itertools.product((0.003, 0.03, 0.06, 0.118), (0.3, 0.5), (0.6, 1.0)))
    results, arrays = [], {}
    start = time.monotonic()
    for candidate, order in candidates:
        rules = (
            []
            if candidate == "legacy"
            else [
                dict(r, primitive_quadrature_order=r["primitive_quadrature_order"] if order is None else order)
                for r in profile["solver"]["surface_resolution"]
            ]
        )
        surfaces, rows = collision_surfaces(model, ".*", 12, resolutions=rules)
        orders = [row.get("resolution", {}).get("primitive_quadrature_order", 0) for row in rows]
        probe = SolverRFT(
            model,
            surfaces,
            config=env.rft.config,
            surface_height=0.118,
            surface_bottom=0.0,
            surface_shape_ids=[r["shape"] for r in rows],
            surface_quadrature_orders=orders,
        )
        material = probe.material.numpy()
        material[:, 0] = 1800.0
        values = []
        for height, friction, roughness in material_cases:
            material[:, 1:3] = [friction, friction * roughness]
            material[:, 3] = 1800 * 9.81 * (894 * friction**3 - 386 * friction**2 + 89 * friction) * 1.2
            probe.material.assign(material)
            probe.surface_height.fill_(height)
            values.append(probe.evaluate(state).numpy().reshape(len(poses), original.body_count, 6))
        key = f"{candidate}-{order}"
        arrays[key] = np.array(values)
        results.append(
            dict(
                candidate=candidate,
                quadrature_order=order,
                triangles_per_robot=probe.triangle_count // len(poses),
                analytic_points_per_robot=probe.curved_point_count // len(poses),
            )
        )
        args.output.with_suffix(".progress.json").write_text(json.dumps(dict(candidate=key, completed=len(results))))
    reference = arrays[f"curved-{args.orders[-1]}"]
    for row, values in zip(results, arrays.values(), strict=True):
        row["feet"] = errors(values[..., feet, :], reference[..., feet, :])
        row["calves"] = errors(values[..., calves, :], reference[..., calves, :])
        row["by_height"] = [
            dict(
                height=height,
                feet=errors(values[i, :, feet], reference[i, :, feet]),
                calves=errors(values[i, :, calves], reference[i, :, calves]),
            )
            for i, (height, _, _) in enumerate(material_cases)
        ]
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays, body_pose=poses, body_twist=twists)
    return dict(
        results=results,
        sources=sources,
        material_cases=material_cases,
        profile=profile,
        wall_seconds=time.monotonic() - start,
        body_labels=labels,
        source_inventory=inventory,
        config=dict(
            density=1800.0,
            strength_multiplier=1.2,
            bearing=env.rft.config.bearing,
            velocity_regularization=env.rft.config.velocity_regularization,
        ),
        interpretation="Exact common rigid-policy motion; no policy or material validation. "
        "Reference refines curved quadrature on the same selected contact geometry. "
        "Unselected surfaces remain unchanged. Normalization floors: 1 N and 0.01 Nm.",
    )


def performance(args, env, profile):
    """Measure actual vectorized environment stepping and captured RFT evaluation."""
    action = torch.zeros((env.num_envs, 12), device=env.device)
    for _ in range(10):
        env.step(action)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    model = env.rft.model
    state = model.state()
    with np.load(args.performance_motion) as data:
        pose, twist = data["body_pose"][100].copy(), data["body_twist"][100]
    assert model.body_count == env.num_envs * len(pose)
    origins = env.scene.env_origins.cpu().numpy()
    pose[:, :2] -= pose[0, :2]
    evaluation = {}
    state.body_qd.assign(np.tile(twist, (env.num_envs, 1)))
    env.rft.surface_height.fill_(0.118)
    env.rft.surface_bottom.fill_(0)
    for name, lift in (("shallow", 0.09), ("deep", 0.0)):
        poses = np.tile(pose, (env.num_envs, 1, 1))
        poses[:, :, :3] += origins[:, None, :]
        poses[:, :, 2] += lift
        state.body_q.assign(poses.reshape(-1, 7))
        env.rft.evaluate(state)
        with wp.ScopedCapture(device=model.device) as capture:
            env.rft.evaluate(state)
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        evaluation[name] = (time.perf_counter() - start) / 100
    free, total = torch.cuda.mem_get_info()
    return dict(
        num_envs=env.num_envs,
        steps=args.steps,
        wall_seconds=elapsed,
        environment_steps_per_second=env.num_envs * args.steps / elapsed,
        control_step_seconds=elapsed / args.steps,
        common_motion_rft_evaluation_seconds=evaluation,
        performance_motion=str(args.performance_motion),
        performance_motion_frame=100,
        performance_motion_sha256=hashlib.sha256(args.performance_motion.read_bytes()).hexdigest(),
        gpu_used_bytes=total - free,
        gpu_total_bytes=total,
        triangles_per_robot=env.rft.triangle_count // env.num_envs,
        analytic_points_per_robot=env.rft.curved_point_count // env.num_envs,
        inventory=env.rft.surface_inventory[: len(env.rft.surface_inventory) // env.num_envs],
        profile=profile,
        interpretation="4096 native Go2 worlds; zero-action throughput only, no training. "
        "Control-step trajectories can differ. RFT evaluations use identical recorded poses/twists "
        "and layer heights, at two vertical offsets. "
        "GPU usage includes the CUDA context and other processes.",
    )


def main(args):
    """Keep validation profiles explicit and historical task defaults unchanged."""
    profile = json.loads(args.profile.read_text())
    cfg = Go2RFTCurriculumEnvCfg()
    cfg.scene.num_envs = args.num_envs if args.mode == "performance" else 1
    cfg.curriculum_enabled = False
    cfg.randomize_material = False
    cfg.sim.visualizer_cfgs = []
    if args.mode == "performance" and not args.legacy:
        for name, value in profile["solver"].items():
            setattr(cfg.sim.physics.solver_cfg, name, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg):
        env = Go2RFTEnv(cfg)
        try:
            env.reset()
            result = performance(args, env, profile) if args.mode == "performance" else motion_audit(args, env, profile)
            result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            result["newton_kernel_sha256"] = hashlib.sha256(
                (Path(newton.__file__).parent / "_src/solvers/rft/kernels.py").read_bytes()
            ).hexdigest()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("motion", "performance"), default="motion")
    parser.add_argument("--profile", type=Path, default=Path("scripts/benchmarks/configs/go2_rft_contact_128.json"))
    parser.add_argument("--trajectories", nargs="+", type=Path, default=[])
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3, 4, 6, 10])
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--performance_motion", type=Path, default=Path("logs/go2_granular/go2/refinement/motion-rigid-1/rollout.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
