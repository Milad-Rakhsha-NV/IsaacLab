# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen direct Go2 policy in RFT or MPM; MPM is never used for training.

uv run python scripts/benchmarks/evaluate_go2_granular.py --steps 1000 --stand
uv run python scripts/benchmarks/evaluate_go2_granular.py --checkpoint MODEL --field mpm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from isaaclab.app import launch_simulation
from isaaclab.app.logging_utils import configure_console_logging

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--field", choices=("rft", "mpm", "rigid"), default="rft")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output", type=Path, default=Path("logs/go2_granular/stance"))
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stand", action="store_true")
parser.add_argument(
    "--command_schedule", action="store_true", help="Rigid-floor stand/walk/max-speed/steering/stop test."
)
parser.add_argument("--video", action="store_true")
parser.add_argument("--no-bearing", action="store_true")
parser.add_argument("--randomize", action="store_true")
parser.add_argument("--segments", type=int, default=12)
parser.add_argument("--voxel_size", type=float, default=0.02)
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--density", type=float, default=1600.0)
parser.add_argument("--friction", type=float, default=0.4)
parser.add_argument("--roughness", type=float, default=1.0)
parser.add_argument("--proxy_relaxation", type=float, default=0.1)
parser.add_argument("--mpm_modulus", type=float, default=1e15)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--bed_depth", type=float, default=0.4)
parser.add_argument(
    "--substrate_depth", type=float, default=None, help="Optional RFT layer thickness; zero tests rigid limit."
)
parser.add_argument("--difficulty", type=float, default=1.0, help="Frozen curriculum level for randomized validation.")
args = parser.parse_args()
configure_console_logging()

from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTEnvCfg, Go2RigidEnvCfg

cfg = Go2RigidEnvCfg() if args.field == "rigid" else Go2RFTEnvCfg()
cfg.sim.visualizer_cfgs = [cfg.sim.visualizer_cfgs]
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.randomize_material = args.randomize and args.field == "rft"
cfg.randomize_commands = args.randomize and not args.stand
cfg.surface_relative_height = True
cfg.material_difficulty = args.difficulty
cfg.observation_noise = False
cfg.command_speed = 0.0 if args.stand else args.speed
schedule = (
    (
        (0.0, (0.0, 0.0, 0.0), "Stand"),
        (3.0, (0.3, 0.0, 0.0), "Walk"),
        (10.0, (0.5, 0.0, 0.0), "Maximum forward command"),
        (17.0, (0.3, 0.1, 0.0), "Walk and move left"),
        (23.0, (0.3, 0.0, 0.3), "Walk and turn"),
        (29.0, (0.0, 0.0, 0.0), "Stop"),
    )
    if args.command_schedule
    else ()
)
if schedule and (args.field != "rigid" or args.randomize or args.stand):
    parser.error("The command schedule requires rigid inference without --randomize or --stand")
if args.field == "rft":
    cfg.bed_depth = args.substrate_depth
    cfg.sim.physics.solver_cfg.bearing = 0.0 if args.no_bearing else 0.87574
    cfg.sim.physics.solver_cfg.segments = args.segments
    cfg.sim.physics.solver_cfg.bulk_density = args.density
    cfg.sim.physics.solver_cfg.internal_friction = args.friction
    cfg.sim.physics.solver_cfg.surface_friction = args.friction * args.roughness
cfg.episode_length_s = args.steps * 0.02 + 10
cfg.sim.physics.num_substeps = args.substeps
if args.video:
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    cfg.sim.visualizer_cfgs[0].backend = NewtonGLBackendCfg(enable_picking=False)
    cfg.sim.visualizer_cfgs[0].headless = True
    cfg.sim.visualizer_cfgs[0].window.size = (960, 640)
    cfg.sim.visualizer_cfgs[0].focal_length = 26
else:
    cfg.sim.visualizer_cfgs[0].backend = []

if args.field == "mpm":
    if args.num_envs != 1:
        raise ValueError("The initial MPM reference supports one isolated bed")
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import CuboidCfg, PreviewSurfaceCfg, UsdPhysicsCollisionCfg

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    cfg.randomize_material = False
    cfg.scene.bed = None
    cfg.scene.floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=CuboidCfg(
            size=(2.4, 1.6, 0.1),
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0, -args.bed_depth - 0.05)),
    )
    cfg.scene.sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            lower=(-0.7, -0.6, -args.bed_depth),
            upper=(1.3, 0.6, 0),
            voxel_size=args.voxel_size,
            particles_per_cell=2,
            particle_placement="cell_center",
            jitter=0,
            material=MPMParticleMaterialCfg(
                density=args.density,
                young_modulus=args.mpm_modulus,
                friction=args.friction,
                yield_pressure=1e15,
                tensile_yield_ratio=0,
            ),
            visual_color=(0.68, 0.46, 0.25),
        ),
    )
    cfg.sim.physics = NewtonCfg(
        num_substeps=args.substeps,
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(njmax=128, nconmax=64),
                    bodies=[r"/World/envs/env_.*/Robot"],
                    include_static_shapes=False,
                ),
                CouplerEntryCfg(
                    name="mpm",
                    solver_cfg=MPMSolverCfg(
                        voxel_size=args.voxel_size,
                        grid_type="sparse",
                        separate_worlds=True,
                        max_active_cell_count=max(
                            1 << 19, 1 << int(np.ceil(np.log2(4.8 * args.bed_depth / args.voxel_size**3)))
                        ),
                        max_leaf_node_count=1 << 15,
                        max_lower_node_count=1 << 11,
                        max_upper_node_count=1 << 9,
                        max_iterations=50,
                        tolerance=1e-4,
                        collider_basis="pic27",
                        strain_basis="P0",
                        transfer_scheme="pic",
                        air_drag=1.0,
                        project_outside_colliders=False,
                    ),
                    all_particles=True,
                    include_child_joints=False,
                    include_static_shapes=True,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="mpm",
                    bodies=[r"/World/envs/env_.*/Robot"],
                    mode="staggered",
                    proxy_relaxation=args.proxy_relaxation,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
    )

args.output.mkdir(parents=True, exist_ok=True)
with launch_simulation(cfg):
    import torch
    import warp as wp

    from isaaclab_tasks.core.velocity.config.go2.rft_env import Go2RFTEnv

    class InferenceEnv(Go2RFTEnv):
        def _get_dones(self):
            # Preserve failed poses for recording; do not hide falls by resetting.
            zero = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return zero, zero

    env = InferenceEnv(cfg)
    obs, _ = env.reset()
    solver = env.sim.backend_manager.backend.solver
    model = solver.model
    labels = model.body_label
    checkpoint = None
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.utils import to_dict

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_rft_ppo_cfg import Go2RFTPPORunnerCfg

        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, to_dict(Go2RFTPPORunnerCfg()), log_dir=None, device=env.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        checkpoint = str(args.checkpoint.resolve())
        obs = wrapped.get_observations()
    else:

        def policy(observations):
            return torch.zeros(args.num_envs, 12, device=env.device)

    if args.field == "rft":
        inventory = solver.surface_inventory
        (args.output / "surfaces.json").write_text(json.dumps(inventory, indent=2))
        mpm = None
    elif args.field == "mpm":
        mpm = solver.solver("mpm")
        # Asset-local material bindings can override a root USD material.
        # Set the actual MPM collider friction through its public setup API,
        # before capture, rather than assume the authored override propagated.
        collider_mu = wp.full(
            mpm.collider_body_index.shape[0], args.friction * args.roughness, dtype=float, device=model.device
        )
        mpm.setup_collider(
            collider_friction=collider_mu.numpy().tolist(),
            body_mass=mpm.model.body_mass,
            body_inv_inertia=mpm.model.body_inv_inertia,
        )
        collision_data = mpm._mpm_model.collider
        face_mu = collision_data.material_friction.numpy()[collision_data.face_material_index.numpy()]
        np.testing.assert_allclose(face_mu, args.friction * args.roughness, rtol=1e-6)
        (args.output / "mpm_bindings.json").write_text(
            json.dumps(
                dict(
                    body_labels=mpm.model.body_label,
                    collider_friction=collider_mu.numpy().tolist(),
                    actual_face_friction_range=[float(face_mu.min()), float(face_mu.max())],
                    shape_labels=mpm.model.shape_label,
                    shape_flags=mpm.model.shape_flags.numpy().tolist(),
                    shape_friction=mpm.model.shape_material_mu.numpy().tolist(),
                    shape_body=mpm.model.shape_body.numpy().tolist(),
                    body_world=mpm.model.body_world.numpy().tolist(),
                    collider_body=mpm.collider_body_index.numpy().tolist(),
                ),
                indent=2,
            )
        )
    else:
        mpm = None
    rigid = solver.rigid_solver if args.field == "rft" else solver.solver("rigid") if mpm is not None else solver
    ambient_density = float(rigid.mj_model.opt.density)
    if ambient_density != 0:
        raise ValueError(
            f"Unexpected ambient fluid density {ambient_density}; granular bulk density must remain separate"
        )
    writer = None
    if args.video:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(args.output / "rollout.mp4"),
            (960, 640),
            fps=50,
            codec="libx264",
            output_params=["-crf", "21", "-movflags", "+faststart"],
        )
        writer.send(None)
    rows, force_rows, joint_rows, pose_rows, metric_rows, gait_rows, command_rows = [], [], [], [], [], [], []
    steps_per_foot = np.zeros((args.num_envs, 4), dtype=int)
    phase = None
    max_rigid_overflow = 0
    previous_impulse = solver.body_impulse.numpy().copy() if args.field == "rft" else None
    start = time.monotonic()
    command_manifest = env.commands.cpu().tolist()
    material_manifest = solver.material.numpy().tolist() if args.field == "rft" else None
    height_manifest = solver.surface_height.numpy().tolist() if args.field == "rft" else [0.0]
    body_com_manifest = model.body_com.numpy().tolist()
    execution_error = None
    first_failure = None
    first_exit = None
    first_bottom = None
    particles_finite = True
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                if schedule:
                    current = max(entry for entry in schedule if entry[0] <= step * env.step_dt)
                    if phase != current[2]:
                        phase = current[2]
                        env.commands[:] = torch.tensor(current[1], device=env.device)
                        if args.checkpoint:
                            from tensordict import TensorDict

                            obs = TensorDict(env._get_observations(), batch_size=[args.num_envs])
                        else:
                            obs = env._get_observations()
                action = policy(obs)
                if args.checkpoint:
                    obs, reward, _, _ = wrapped.step(action)
                else:
                    obs, reward, _, _, _ = env.step(action)
            p = env.robot.data.root_pos_w.torch.cpu().numpy()
            v = env.robot.data.root_lin_vel_b.torch.cpu().numpy()
            g = env.robot.data.projected_gravity_b.torch.cpu().numpy()
            overflow = rigid.mjw_data.overflow.numpy()
            max_rigid_overflow |= int(np.bitwise_or.reduce(overflow.reshape(-1)))
            effort = env.robot.actuators.applied_effort.torch.cpu().numpy()
            joint_velocity = env.robot.data.joint_vel.torch.cpu().numpy()
            limits = env.robot.data.joint_effort_limits.torch.cpu().numpy()
            command_rows.append(env.commands.cpu().numpy().copy())
            if cfg.gait is not None:
                signed_y = env.signed_foot_y().cpu().numpy()
                contact = env.contact_forces.data.net_normal_forces_w.torch[:, env.foot_sensor_ids].norm(dim=-1) > 1.0
                slip = env.robot.data.body_lin_vel_w.torch[:, env.foot_ids, :2].norm(dim=-1)
                mean_slip = ((slip * contact).sum(-1) / contact.sum(-1).clamp_min(1)).cpu().numpy()
                gait_rows.append(np.column_stack((signed_y, contact.cpu().numpy(), mean_slip)))
                first = env.contact_forces.compute_first_contact(env.step_dt).torch[:, env.foot_sensor_ids]
                air = env.contact_forces.data.last_air_time.torch[:, env.foot_sensor_ids]
                moving = env.commands.norm(dim=-1) > 0.05
                if step * env.step_dt > 1.0:
                    steps_per_foot += ((air > 0.04) & first & moving[:, None]).cpu().numpy()
            if args.field == "rft":
                impulse = solver.body_impulse.numpy()
                force = (impulse - previous_impulse) / env.step_dt
                previous_impulse = impulse.copy()
            elif mpm is not None:
                force = solver.get_proxy_body_wrenches("rigid", "mpm").numpy()
                particles_finite = bool(torch.isfinite(env.scene["sand"].data.particle_pos_w.torch).all())
            else:
                force = np.zeros((model.body_count, 6), dtype=np.float32)
            height = env.relative_base_height().cpu().numpy()
            failed = (height < 0.16) | (g[:, 2] > -0.3) | ~np.isfinite(p).all(-1)
            local_p = p - env.scene.env_origins.cpu().numpy()
            if first_exit is None and (
                np.any(local_p[:, 0] > 1.0) or np.any(local_p[:, 0] < -0.4) or np.any(np.abs(local_p[:, 1]) > 0.35)
            ):
                first_exit = (step + 1) * env.step_dt
            poses_now = env.sim.backend_manager.backend.state_0.body_q.numpy()
            feet = [i for i, label in enumerate(labels) if label.endswith("_foot")]
            if first_bottom is None and np.any(poses_now[feet, 2] - 0.022 < -args.bed_depth):
                first_bottom = (step + 1) * env.step_dt
            if first_failure is None and failed.any():
                first_failure = (step + 1) * env.step_dt
            rows.append(
                np.column_stack(
                    (
                        np.full(args.num_envs, (step + 1) * env.step_dt),
                        p,
                        v,
                        height,
                        np.linalg.norm(v[:, :2] - env.commands[:, :2].cpu().numpy(), axis=-1),
                        failed,
                    )
                )
            )
            force_rows.append(force)
            pose_rows.append(env.sim.backend_manager.backend.state_0.body_q.numpy().copy())
            joint_rows.append(env.robot.data.joint_pos.torch.cpu().numpy())
            metric_rows.append(
                np.column_stack(
                    (
                        np.abs(
                            env.robot.data.root_ang_vel_b.torch[:, 2].cpu().numpy() - env.commands[:, 2].cpu().numpy()
                        ),
                        np.abs(effort * joint_velocity).sum(-1),
                        (np.abs(effort) >= 0.95 * limits).mean(-1),
                        np.arccos(np.clip(-g[:, 2], -1, 1)),
                    )
                )
            )
            if args.video and np.isfinite(p).all():
                from PIL import Image, ImageDraw

                origin = p[0]
                env.sim.set_camera_view(tuple(origin + np.array([1.0, -1.3, 0.5])), tuple(origin))
                viz = env.sim.visualizers[0]
                viewer = viz._viewer
                poses = env.sim.backend_manager.backend.state_0.body_q.numpy()
                xyz, scalar = poses[:, 3:6], poses[:, 6:7]
                com = np.asarray(body_com_manifest)
                centers = poses[:, :3] + com + 2 * np.cross(xyz, np.cross(xyz, com) + scalar * com)
                active = np.linalg.norm(force[:, :3], axis=-1) > 1
                selected = np.array(
                    [i for i, label in enumerate(labels) if "/env_0/Robot/" in label and active[i]], dtype=int
                )
                viewer.log_arrows(
                    "granular/body_forces",
                    wp.array(centers[selected], dtype=wp.vec3),
                    wp.array(centers[selected] + force[selected, :3] / 200.0, dtype=wp.vec3),
                    wp.array(np.tile([0.1, 0.8, 0.2], (len(selected), 1)), dtype=wp.vec3),
                )
                frame = np.asarray(viz.render_rgb_array(), dtype=np.uint8)[..., :3]
                im = Image.fromarray(frame)
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 960, 65), fill=(15, 23, 42))
                draw.text(
                    (15, 12),
                    (
                        f"Go2 {args.field.upper()} | t={(step + 1) * env.step_dt:.2f}s | "
                        f"vx={v[0, 0]:.2f} m/s | z={height[0]:.3f} m"
                    ),
                    fill="white",
                )
                command = env.commands[0].cpu().tolist()
                draw.text(
                    (15, 35),
                    f"Command vx={command[0]:.2f}, vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s"
                    f" | {phase or 'Fixed command'} | fall: {first_failure}",
                    fill="white",
                )
                writer.send(np.ascontiguousarray(np.asarray(im)))
                if step == min(500 if schedule else 100, args.steps - 1):
                    im.save(args.output / "preview.jpg")
            if not np.isfinite(p).all() or not particles_finite:
                break
    except Exception as error:
        execution_error = str(error)
        raise
    finally:
        if writer:
            writer.close()
        np.savez_compressed(
            args.output / "rollout.npz",
            state=np.array(rows),
            body_wrench=np.array(force_rows),
            joint_pos=np.array(joint_rows),
            body_pose=np.array(pose_rows),
            metrics=np.array(metric_rows),
            gait=np.array(gait_rows),
            commands=np.array(command_rows),
        )
        summary = dict(
            field=args.field,
            ambient_fluid_density=ambient_density,
            max_rigid_overflow=max_rigid_overflow,
            surface_relative_height=True,
            validation_difficulty=args.difficulty,
            substrate_depth=args.substrate_depth,
            checkpoint=checkpoint,
            steps=len(rows),
            seed=args.seed,
            simulated_seconds=len(rows) * env.step_dt,
            wall_seconds=time.monotonic() - start,
            first_failure_seconds=first_failure,
            body_labels=labels,
            body_com=body_com_manifest,
            particles_finite=particles_finite,
            first_bed_exit_seconds=first_exit,
            first_bottom_contact_seconds=first_bottom,
            bed_depth=args.bed_depth,
            density=args.density,
            internal_friction=args.friction,
            roughness=args.roughness,
            commands=command_manifest,
            command_schedule=schedule,
            material_parameters=material_manifest,
            surface_heights=height_manifest,
            execution_error=execution_error,
            bearing=cfg.sim.physics.solver_cfg.bearing if args.field == "rft" else None,
            rigid_support_plane=args.field == "rigid" or args.substrate_depth is not None,
            material_randomization=cfg.randomize_material,
            voxel_size=args.voxel_size if mpm is not None else None,
            mpm_floor_center_xy=[0.3, 0.0] if mpm is not None else None,
            max_active_cell_count=cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_active_cell_count
            if mpm is not None
            else None,
            proxy_relaxation=args.proxy_relaxation if mpm is not None else None,
            mpm_modulus=args.mpm_modulus if mpm is not None else None,
            substeps=args.substeps,
            force_sampling="control-interval impulse average"
            if args.field == "rft"
            else "last coupled proxy feedback, including relaxation; not an interval average"
            if mpm is not None
            else "zero granular load; rigid contact reactions are not included",
        )
        if rows:
            state_history = np.array(rows)
            summary["failure_fraction"] = float(np.any(state_history[:, :, 9], axis=0).mean())
            summary["per_env_velocity_error"] = state_history[:, :, 8].mean(0).tolist()
            summary.update(
                mean_velocity_error=float(np.array(rows)[:, :, 8].mean()),
                final_position=p.tolist(),
                final_height=height.tolist(),
                mean_yaw_error=float(np.array(metric_rows)[:, :, 0].mean()),
                mean_absolute_joint_power_W=float(np.array(metric_rows)[:, :, 1].mean()),
                motor_saturation_fraction=float(np.array(metric_rows)[:, :, 2].mean()),
                max_tilt_radians=float(np.array(metric_rows)[:, :, 3].max()),
            )
            if gait_rows:
                gait_history = np.array(gait_rows)
                violations = (gait_history[:, :, :4] < cfg.gait.min_signed_foot_y).any(-1)
                summary.update(
                    gait_pass_fraction=float((~violations.any(axis=0)).mean()),
                    min_signed_foot_y=float(gait_history[:, :, :4].min()),
                    mean_contact_slip_m_s=float(gait_history[min(50, len(rows) - 1) :, :, 8].mean()),
                    steps_per_foot=steps_per_foot.tolist(),
                )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        env.close()
        print(
            json.dumps(
                {
                    key: summary.get(key)
                    for key in ("field", "steps", "first_failure_seconds", "particles_finite", "mean_velocity_error")
                }
            )
        )
