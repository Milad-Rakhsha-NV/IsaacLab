# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a release checkpoint with fixed forward commands and optional offscreen video.

Accepts the RSL-RL playback arguments and physics overrides, plus ``--metrics PATH``,
``--steps 1200``, ``--forward-speed 0.5``, and optional ``--video-file PATH``.
Records raw body-state finiteness, upright posture, motion, and premature terminations.
Reward totals are deliberately not an acceptance criterion across MDP versions.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import warp as wp
from dvi_release_metrics import world_velocity_metrics
from newton import JointType

wp.config.enable_backward = False

from isaaclab.app import launch_simulation  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl, play_rsl_rl  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def joint_anchor_error(body_poses: np.ndarray, model) -> float | None:
    """Measure coincident joint-anchor drift in the recorded first world's body poses."""
    count = model.joint_count // model.world_count
    parent = model.joint_parent.numpy()[:count]
    child = model.joint_child.numpy()[:count]
    types = model.joint_type.numpy()[:count]
    selected = (parent >= 0) & np.isin(types, [JointType.REVOLUTE, JointType.BALL, JointType.FIXED])
    if not selected.any():
        return None

    def anchors(indices, frames):
        poses = body_poses[:, indices]
        vectors = frames[:, :3]
        cross = 2.0 * np.cross(poses[..., 3:6], vectors)
        rotated = vectors + poses[..., 6:7] * cross + np.cross(poses[..., 3:6], cross)
        return poses[..., :3] + rotated

    parent_anchor = anchors(parent[selected], model.joint_X_p.numpy()[:count][selected])
    child_anchor = anchors(child[selected], model.joint_X_c.numpy()[:count][selected])
    return float(np.linalg.norm(child_anchor - parent_anchor, axis=-1).max())


def set_stationary_formation_camera(viewer, poses: np.ndarray, body_scale: float) -> None:
    """Tightly frame initial physical positions once from an oblique angle."""
    pitch, yaw = -25.0, 130.0
    pitch_rad, yaw_rad = np.deg2rad([pitch, yaw])
    forward = np.array([np.cos(pitch_rad) * np.cos(yaw_rad), np.cos(pitch_rad) * np.sin(yaw_rad), np.sin(pitch_rad)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    positions = poses[:, :3]
    center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    relative = positions - center
    depth = relative @ forward
    tangent_vertical = np.tan(np.deg2rad(viewer.camera.fov) * 0.5)
    tangent_horizontal = tangent_vertical * viewer.camera.width / viewer.camera.height
    distance = (
        max(
            np.max(np.abs(relative @ right) / tangent_horizontal - depth),
            np.max(np.abs(relative @ up) / tangent_vertical - depth),
        )
        + 0.75 * body_scale / tangent_vertical
    )
    viewer.set_camera(wp.vec3(center - distance * forward), pitch=pitch, yaw=yaw)


def main() -> None:
    """Load a trained policy and save behavior measurements from real DVI steps."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--forward-speed", type=float, default=0.5)
    parser.add_argument("--video-file", type=Path)
    parser.add_argument("--video-layout", choices=["single", "grid"], default="single")
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=30.0)
    custom, remaining = parser.parse_known_args()
    evaluator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if custom.steps <= 0 or custom.video_fps <= 0:
        raise ValueError("Steps and video frame rate must be positive")
    if min(custom.video_width, custom.video_height) <= 0 or custom.video_width % 2 or custom.video_height % 2:
        raise ValueError("Video dimensions must be positive and even")
    args = play_rsl_rl._parse_args(remaining)
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise ValueError("Evaluation requires an explicit local --checkpoint file")
    cfg, agent = resolve_task_config(args.task, args.agent, play_mode=True)
    apply_env_overrides(args, cfg)
    cfg.seed = args.seed if args.seed is not None else 42
    if hasattr(cfg, "observations"):
        cfg.observations.policy.enable_corruption = False
    if hasattr(cfg, "commands") and hasattr(cfg.commands, "base_velocity"):
        command = cfg.commands.base_velocity
        command.heading_command = False
        command.ranges.heading = None
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (custom.forward_speed, custom.forward_speed)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.debug_vis = False
    if hasattr(cfg, "events"):
        for name in ("push_robot", "base_external_force_torque"):
            if hasattr(cfg.events, name):
                setattr(cfg.events, name, None)
    if custom.video_file:
        cfg.sim.physics.load_visual_shapes = True
    custom.metrics.parent.mkdir(parents=True, exist_ok=True)

    with launch_simulation(cfg, args), contextlib.ExitStack() as cleanup:
        env = create_isaaclab_env(args.task, cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        agent = cli_args_rsl_rl.update_rsl_rl_cfg(agent, args)
        agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
        runner = create_rsl_rl_runner(env, agent)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        manager = env.unwrapped.sim.physics_manager
        model = manager.get_model()
        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        viewer = encoder = None
        encoded_path = custom.metrics.with_suffix(".recording.mp4")
        initial_positions = manager.get_state_0().body_q.numpy()[: robot.num_bodies, :3]
        camera_scale = float(np.clip(np.ptp(initial_positions, axis=0).max(), 0.25, 1.0))
        stride = max(1, round(1.0 / (custom.video_fps * dt)))
        if custom.video_file:
            import imageio_ffmpeg  # noqa: PLC0415
            import pyglet  # noqa: PLC0415

            pyglet.options["headless"] = True
            from newton.viewer import ViewerGL  # noqa: PLC0415

            custom.video_file.parent.mkdir(parents=True, exist_ok=True)
            viewer = ViewerGL(headless=True, width=custom.video_width, height=custom.video_height)
            cleanup.callback(viewer.close)
            viewer.set_model(model)
            viewer.set_visible_worlds(None if custom.video_layout == "grid" else [0])
            # Isaac Lab already places worlds at their physical environment origins.
            viewer.set_world_offsets((0.0, 0.0, 0.0))
            initial_roots = robot.data.root_link_pose_w.warp.numpy()
            if custom.video_layout == "grid":
                set_stationary_formation_camera(viewer, initial_roots, camera_scale)
            else:
                viewer.set_camera(
                    wp.vec3(initial_roots[0, :3] + camera_scale * np.array([2.5, -3.0, 1.8])),
                    pitch=-20.0,
                    yaw=130.0,
                )
            encoder = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{custom.video_width}x{custom.video_height}",
                    "-framerate",
                    str(1.0 / (stride * dt)),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "4",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(encoded_path),
                ],
                stdin=subprocess.PIPE,
            )
        poses, velocities, gravity, bodies = [], [], [], []
        done_count = failure_count = 0
        completed = False
        try:
            with torch.inference_mode():
                for step in range(custom.steps):
                    obs, _, dones, info = env.step(policy(obs))
                    policy.reset(dones)
                    for name, value in obs.items():
                        if not torch.isfinite(value).all():
                            raise RuntimeError(f"Non-finite observation {name} at step {step}")
                    state = manager.get_state_0()
                    if not np.isfinite(state.body_q.numpy()).all() or not np.isfinite(state.body_qd.numpy()).all():
                        raise RuntimeError(f"Non-finite native body state at step {step}")
                    pose = robot.data.root_link_pose_w.warp.numpy()
                    poses.append(pose)
                    velocities.append(robot.data.root_lin_vel_b.warp.numpy())
                    gravity.append(robot.data.projected_gravity_b.warp.numpy())
                    done_count += int(dones.sum().item())
                    timeouts = info.get("time_outs", torch.zeros_like(dones))
                    failure_count += int((dones.bool() & ~timeouts.bool()).sum().item())
                    if step % stride == 0:
                        bodies.append(state.body_q.numpy()[: robot.num_bodies])
                        if viewer is not None:
                            viewer.begin_frame(step * dt)
                            viewer.log_state(state)
                            viewer.end_frame()
                            encoder.stdin.write(viewer.get_frame().numpy().tobytes())
            completed = True
        finally:
            if encoder is not None:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError("Video encoder failed")
                if completed:
                    encoded_path.replace(custom.video_file)
        poses, velocities, gravity = map(np.asarray, (poses, velocities, gravity))
        bodies = np.asarray(bodies)
        second_half = velocities[custom.steps // 2 :]
        body_parent_f = getattr(manager.get_state_0(), "body_parent_f", None)
        metrics = {
            "task": args.task,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "physics_dt": cfg.sim.dt,
            "step_dt": dt,
            "steps": custom.steps,
            "num_envs": env.unwrapped.num_envs,
            "duration_seconds": custom.steps * dt,
            "finite_native_state": True,
            "body_parent_wrench_max_abs_final": (
                float(np.abs(body_parent_f.numpy()).max()) if body_parent_f is not None else None
            ),
            "forward_command": custom.forward_speed if hasattr(cfg, "commands") else None,
            "mean_forward_speed_last_half": float(second_half[..., 0].mean()),
            "median_forward_speed_last_half": float(np.median(second_half[..., 0])),
            "mean_lateral_speed_last_half": float(np.abs(second_half[..., 1]).mean()),
            "upright_fraction": float((gravity[..., 2] < -0.8).mean()),
            "median_root_height": float(np.median(poses[..., 2])),
            "episode_ends": done_count,
            "premature_terminations": failure_count,
            "max_joint_anchor_error_first_world_m": joint_anchor_error(bodies, model),
            **world_velocity_metrics(poses, velocities),
            "video": {
                "path": str(custom.video_file.resolve()),
                "width": custom.video_width,
                "height": custom.video_height,
                "fps": 1.0 / (stride * dt),
                "visible_worlds": env.unwrapped.num_envs if custom.video_layout == "grid" else 1,
                "layout": custom.video_layout,
                "root_centered_grid": False,
                "world_offsets": "zero",
                "camera_stationary": True,
                "camera_position_w": list(viewer.camera.pos),
                "camera_fov_degrees": viewer.camera.fov,
                "framing": "close_initial_formation",
                "all_worlds_stay_in_frame": False,
                "camera_pitch_degrees": -25.0 if custom.video_layout == "grid" else -20.0,
                "renderer_version": "stationary-world-oblique-v4" if custom.video_layout == "grid" else "single",
                "evaluator_sha256": evaluator_sha256,
            }
            if custom.video_file
            else None,
        }
        custom.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
        np.savez_compressed(
            custom.metrics.with_suffix(".npz"),
            root_pose=poses,
            root_velocity=velocities,
            projected_gravity=gravity,
            body_pose=bodies,
            dt=dt,
            body_labels=model.body_label[: robot.num_bodies],
            joint_parent=model.joint_parent.numpy()[: model.joint_count // model.world_count],
            joint_child=model.joint_child.numpy()[: model.joint_count // model.world_count],
            joint_X_p=model.joint_X_p.numpy()[: model.joint_count // model.world_count],
            joint_X_c=model.joint_X_c.numpy()[: model.joint_count // model.world_count],
            joint_type=model.joint_type.numpy()[: model.joint_count // model.world_count],
        )
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
