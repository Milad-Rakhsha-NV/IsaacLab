# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure full-articulation Go2 stance with static RFT reactions and no backing floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.go2.rft_env import Go2RFTEnv
from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTCurriculumEnvCfg


def run(args):
    """Keep fixed PD joint targets and record loaded foot motion, reactions and residuals."""
    profile = json.loads(args.profile.read_text())
    cfg = Go2RFTCurriculumEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.video:
        from isaaclab_visualizers.newton import NewtonGLBackendCfg

        from isaaclab.visualizers import VisualizerCfg

        visualizer = VisualizerCfg(backend=NewtonGLBackendCfg(enable_picking=False), headless=True)
        visualizer.window.size = (960, 640)
        visualizer.focal_length = 26
        cfg.sim.visualizer_cfgs = [visualizer]
    else:
        cfg.sim.visualizer_cfgs = []
    cfg.curriculum_enabled = False
    cfg.randomize_material = False
    cfg.randomize_commands = False
    cfg.command_speed = 0.0
    cfg.observation_noise = False
    cfg.bed_depth = None
    cfg.bed_depth_range = None
    cfg.scene.substrate = None
    cfg.surface_relative_height = False
    cfg.minimum_base_height = 0.01
    cfg.gait = None
    cfg.episode_length_s = args.seconds + 10
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
    for name, value in profile["solver"].items():
        setattr(cfg.sim.physics.solver_cfg, name, value)
    cfg.sim.physics.solver_cfg.surface_bottom = None
    cfg.sim.physics.solver_cfg.strength_multiplier = args.strength_multiplier
    cfg.sim.physics.solver_cfg.static_contact_iterations = args.iterations
    cfg.sim.physics.solver_cfg.static_contact_speed = args.transition_speed
    cfg.sim.physics.num_substeps = args.substeps
    rows, feet, forces, poses = [], [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg):
        env = Go2RFTEnv(cfg)
        writer = None
        try:
            env.reset(seed=42)
            if args.video:
                import imageio_ffmpeg
                from PIL import Image, ImageDraw, ImageFont

                font = ImageFont.load_default(size=18)
                writer = imageio_ffmpeg.write_frames(
                    str(args.output / "stance.mp4"),
                    (960, 640),
                    fps=50,
                    codec="libx264",
                    output_params=["-crf", "21", "-movflags", "+faststart"],
                )
                writer.send(None)
            foot_ids, _ = env.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
            native_ids = [row["body"] for row in env.rft.surface_inventory if row["body_label"].endswith("_foot")]
            native_ids = np.array(native_ids).reshape(args.num_envs, 4)
            impulse = env.rft.body_impulse.numpy().copy()
            start = time.monotonic()
            for frame in range(round(args.seconds / env.step_dt)):
                _, _, failed, _, _ = env.step(torch.zeros(args.num_envs, cfg.action_space, device=env.device))
                current = env.rft.body_impulse.numpy().copy()
                force = (current - impulse) / env.step_dt
                impulse = current
                q = env.robot.data.body_link_pos_w.torch[:, foot_ids].cpu().numpy()
                feet.append(q)
                poses.append(env.robot.data.body_link_pose_w.torch.cpu().numpy())
                forces.append(force[native_ids])
                residual = float(env.rft.static_contact_residual.numpy().max())
                rows.append(
                    [
                        (frame + 1) * env.step_dt,
                        float(env.robot.data.root_pos_w.torch[0, 2]),
                        float(env.robot.data.root_lin_vel_b.torch.norm(dim=-1).max()),
                        float(
                            np.linalg.norm(
                                env.robot.data.body_com_lin_vel_w.torch[:, foot_ids].cpu().numpy(), axis=-1
                            ).max()
                        ),
                        residual,
                        float(failed.sum()),
                        float(force[:, 2].sum() / args.num_envs),
                    ]
                )
                if frame % 50 == 0:
                    print(
                        json.dumps(
                            dict(time_s=rows[-1][0], root_z=rows[-1][1], foot_speed=rows[-1][3], residual=residual)
                        ),
                        flush=True,
                    )
                if writer is not None:
                    root = env.robot.data.root_pos_w.torch[0].cpu().numpy()
                    env.sim.set_camera_view(tuple(root + np.array([1.0, -1.3, 0.5])), tuple(root))
                    viz = env.sim.visualizers[0]
                    pose = env.sim.backend_manager.backend.state_0.body_q.numpy()
                    xyz, scalar = pose[:, 3:6], pose[:, 6:7]
                    com = env.rft.model.body_com.numpy()
                    centers = pose[:, :3] + com + 2 * np.cross(xyz, np.cross(xyz, com) + scalar * com)
                    active = np.linalg.norm(force[:, :3], axis=-1) > 1
                    viz._viewer.log_arrows(
                        "granular/body_forces",
                        wp.array(centers[active], dtype=wp.vec3),
                        wp.array(centers[active] + force[active, :3] / 200.0, dtype=wp.vec3),
                        wp.array(np.tile([0.1, 0.8, 0.2], (active.sum(), 1)), dtype=wp.vec3),
                    )
                    image = Image.fromarray(np.asarray(viz.render_rgb_array(), dtype=np.uint8)[..., :3])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 960, 94), fill=(15, 23, 42))
                    draw.text(
                        (15, 9),
                        "Go2 | Stationary-support check in RFT | Fixed PD joint targets",
                        fill="white",
                        font=font,
                    )
                    draw.text(
                        (15, 36),
                        f"No backing floor | 128 triangles/foot | {args.iterations} reaction iterations",
                        fill="white",
                        font=font,
                    )
                    draw.text(
                        (15, 63),
                        f"t={rows[-1][0]:.2f}s | RFT support={rows[-1][6]:.1f}N | Green arrows: body forces",
                        fill="white",
                        font=font,
                    )
                    writer.send(np.ascontiguousarray(np.asarray(image)))
                    if frame == 100:
                        image.save(args.output / "preview.jpg")
            summary = dict(
                settings=vars(args) | dict(profile=str(args.profile), output=str(args.output)),
                profile=profile,
                no_backing_floor=True,
                interpretation=(
                    "Fixed joint PD targets, not a learned policy. "
                    "Body height may settle through compliant joint motion."
                ),
                wall_seconds=time.monotonic() - start,
                trace_columns=[
                    "time_s",
                    "root_z_m",
                    "max_base_speed_m_s",
                    "max_foot_com_speed_m_s",
                    "max_projected_residual_m_s",
                    "failures",
                    "mean_vertical_RFT_force_N",
                ],
                newton_source=str(Path(newton.__file__).resolve()),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                final=rows[-1],
            )
            np.savez_compressed(args.output / "trace.npz", trace=rows, feet=feet, forces=forces, body_pose=poses)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            if writer is not None:
                writer.close()
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("scripts/benchmarks/configs/go2_rft_static_contact.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--transition_speed", type=float, default=0.2)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--root_height", type=float, default=0.32)
    parser.add_argument("--strength_multiplier", type=float, default=0.373)
    parser.add_argument("--video", action="store_true")
    run(parser.parse_args())
