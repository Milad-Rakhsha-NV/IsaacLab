# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record real G1 policy rollouts using an immutable checkpoint and headless Newton GL."""

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from projects.supermimic.motions import MotionLibrary
from projects.supermimic.networks import Actor, goal


def annotate(pixels, name, iteration, seconds, first_failure, reward):
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    draw.rectangle((0, 0, image.width, 76), fill=(17, 24, 39))
    draw.text((20, 10), f"SuperMimic / IsaacLab  |  {name}  |  checkpoint {iteration:,}", font=font, fill="white")
    status = "tracking" if first_failure is None else f"tracking threshold exceeded at {first_failure:.2f}s"
    color = (134, 239, 172) if first_failure is None else (253, 186, 116)
    draw.text(
        (20, 42),
        f"Actual physics rollout  |  {seconds:5.2f}s  |  {status}  |  step reward {reward:.3f}",
        font=font,
        fill=color,
    )
    return np.asarray(image)


def gallery(out, results, iteration, total):
    cards = []
    for r in results:
        cards.append(f"""<article><h2>{html.escape(r["motion"])}</h2>
<video controls playsinline preload="metadata" poster="{r["poster"]}">
<source src="{r["video"]}" type="video/mp4"></video>
<p>{r["duration_seconds"]:.1f}s at real-time playback · normalized return {r["normalized_return"]:.3f}
· first tracking failure: {r["first_failure_seconds"] if r["first_failure_seconds"] is not None else "none"}
</p></article>""")
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SuperMimic rollouts</title>
<style>body{{font:16px system-ui;background:#f1f5f9;color:#0f172a;margin:32px auto;max-width:1200px;padding:0 20px}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:24px}}
article{{background:white;padding:16px;border-radius:12px}}video{{width:100%}}h2{{font-size:19px}}p{{color:#475569}}</style>
<h1>SuperMimic: current policy rollouts</h1><p>{len(results)} / {total} recordings available.
Checkpoint iteration {iteration:,}. Actual Newton/MJWarp physics;
no observation/action/reset noise, no resets after tracking failures. Training continues separately.
These recordings are longer than the five-second periodic evaluation when requested.</p>
<main>{"".join(cards)}</main></html>
""")
    rows = [
        "# Current policy videos",
        "",
        f"Checkpoint iteration: **{iteration:,}**.",
        "",
        f"Recordings available: **{len(results)} / {total}**.",
        "",
        "Actual Newton/MJWarp policy rollouts at real-time playback. No exploration or sensor noise.",
        "Tracking failures are marked on screen; the robot is not reset after a failure.",
        "",
        "[Open the video gallery](index.html)",
        "",
    ]
    for r in results:
        rows += [
            f"## {r['motion']}",
            "",
            f"[![Video preview]({r['poster']})]({r['video']})",
            "",
            f"[Watch MP4]({r['video']}) · {r['duration_seconds']:.1f}s · "
            f"normalized return {r['normalized_return']:.3f}",
            "",
        ]
    (out / "README.md").write_text("\n".join(rows))


@torch.no_grad()
def record(env, actor, name, out, seconds, iteration):
    index = env.library.names.index(name)
    ids = torch.zeros(1, device=env.device, dtype=torch.long)
    env.reset(ids, torch.full_like(ids, index), torch.zeros_like(ids), noise=False)
    frames = min(round(seconds * 50), int(env.library.lengths[index]))
    writer = imageio_ffmpeg.write_frames(
        str(out / f"{name}.mp4"),
        (1280, 720),
        fps=50,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)
    first_failure, reward_sum, reward, center = None, 0.0, 0.0, None
    try:
        for frame in range(frames):
            if frame:
                target = goal(env.read(), env.library.future(env.clip, env.frame))
                action = actor(env.history, target)
                transition = env.step(action, noise=False)
                reward = float(transition["reward"][0])
                reward_sum += reward
                if first_failure is None and bool(transition["failure"][0]):
                    first_failure = frame / 50
            position = env.robot.data.root_link_pos_w.torch[0].cpu().numpy()
            desired = np.array([position[0], position[1], max(0.7, position[2] + 0.1)])
            center = desired if center is None else 0.15 * desired + 0.85 * center
            env.sim.set_camera_view(tuple(center + np.array([1.8, -2.7, 1.15])), tuple(center))
            pixels = env.sim.visualizers[0].render_rgb_array()
            pixels = np.asarray(pixels, dtype=np.uint8)[..., :3]
            if pixels.shape != (720, 1280, 3) or pixels.std() < 1:
                raise RuntimeError(f"Invalid rendered frame: {pixels.shape}")
            pixels = annotate(pixels, name, iteration, frame / 50, first_failure, reward)
            writer.send(np.ascontiguousarray(pixels))
            if frame == min(500, frames - 1):
                Image.fromarray(pixels).save(out / f"{name}.jpg", quality=92)
    finally:
        writer.close()
    result = {
        "motion": name,
        "video": f"{name}.mp4",
        "poster": f"{name}.jpg",
        "frames": frames,
        "fps": 50,
        "duration_seconds": frames / 50,
        "first_failure_seconds": first_failure,
        "normalized_return": reward_sum / (max(1, frames - 1) * 0.1),
    }
    print("VIDEO_RESULT", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--clips", nargs="+", default=["walk1_subject1", "dance2_subject1", "jumps1_subject2", "fallAndGetUp1_subject1"]
    )
    selection.add_argument("--all-clips", action="store_true", help="Record every motion in the checkpoint's library")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Duration must be positive")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "checkpoint.pt"
    if args.checkpoint.resolve() != checkpoint_path.resolve():
        shutil.copyfile(args.checkpoint, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    iteration = saved["iteration"]
    library = MotionLibrary(args.motions, "cuda:0")
    if library.sha256 != saved["motion_sha256"] or library.names != saved["clips"]:
        raise ValueError("Checkpoint and motion-library provenance differ")
    if args.all_clips:
        args.clips = library.names
    if any(name not in library.names for name in args.clips):
        raise ValueError("Unknown requested motion")
    hidden = (256, 128, 64) if saved["config"]["small_networks"] else (2048, 1024, 512, 256)
    actor = Actor(hidden).to("cuda:0")
    actor.load_state_dict(saved["actor"])
    actor.eval()
    metadata = {
        "checkpoint_iteration": iteration,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "motion_sha256": library.sha256,
        "physics": "Newton/MJWarp",
        "action_noise": False,
        "observation_noise": False,
        "reset_noise": False,
        "reset_after_failure": False,
        "requested_duration_seconds": args.seconds,
        "requested_clips": args.clips,
    }
    del saved
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.app import launch_simulation
    from isaaclab.visualizers import VisualizerCfg
    from isaaclab.visualizers.visualizer_cfg import WindowCfg

    from projects.supermimic.environment import TrackingEnv, simulation_cfg

    cfg = simulation_cfg("cuda:0")
    cfg.visualizer_cfgs = [
        VisualizerCfg(
            backend=NewtonGLBackendCfg(enable_picking=False, enable_sky=False),
            headless=True,
            window=WindowCfg(size=(1280, 720)),
            focal_length=18.0,
            background_color=(0.92, 0.94, 0.97),
            enable_markers=False,
        )
    ]
    with launch_simulation(cfg, {"headless": True}):
        env = TrackingEnv(cfg, library, 1)
        try:
            # Rendering-only colors make the robot legible against the pale floor.
            names = tuple(env.robot.body_names)
            colors = torch.tensor([0.06, 0.08, 0.12], device=env.device).expand(1, len(names), 3).clone()
            for body, name in enumerate(names):
                if name in ("pelvis", "torso_link"):
                    colors[0, body] = torch.tensor([0.025, 0.14, 0.38], device=env.device)
            color_writer = env.sim.backend_manager.create_visual_shape_color_writer(env.robot, names)
            color_writer(colors, torch.zeros(1, dtype=torch.int32, device=env.device))
            results = []
            for name in args.clips:
                results.append(record(env, actor, name, args.out, args.seconds, iteration))
                metadata["videos"] = results
                temporary = args.out / "metadata.tmp"
                temporary.write_text(json.dumps(metadata, indent=2) + "\n")
                temporary.replace(args.out / "metadata.json")
                gallery(args.out, results, iteration, len(args.clips))
        finally:
            env.close()


if __name__ == "__main__":
    main()
