# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish only the validated rigid control and the current shallow-layer RFT study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from go2_granular_videos import render_videos
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular")
OUT = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def training_figure(family: str, title: str, name: str, extra_tags: tuple) -> dict | None:
    runs = sorted(Path("logs/rsl_rl", family).glob("20*"))
    if not runs:
        return None
    events = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0}).Reload()
    tags = events.Tags()["scalars"]
    if "Perf/total_fps" not in tags:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    metrics = (
        ("Train/mean_reward", "Training return"),
        ("Train/mean_episode_length", "Episode length [control steps; maximum 500]"),
        ("Metrics/velocity_error", "Planar tracking error [m/s]"),
        *extra_tags,
        ("Metrics/gait_failure_fraction", "Foot-placement failure fraction"),
    )
    for ax, (tag, label) in zip(axes.flat, metrics, strict=True):
        if tag in tags:
            values = events.Scalars(tag)
            x = np.array([v.step for v in values])
            y = np.array([v.value for v in values])
            ax.plot(x, y, alpha=0.2)
            window = min(20, len(y))
            ax.plot(x[window - 1 :], np.convolve(y, np.ones(window) / window, mode="valid"))
        ax.set(title=label, xlabel="PPO iteration")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    save(fig, name)
    last = events.Scalars("Perf/total_fps")[-1]
    return dict(directory=str(runs[-1]), completed_iterations=last.step + 1, transitions=(last.step + 1) * 4096 * 24)


rigid = training_figure(
    "go2_rigid_gait_control",
    "Validated rigid-ground control",
    "rigid-training.png",
    (("Metrics/min_foot_clearance", "Foot clearance from body centerline [m]"), ("Perf/total_fps", "Transitions/s")),
)
rft = training_figure(
    "go2_rft_shallow_curriculum",
    "New RFT training: shared gait criteria and reset-randomized shallow layers",
    "rft-training.png",
    (
        ("Curriculum/difficulty", "Material curriculum width [0–1]"),
        ("Material/depth_max", "Largest layer depth in reset batch [m]"),
    ),
)
case_names = (
    "rigid_gait_control",
    "rigid_gait_stand_validation",
    "rigid_gait_walk_validation",
    "rigid_gait_maximum_validation",
    "rigid_gait_commands_123",
    "rigid_gait_commands_456",
    "rigid_gait_commands_789",
    "rft_shallow_control",
    "rft_shallow_stand_validation",
    "rft_shallow_walk_validation",
    "rft_shallow_maximum_validation",
    "rft_shallow_commands_123",
    "rft_shallow_commands_456",
    "rft_shallow_commands_789",
    "rft_shallow_mpm",
)
cases = []
for name in case_names:
    path = ROOT / name / "summary.json"
    if path.exists():
        cases.append(dict(case=name, **json.loads(path.read_text())))

gait = {}
for case in cases:
    if case["case"] not in ("rigid_gait_control", "rft_shallow_control"):
        continue
    with np.load(ROOT / case["case"] / "rollout.npz") as data:
        poses = data["body_pose"]
        labels = case["body_labels"]
        base = poses[:, next(i for i, label in enumerate(labels) if label.endswith("/base"))]
        feet = []
        for leg in ("RL", "RR"):
            foot = poses[:, next(i for i, label in enumerate(labels) if label.endswith(f"/{leg}_foot"))]
            offset = foot[:, :3] - base[:, :3]
            xyz, scalar = -base[:, 3:6], base[:, 6:7]
            feet.append((offset + 2 * np.cross(xyz, np.cross(xyz, offset) + scalar * offset))[:, 1])
        separation = feet[0] - feet[1]
        crossed = separation < 0
        gait[case["case"]] = dict(
            reversed_rear_foot_order_fraction=float(crossed.mean()),
            first_rear_foot_order_reversal_s=float(data["state"][np.flatnonzero(crossed)[0], 0, 0])
            if crossed.any()
            else None,
            mean_rear_foot_separation_m=float(separation.mean()),
        )
(OUT / "gait_diagnostics.json").write_text(json.dumps(gait, indent=2))

rigid_status_path = ROOT / "training/evaluation_status.json"
rft_status_path = ROOT / "training/rft-evaluation-status.json"
rigid_status = json.loads(rigid_status_path.read_text()) if rigid_status_path.exists() else {}
rft_status = json.loads(rft_status_path.read_text()) if rft_status_path.exists() else {}
(OUT / "results.json").write_text(
    json.dumps(
        dict(
            rigid_training=rigid,
            rft_training=rft,
            cases=cases,
            rigid_validation=rigid_status.get("rigid_validation_gates"),
            rft_validation=rft_status.get("rft_validation_gates"),
        ),
        indent=2,
    )
)
figures = "".join(
    f"<figure><h3>{title}</h3><img src='{name}' alt='{title}' loading='lazy'></figure>"
    for name, title in (
        ("rft-training.png", "New RFT learning curves"),
        ("rigid-training.png", "Rigid-control learning curves"),
    )
    if (OUT / name).exists()
)
new_status = rft_status.get("stage", "New RFT training is being prepared; final recordings follow training.")
(OUT / "index.html").write_text(
    """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2: validated rigid control and new RFT training</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
video,img{width:100%;border-radius:6px}.video-grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
article,figure{background:white;padding:20px;margin:0 0 24px;border-radius:10px}figure{margin:20px 0}
h1,h2,h3{line-height:1.2}h3{margin-top:0}details{margin:12px 0}summary,a{color:#1d4ed8}
code{overflow-wrap:anywhere}.status{padding:16px;background:#dbeafe}</style>
<h1>Go2: shallow-layer RFT training</h1>
<p>The rigid-ground control passed standing, walking, maximum-speed, steering and held-out command tests.
The new RFT run uses the same robot, self-collision, posture and foot-placement criteria.</p>
<p>The Go2 foot collider is 44 mm across. Training starts with <strong>0–22 mm of sand above a rigid floor</strong>.
The depth range widens toward 0–100 mm after performance gates pass.
Only material properties and layer depth have a curriculum; commands use the full range from the first reset.</p>
<p class="status">"""
    + new_status
    + """</p>
<p><a href="../live/">Live status and learning curves</a></p>
<h2>Current policy recordings</h2>
<p>The rigid-control video includes standing, 0.30 m/s walking, the 0.50 m/s training maximum, steering and stopping.
New RFT and MPM recordings will appear only for the new policy after training and evaluation.</p>"""
    + render_videos(ROOT, cases, gait)
    + "<h2>Training curves</h2>"
    + figures
    + """<details><summary>Material randomization and contact criteria</summary>
<p>Each resetting environment samples its own density (1,400–1,800 kg/m³), internal friction (0.3–0.5),
surface/internal friction ratio (0.6–1.0), strength multiplier (0.8–1.2), layer depth and commands.
The material distribution widens after sustained episode completion and tracking performance.</p>
<p>Posture, joint limits, motion smoothness, foot slip, loaded/unloaded foot timing, non-foot contacts and foot
placement use the rigid-control reward settings. RFT contact cues combine applied granular body impulses
with rigid-floor collision contacts; granular loads are not discrete collision contacts.</p>
<p>The RFT force law includes an experimental near-rest bearing term. Finite-layer clipping does not model
confinement, compaction or an evolving free surface. MPM is reserved for inference after training.</p>
<p><a href="results.json">Current numerical records</a> ·
<a href="gait_diagnostics.json">Current gait measurements</a></p>
</details>"""
)
print(f"Built {OUT / 'index.html'}")
