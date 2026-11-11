# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish only the validated rigid control and the current finite-layer RFT study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from go2_granular_videos import render_videos
from monitor_go2_granular import load_study
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular")
STUDY = load_study()
CASE_ROOT = ROOT / STUDY.get("case_directory", "")
MAX_COMMAND = STUDY.get("max_forward_command", 0.5)
FAMILIES = {row[0]: row[2] for row in STUDY.get("runs", [])}
MEDIA_PREFIX = f"../{STUDY['case_directory']}/" if STUDY.get("case_directory") else "../"
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
    FAMILIES.get("rigid", "go2_rigid_gait_control"),
    "Rigid-ground training",
    "rigid-training.png",
    (("Metrics/min_foot_clearance", "Foot clearance from body centerline [m]"), ("Perf/total_fps", "Transitions/s")),
)
rft = training_figure(
    FAMILIES.get("rft", "go2_rft_knee_depth_curriculum"),
    "RFT training: shared gait criteria and reset-randomized 0–118 mm layers",
    "rft-training.png",
    (
        ("Curriculum/difficulty", "Material curriculum width [0–1]"),
        ("Material/depth_max", "Largest layer depth in reset batch [m]"),
    ),
)
case_names = (
    "rigid_max_forward",
    "rft_max_forward",
    "mpm_max_forward",
    "rigid_mpm_max_forward",
    "rigid_gait_stand_validation",
    "rigid_gait_walk_validation",
    "rigid_gait_maximum_validation",
    "rigid_gait_commands_123",
    "rigid_gait_commands_456",
    "rigid_gait_commands_789",
    "rft_knee_depth_stand_validation",
    "rft_knee_depth_walk_validation",
    "rft_knee_depth_maximum_validation",
    "rft_knee_depth_commands_123",
    "rft_knee_depth_commands_456",
    "rft_knee_depth_commands_789",
)
if STUDY:
    case_names = (*case_names[:4], *STUDY.get("validation_cases", []))
cases = []
for name in case_names:
    path = CASE_ROOT / name / "summary.json"
    if path.exists():
        cases.append(dict(case=name, **json.loads(path.read_text())))

gait = {}
for case in cases:
    if case["case"] not in ("rigid_max_forward", "rft_max_forward", "mpm_max_forward", "rigid_mpm_max_forward"):
        continue
    with np.load(CASE_ROOT / case["case"] / "rollout.npz") as data:
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

forward_labels = {
    "rigid_max_forward": "Rigid policy / rigid floor",
    "rft_max_forward": "RFT policy / RFT sand",
    "mpm_max_forward": "RFT policy / MPM sand",
    "rigid_mpm_max_forward": "Rigid policy / MPM sand",
}
ready_cases = [
    case
    for case in cases
    if not STUDY or STUDY.get("videos_ready") or case["case"] in STUDY.get("published_videos", [])
]
forward_cases = [case for case in ready_cases if case["case"] in forward_labels]
forward_figure = ""
if forward_cases:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for case in forward_cases:
        with np.load(CASE_ROOT / case["case"] / "rollout.npz") as data:
            state = data["state"][:, 0]
            for ax, column in zip(axes, (4, 8, 7), strict=True):
                ax.plot(state[:, 0], state[:, column], label=forward_labels[case["case"]])
    axes[0].axhline(MAX_COMMAND, color="grey", linestyle="--", label="Forward command")
    axes[0].legend(fontsize=7)
    height_limit = axes[2].axhline(0.16, color="grey", linestyle=":", label="Body-height failure threshold")
    axes[2].legend(handles=[height_limit], fontsize=7, loc="upper right")
    for ax, label in zip(
        axes, ("Forward speed [m/s]", "Planar tracking error [m/s]", "Height above initial surface [m]"), strict=True
    ):
        ax.set(xlabel="Simulation time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    save(fig, "forward-comparison.png")
    forward_figure = (
        "<h2>Maximum-forward comparison</h2>"
        f"<p>All clips command {MAX_COMMAND:.2f} m/s forward with zero lateral and yaw commands. "
        "The curves include the initial acceleration and any time after a fall.</p>"
        "<img src='forward-comparison.png' alt='Speed, tracking error and body height for the four recordings'>"
    )

rigid_status_path = ROOT / STUDY.get("status_file", "training/evaluation_status.json")
rft_status_path = ROOT / STUDY.get("status_file", "training/rft-evaluation-status.json")
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
    if (rft if name == "rft-training.png" else rigid) is not None
)
new_status = rft_status.get("stage", "New RFT training is being prepared; final recordings follow training.")
previous_note = "<p><a href='../previous_0p5_report/'>Previous 0.5 m/s recordings and results</a></p>" if STUDY else ""
material_comparison = (
    f"<h2>Material comparison at {MAX_COMMAND:.2f} m/s</h2>"
    "<p>Both new policies are frozen and tested for 3 s at each material setting, using seed 123. "
    "Lower: 1,400 kg/m³, friction 0.3, surface/internal ratio 0.6, depth 60 mm. "
    "Nominal: 1,600, 0.4, 1.0, 118 mm. Higher: 1,800, 0.5, 0.8, 100 mm. "
    "The dotted line evaluates the RFT policy in RFT; bars evaluate both policies in MPM. "
    "These are exploratory short trials. MPM numerical validation remains incomplete.</p>"
    "<img src='policy-and-support.png' alt='New-policy material tracking and granular versus floor support'>"
    f"<p><a href='../{STUDY.get('case_directory', '')}/materials/materials.html'>MPM trial records</a></p>"
    if STUDY.get("material_comparison_ready")
    else ""
)
recording_text = (
    f"Four 10 s recordings {'command' if STUDY.get('videos_ready') else 'will command'} {MAX_COMMAND:.2f} m/s forward. "
    "The two policies share a 4,096-environment, 1,500-iteration training budget."
    if STUDY
    else "Four 10 s recordings use the maximum forward command, 0.50 m/s."
)
gate_labels = {
    "no_falls": "no falls",
    "foot_placement": "foot placement",
    "tracking": "velocity tracking",
    "standing": "standing drift",
    "contact_slip": "loaded-foot slip",
    "capacity": "solver capacity",
    "every_foot_steps": "stepping with every foot",
    "full_material_curriculum": "full material curriculum",
}
failed_gates = [
    gate_labels.get(key, key) for key, passed in rft_status.get("rft_validation_gates", {}).items() if not passed
]
validation_note = (
    "<p><strong>RFT validation remains incomplete.</strong> Failed acceptance checks: "
    + ", ".join(failed_gates)
    + ". The MPM clips are exploratory transfer tests; they do not certify convergence or material equivalence.</p>"
    if failed_gates
    else ""
)
rigid_failed = [
    gate_labels.get(key, key) for key, passed in rigid_status.get("rigid_validation_gates", {}).items() if not passed
]
if rigid_failed:
    validation_note += (
        "<p><strong>Rigid validation remains incomplete.</strong> Failed checks: " + ", ".join(rigid_failed) + ".</p>"
    )
material_analysis_path = ROOT / "investigation/analysis.json"
material_analysis = json.loads(material_analysis_path.read_text()) if material_analysis_path.exists() else None
material_note = (
    "<p class='status'><strong>Material validation: MPM has not passed the force-convergence gate.</strong> "
    f"The {material_analysis['case_count']}-case investigation also finds that matching density, friction and depth "
    "does not match the complete RFT/MPM force response. "
    "<a href='../investigation/analysis.html'>See convergence plots, force calibration "
    "and frozen-policy comparisons</a>.</p>"
    if material_analysis and not material_analysis["mpm_reference_accepted"]
    else ""
)
if STUDY and material_note:
    material_note += (
        "<p>The linked material investigation retains its earlier 0.5 m/s policy comparisons. "
        "Its standalone force-convergence checks do not depend on the new training command range.</p>"
    )
(OUT / "index.html").write_text(
    """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2: maximum-forward policy comparison</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
video,img{width:100%;border-radius:6px}.video-grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
article,figure{background:white;padding:20px;margin:0 0 24px;border-radius:10px}figure{margin:20px 0}
h1,h2,h3{line-height:1.2}h3{margin-top:0}details{margin:12px 0}summary,a{color:#1d4ed8}
code{overflow-wrap:anywhere}.status{padding:16px;background:#dbeafe}</style>
<h1>Go2: maximum-forward policy comparison</h1>
<p>The rigid and RFT runs use the same robot, self-collision, posture and foot-placement criteria.
Their standing, tracking, slip and stepping acceptance checks are reported separately.</p>
<p>The Go2 knee-to-sole length is approximately 236 mm. Each resetting environment samples
<strong>0–118 mm of sand above a rigid floor</strong>, capped at half that length from the first reset.
Density, friction and strength have a performance-gated curriculum;
depth and commands use their full ranges throughout.</p>
<details><summary>Lower-leg measurement and depth limit</summary>
<img src="../measurements/foot-lengths.png" alt="Go2 knee-to-sole length is 235.9 mm; half is 118 mm"
style="max-width:520px;display:block;margin:auto"></details>
<p class="status">"""
    + new_status
    + "</p>"
    + previous_note
    + validation_note
    + material_note
    + (
        "<p><strong>Material-model investigation:</strong> "
        "<a href='../investigation/analysis.html'>Convergence checks, force measurements and material matching</a>.</p>"
        if material_analysis is None and (ROOT / "investigation/status.json").exists()
        else ""
    )
    + """
<p><a href="../live/">Live status and learning curves</a></p>
<h2>Current policy recordings</h2>
<p>"""
    + recording_text
    + """ There are two frozen policies: one trained on a rigid floor and one trained in randomized RFT sand.
Neither was trained in MPM.
Each policy is first shown in its training simulator and then in the same MPM sand bed, without fine-tuning.
All sand beds start at 118 mm depth with density 1,600 kg/m³ and nominal friction 0.4.</p>"""
    + render_videos(CASE_ROOT, ready_cases, gait, url_prefix=MEDIA_PREFIX)
    + forward_figure
    + material_comparison
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
