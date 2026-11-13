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
from go2_granular_videos import describe_progress, render_videos, video_cases
from monitor_go2_granular import load_study
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular")
STUDY = load_study()
CASE_ROOT = ROOT / STUDY.get("case_directory", "")
MAX_COMMAND = STUDY.get("max_forward_command", 0.5)
SPEEDS = tuple(STUDY.get("video_speeds", [MAX_COMMAND]))
VARIANTS = video_cases(SPEEDS)
VIDEO_NAMES = {name for name, _, _ in VARIANTS}
FAMILIES = {row[0]: row[2] for row in STUDY.get("runs", [])}
MEDIA_PREFIX = f"../{STUDY['case_directory']}/" if STUDY.get("case_directory") else "../"
OUT = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def training_figure(families: dict[str, str]) -> dict:
    """Overlay both training runs on the same metric axes."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    metrics = (
        ("Train/mean_reward", "Training return"),
        ("Train/mean_episode_length", "Episode length [control steps; maximum 500]"),
        ("Metrics/velocity_error", "Planar tracking error [m/s]"),
        ("Metrics/min_foot_clearance", "Foot clearance from body centerline [m]"),
        ("Perf/total_fps", "Transitions/s"),
        ("Metrics/gait_failure_fraction", "Foot-placement failure fraction"),
    )
    completed = {}
    for policy, label, color in (("rigid", "Rigid", "tab:blue"), ("rft", "RFT", "tab:orange")):
        runs = sorted(Path("logs/rsl_rl", families[policy]).glob("20*"))
        if not runs:
            continue
        events = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0}).Reload()
        tags = events.Tags()["scalars"]
        if "Perf/total_fps" not in tags:
            continue
        for ax, (tag, _) in zip(axes.flat, metrics, strict=True):
            if tag not in tags:
                continue
            values = events.Scalars(tag)
            x = np.array([v.step for v in values])
            y = np.array([v.value for v in values])
            ax.plot(x, y, color=color, alpha=0.15)
            window = min(20, len(y))
            ax.plot(x[window - 1 :], np.convolve(y, np.ones(window) / window, mode="valid"), color=color, label=label)
        last = events.Scalars("Perf/total_fps")[-1]
        completed[policy] = dict(
            directory=str(runs[-1]), completed_iterations=last.step + 1, transitions=(last.step + 1) * 4096 * 24
        )
    if not completed:
        plt.close(fig)
        return completed
    for ax, (_, label) in zip(axes.flat, metrics, strict=True):
        ax.set(title=label, xlabel="PPO iteration")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Rigid and RFT training")
    save(fig, "training-comparison.png")
    return completed


training = training_figure(
    dict(
        rigid=FAMILIES.get("rigid", "go2_rigid_gait_control"),
        rft=FAMILIES.get("rft", "go2_rft_knee_depth_curriculum"),
    )
)
rigid = training.get("rigid")
rft = training.get("rft")
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
    case_names = (*[name for name, _, _ in VARIANTS], *STUDY.get("validation_cases", []))
cases = []
for name in case_names:
    path = CASE_ROOT / name / "summary.json"
    if path.exists():
        cases.append(dict(case=name, **json.loads(path.read_text())))

gait = {}
for case in cases:
    if case["case"] not in VIDEO_NAMES:
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
forward_cases = [case for case in ready_cases if case["case"] in VIDEO_NAMES]
mpm_domains = {}
for case in forward_cases:
    if case["field"] == "mpm":
        mpm_domains[case["commands"][0][0]] = (case["bed_length"], case["bed_width"])
domain_description = "; ".join(
    f"{speed:.2f} m/s: {length:g} × {width:g} m" for speed, (length, width) in sorted(mpm_domains.items(), reverse=True)
)
forward_figures = {}
for speed in sorted(SPEEDS):
    names = {name: base for name, base, command in VARIANTS if command == speed}
    selected = [case for case in forward_cases if case["case"] in names]
    if not selected:
        continue
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for case in selected:
        with np.load(CASE_ROOT / case["case"] / "rollout.npz") as data:
            state = data["state"][:, 0]
            for ax, column in zip(axes, (4, 8, 7), strict=True):
                ax.plot(state[:, 0], state[:, column], label=forward_labels[names[case["case"]]])
    axes[0].axhline(speed, color="grey", linestyle="--", label="Forward command")
    axes[0].legend(fontsize=7)
    height_limit = axes[2].axhline(0.16, color="grey", linestyle=":", label="Body-height failure threshold")
    axes[2].legend(handles=[height_limit], fontsize=7, loc="upper right")
    for ax, label in zip(
        axes, ("Forward speed [m/s]", "Planar tracking error [m/s]", "Height above initial surface [m]"), strict=True
    ):
        ax.set(xlabel="Simulation time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Same frozen checkpoints · {speed:.2f} m/s forward command")
    filename = f"forward-comparison-{speed:g}.png"
    save(fig, filename)
    forward_figures[speed] = (
        "<p>Velocities and commands use the robot frame. The curves include initial acceleration "
        "and time after a fall.</p>"
        f"<img src='{filename}' alt='Speed, tracking error and body height at {speed:g} m/s'>"
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
figures = (
    "<p>Rigid and RFT share each metric axis. Solid lines show 20-iteration moving averages; "
    "faint lines show the raw measurements.</p>"
    "<figure><img src='training-comparison.png' alt='Rigid and RFT learning curves on shared axes' "
    "loading='lazy'></figure>"
    if training
    else ""
)
new_status = describe_progress(
    rft_status.get("stage", "New RFT training is being prepared; final recordings follow training."), SPEEDS
)
previous_note = "<p><a href='../previous_0p5_report/'>Previous 0.5 m/s recordings and results</a></p>" if STUDY else ""
material_comparison = ""
if STUDY.get("half_speed_material_comparison_ready"):
    material_comparison = (
        "<h4>Material sensitivity</h4>"
        "<p>Both policies are frozen and tested for 3 s at each material setting, using seed 123. "
        "Lower: 1,400 kg/m³, friction 0.3, surface/internal ratio 0.6, depth 60 mm. "
        "Nominal: 1,600, 0.4, 1.0, 118 mm. Higher: 1,800, 0.5, 0.8, 100 mm. "
        "The dotted line evaluates the RFT policy in RFT; bars evaluate both policies in MPM. "
        "These are exploratory short trials. MPM numerical validation remains incomplete.</p>"
        "<img src='../max1/materials_cmd_0p5/policy-and-support.png' "
        "alt='Tracking and whole-robot support at 0.5 m/s'>"
        "<p>Support forces include all rigid robot contacts and time after a failure threshold. "
        "Floor support alone does not imply successful walking.</p>"
    )
mpm_recheck = ""
if STUDY.get("mpm_recheck_ready"):
    recheck = json.loads((CASE_ROOT / "mpm-recheck/results.json").read_text())
    recheck_rows = []
    for row in recheck["cases"]:
        result = row["summary"]
        first = result["first_failure_seconds"]
        failure = "None in 2 s" if first is None else f"{first:.2f} s"
        recheck_rows.append(
            f"<tr><td>{row['substeps']}</td><td>{row['coupling_passes']}</td>"
            f"<td>{result.get('proxy_relaxation', 0.1):g}</td>"
            f"<td>{result['mean_velocity_error']:.3f}</td><td>{failure}</td></tr>"
        )
    mpm_recheck = (
        "<h4 id='mpm-numerics'>MPM startup check</h4>"
        "<p>The frozen RFT policy is tested in the same prepared 5 × 1.6 × 0.118 m bed for 2 s. "
        "Each row changes substeps, coupling passes or feedback relaxation. "
        "The physics frame is 5 ms; 4, 8 and 16 substeps correspond to 1.25, 0.625 and 0.3125 ms. "
        "Grid, material, preparation, checkpoint and seed stay fixed. "
        "This is a numerical sensitivity check, not a completed MPM convergence study.</p>"
        "<p><strong>The immediate collapse depends on coupling numerics.</strong> "
        "Smaller time steps did not remove it; four coupling passes or full feedback avoided the threshold "
        "during these short tests. This does not establish 10 s stability or MPM convergence. "
        "The main videos retain the original one-pass, 0.1-feedback configuration for both policies and commands. "
        "Their 1.0 m/s falls cannot be attributed solely to policy or RFT model error.</p>"
        "<table><tr><th>Substeps</th><th>Coupling passes</th><th>Feedback factor</th><th>Tracking error [m/s]</th>"
        "<th>First failure threshold</th></tr>" + "".join(recheck_rows) + "</table>"
        "<img src='mpm-startup-recheck.png' alt='Startup height, velocity, tilt and support sensitivity'>"
    )
comparisons = "<h2>Comparisons</h2>" + "".join(
    f"<section><h3>Comparison at {speed:.2f} m/s</h3>"
    + forward_figures[speed]
    + (material_comparison if speed == 0.5 else "")
    + (mpm_recheck if speed == 1.0 else "")
    + "</section>"
    for speed in sorted(forward_figures)
)
recording_text = (
    f"{len(forward_cases)} of {len(VARIANTS)} videos available. "
    f"Each terrain panel compares the same frozen policy at {' and '.join(f'{speed:.2f}' for speed in SPEEDS)} "
    "m/s forward for 10 s. "
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
<title>Go2: policy comparison at two forward speeds</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
video,img{width:100%;border-radius:6px}.video-grid{display:grid;
grid-template-columns:1fr;gap:20px}.paired-videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
article,figure{background:white;padding:20px;margin:0 0 24px;border-radius:10px}figure{margin:20px 0}
h1,h2,h3{line-height:1.2}h3{margin-top:0}details{margin:12px 0}summary,a{color:#1d4ed8}
code{overflow-wrap:anywhere}.status{padding:16px;background:#dbeafe}</style>
<h1>Go2: policy comparison at two forward speeds</h1>
<p class="status"><a href="../../go2-contact-refinement/">Go2 contact-resolution follow-up</a>:
128 triangles per spherical foot, selective lower-calf refinement, accuracy, cost and static-support checks.
The policy videos below retain their original solver and coarse training surfaces.
<a href="../../go2-contact-refinement/low-speed/static-contact/">Latest articulated RFT checks</a>:
stationary support video, MPM loading comparisons, and 4096-world cost. Material matching remains open;
no replacement policy has been trained.</p>
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
    + (
        "<p class='status'><strong>The early 1.0 m/s MPM collapse is sensitive to coupling settings.</strong> "
        "Additional coupling passes or full feedback avoided the failure threshold in short tests. "
        "The main MPM videos retain the original numerical settings. "
        "<a href='#mpm-numerics'>See the controlled comparison and its limits</a>.</p>"
        if STUDY.get("mpm_recheck_ready")
        else ""
    )
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
    + (
        f"<p>MPM bed dimensions are {domain_description}. "
        "Both policies share the same bed dimensions at each command. "
        "An earlier narrow-bed 0.5 m/s clip reached the bed edge before falling and is excluded. "
        "Bed-boundary and numerical-capacity checks are required for publication. "
        "Failure times remain in the numerical results and captions; they are omitted from video banners.</p>"
        if STUDY.get("video_bed_width") == 3.2
        else ""
    )
    + (
        "<p>The 1.0 m/s videos were regenerated from their saved trajectories with new banners. "
        "The 0.5 m/s videos use new rollouts of the same checkpoints. "
        "The differing bed widths limit a strict comparison across commands.</p>"
        if STUDY.get("reused_maximum_command_trajectories")
        else ""
    )
    + render_videos(CASE_ROOT, ready_cases, gait, url_prefix=MEDIA_PREFIX, speeds=SPEEDS)
    + comparisons
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
