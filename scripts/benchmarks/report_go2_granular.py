# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build graphs and a local video gallery from actual Go2 training/inference artifacts."""

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
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def read_case(name):
    path = ROOT / name
    return json.loads((path / "summary.json").read_text()), np.load(path / "rollout.npz")


training = []
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for family, label, color in (
    ("go2_rft", "Randomized RFT", "tab:blue"),
    ("go2_rft_nominal_matched", "Fixed material, same command distribution", "tab:orange"),
):
    runs = sorted(Path("logs/rsl_rl", family).glob("20*"))
    if not runs:
        continue
    run = runs[-1]
    events = EventAccumulator(str(run)).Reload()
    tags = events.Tags()["scalars"]
    for ax, tag, title in zip(
        axes.flat,
        ("Train/mean_reward", "Train/mean_episode_length", "Metrics/velocity_error", "Perf/total_fps"),
        ("Training return", "Episode length [control steps]", "Episode tracking error [m/s]", "Training transitions/s"),
        strict=True,
    ):
        if tag not in tags:
            continue
        data = events.Scalars(tag)
        x = np.array([e.step for e in data])
        y = np.array([e.value for e in data])
        ax.plot(x, y, alpha=0.15, color=color)
        if len(y) >= 20:
            ax.plot(x[19:], np.convolve(y, np.ones(20) / 20, mode="valid"), label=label, color=color)
        else:
            ax.plot(x, y, label=label, color=color)
        ax.set(title=title, xlabel="PPO iteration")
        ax.grid(alpha=0.2)
    rewards = events.Scalars("Train/mean_reward")
    training.append(
        dict(
            label=label,
            directory=str(run),
            iterations=len(rewards),
            checkpoint=str(run / f"model_{rewards[-1].step}.pt"),
            transitions=len(rewards) * 512 * 24,
            last_return=rewards[-1].value,
        )
    )
axes[0, 0].legend()
fig.suptitle("Historical training with unintended ambient fluid drag; single seed", y=1.02)
save(fig, "01-training.png")

cases = []
for name in (
    "dr_rft_nominal",
    "dr_mpm_deep",
    "nominal_rft_nominal",
    "dr_rft_random_matched",
    "matched_rft_nominal",
    "matched_mpm_deep",
    "matched_rft_random",
    "dr_rft_weak",
    "dr_mpm_weak",
    "dr_rft_strong",
    "dr_mpm_strong",
):
    if (ROOT / name / "summary.json").exists():
        summary, data = read_case(name)
        cases.append(dict(case=name, **summary))

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for name, label in (("dr_rft_nominal", "RFT + bearing"), ("dr_mpm_deep", "MPM (coarse pilot)")):
    if not (ROOT / name / "summary.json").exists():
        continue
    summary, data = read_case(name)
    x = data["state"][:, 0]
    mask = x[:, 0] <= 2.0
    x = x[mask]
    f = data["body_wrench"][mask]
    axes[0, 0].plot(x[:, 0], x[:, 4], label=label)
    axes[0, 1].plot(x[:, 0], x[:, 7], label=label)
    axes[1, 0].plot(x[:, 0], f[:, :, 2].sum(1), label=label)
    axes[1, 1].plot(x[:, 1], x[:, 2], label=label)
    failure = summary["first_failure_seconds"]
    if failure is not None:
        axes[0, 1].axvline(failure, color="firebrick", ls=":", label=f"Failure at {failure:.2f}s")
axes[0, 0].axhline(0.3, ls="--", color="black", label="Command")
axes[0, 0].set(xlabel="Time [s]", ylabel="Body-forward speed [m/s]")
axes[0, 1].axhline(0.16, ls="--", color="black", label="Height failure threshold")
axes[0, 1].set(xlabel="Time [s]", ylabel="Base height above initial bed [m]")
axes[1, 0].set(
    xlabel="Time [s]", ylabel="Vertical granular load [N]", title="RFT: interval mean; MPM: relaxed feedback sample"
)
axes[1, 1].set(xlabel="World x [m]", ylabel="World y [m]")
for ax in axes.flat:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("Historical policy with fluid-density bug: transfer comparison is confounded", y=1.02)
save(fig, "02-transfer.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, label in (
    ("stance_union", "RFT, zero actions"),
    ("mpm_stance_relaxed", "MPM 20 mm"),
    ("mpm_stance_fine", "MPM 10 mm"),
):
    if not (ROOT / name / "summary.json").exists():
        continue
    summary, data = read_case(name)
    x = data["state"][:, 0]
    mask = x[:, 0] <= 1
    axes[0].plot(x[mask, 0], x[mask, 7], label=label)
    axes[1].plot(x[mask, 0], data["body_wrench"][mask, :, 2].sum(1), label=label)
axes[0].set(xlabel="Time [s]", ylabel="Base height [m]")
axes[1].set(xlabel="Time [s]", ylabel="Vertical granular load [N]")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
fig.suptitle("Historical standing probes — static-support gate unresolved; MPM surface friction was 1.0")
save(fig, "03-stance-reference.png")

if all((ROOT / name / "summary.json").exists() for name in ("dr_rft_random_matched", "matched_rft_random")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bank = []
    for index, (name, label, color) in enumerate(
        (
            ("dr_rft_random_matched", "Randomized training", "tab:blue"),
            ("matched_rft_random", "Nominal-material training", "tab:orange"),
        )
    ):
        summary, data = read_case(name)
        bank.append(summary)
        state = data["state"]
        failed = np.any(state[:, :, 9], axis=0)
        centers = np.arange(state.shape[1]) + (index - 0.5) * 0.38
        axes[0].bar(centers, state[:, :, 8].mean(0), width=0.36, color=color, label=label)
        axes[0].scatter(centers[failed], state[:, failed, 8].mean(0) + 0.008, color="firebrick", marker="x")
        axes[1].bar(label, failed.mean() * 100, color=color)
        axes[1].text(index, failed.mean() * 100 + 2, f"{failed.sum()}/{len(failed)}", ha="center")
    for key in ("commands", "material_parameters", "surface_heights"):
        assert bank[0][key] == bank[1][key], f"Unmatched evaluation bank: {key}"
    axes[0].set(xlabel="Matched field/command trial", ylabel="Mean tracking error [m/s]")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Trials with a failure [%]", ylim=(0, 100))
    fig.suptitle("Same 16 fields/commands, 10 s each; red crosses mark failures; one training seed")
    save(fig, "04-random-fields.png")

# Reconstruct the authored primitive surfaces at one recorded robot pose.
path = ROOT / "dr_rft_nominal"
if (path / "surfaces.json").exists():
    import warp as wp
    from newton import Axis, Mesh

    inventory = json.loads((path / "surfaces.json").read_text())
    data = np.load(path / "rollout.npz")
    if "body_pose" in data:
        poses = data["body_pose"][min(25, len(data["body_pose"]) - 1)]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        colors = {"foot": "royalblue", "calf": "darkorange", "thigh": "purple", "hip": "seagreen"}
        for row in inventory:
            dims = row["scale"]
            if row["kind"] == "SPHERE":
                mesh = Mesh.create_sphere(dims[0], num_latitudes=6, num_longitudes=12, compute_inertia=False)
            elif row["kind"] == "CYLINDER":
                mesh = Mesh.create_cylinder(dims[0], dims[1], up_axis=Axis.Z, segments=12, compute_inertia=False)
            elif row["kind"] == "BOX":
                mesh = Mesh.create_box(*dims, compute_inertia=False)
            else:
                continue
            local = row["local_pose"]
            pose = poses[row["body"]]
            local_tf = wp.transform(wp.vec3(*local[:3]), wp.quat(*local[3:]))
            body_tf = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            points = np.asarray(
                [wp.transform_point(body_tf, wp.transform_point(local_tf, wp.vec3(*v))) for v in mesh.vertices]
            )
            kind = row["body_label"].split("_")[-1]
            ax.plot_trisurf(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                triangles=np.asarray(mesh.indices).reshape(-1, 3),
                color=colors.get(kind, "gray"),
                edgecolor="black",
                linewidth=0.15,
                alpha=0.45,
            )
        center = poses[0, :3]
        gx, gy = np.meshgrid(center[0] + np.array([-0.6, 0.6]), center[1] + np.array([-0.45, 0.45]))
        ax.plot_surface(gx, gy, np.zeros_like(gx), color="sandybrown", alpha=0.15)
        ax.set(
            xlabel="World x [m]",
            ylabel="World y [m]",
            zlabel="World z [m]",
            title=(
                "All 27 collision surfaces: feet blue, calves orange, thighs purple\n"
                "Transparent plane: RFT free surface; mesh is fixed during training"
            ),
        )
        ax.set_zlabel("")
        fig.text(0.94, 0.5, "World z [m]", rotation=90)
        ax.set_box_aspect((1.2, 0.9, 0.6))
        save(fig, "05-surfaces.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for filename, label in (
    ("surface_convergence.json", "Angular refinement only"),
    ("surface_quadrature_convergence.json", "Angular + edge refinement"),
):
    if not (OUT / filename).exists():
        continue
    rows = json.loads((OUT / filename).read_text())["results"][:-1]
    for ax, key in zip(axes, ("force_relative_error", "torque_relative_error"), strict=True):
        ax.plot([r["triangles"] for r in rows], [100 * r[key] for r in rows], "o-", label=label)
        ax.set_xscale("log")
        ax.set(xlabel="Quadrature triangles", ylabel="Relative load error [%]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
axes[0].set_title("Body forces")
axes[1].set_title("Body torques")
fig.suptitle("Four common recorded poses/twists vs finest mesh; self-convergence, not physical accuracy")
save(fig, "06-surface-convergence.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, suffix, label in zip(
    axes, ("weak", "nominal", "strong"), ("Weaker field", "Nominal field", "Stronger field"), strict=True
):
    names = (f"dr_rft_{suffix}", "dr_mpm_deep" if suffix == "nominal" else f"dr_mpm_{suffix}")
    for name in names:
        if not (ROOT / name / "summary.json").exists():
            continue
        summary, data = read_case(name)
        state = data["state"][:, 0]
        mask = state[:, 0] <= 2
        ax.plot(state[mask, 0], state[mask, 7], label=summary["field"].upper())
    ax.axhline(0.16, color="black", ls="--", lw=0.8)
    ax.set(title=label, xlabel="Time [s]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Base height above initial bed [m]")
fig.suptitle("Frozen randomized policy across matched material inputs; MPM reference is not converged")
save(fig, "07-material-transfer.png")

convergence = []
manifest = None
for iteration in (500, 750, 999):
    name = f"convergence_dr_{iteration}"
    if not (ROOT / name / "summary.json").exists():
        continue
    summary, data = read_case(name)
    current = {key: summary[key] for key in ("commands", "material_parameters", "surface_heights")}
    if manifest is not None:
        assert current == manifest, "Checkpoint validation must use identical field/command samples"
    manifest = current
    failed = np.any(data["state"][:, :, 9], axis=0)
    convergence.append(
        dict(
            iteration=iteration,
            case=name,
            seed=summary["seed"],
            trials=len(failed),
            failures=int(failed.sum()),
            mean_velocity_error=summary["mean_velocity_error"],
        )
    )
if convergence:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [row["iteration"] for row in convergence]
    axes[0].plot(x, [row["mean_velocity_error"] for row in convergence], "o-")
    axes[0].set(xlabel="Training checkpoint iteration", ylabel="Mean tracking error [m/s]")
    axes[1].plot(x, [100 * row["failures"] / row["trials"] for row in convergence], "o-", color="firebrick")
    axes[1].set(xlabel="Training checkpoint iteration", ylabel="Trials with failure [%]", ylim=(0, 100))
    for row in convergence:
        axes[1].annotate(
            f"{row['failures']}/{row['trials']}",
            (row["iteration"], 100 * row["failures"] / row["trials"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Randomized policy: fixed 16-trial validation bank, seed 123, 10 s; convergence not established")
    save(fig, "08-checkpoint-validation.png")

corrected_training = []
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for family, label, color in (("go2_rigid_gait_control", "Training the repaired rigid-ground control", "tab:orange"),):
    runs = sorted(Path("logs/rsl_rl", family).glob("20*"))
    if not runs:
        continue
    events = EventAccumulator(str(runs[-1])).Reload()
    available = events.Tags()["scalars"]
    for ax, tag, title in zip(
        axes.flat,
        (
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Metrics/velocity_error",
            "Metrics/min_foot_clearance",
            "Metrics/gait_failure_fraction",
            "Perf/total_fps",
        ),
        (
            "Training return",
            "Episode length [control steps]",
            "Training tracking error [m/s]",
            "Minimum lateral foot clearance [m]",
            "Foot-placement failure fraction",
            "Training transitions/s",
        ),
        strict=True,
    ):
        if tag not in available:
            continue
        data = events.Scalars(tag)
        x = np.array([event.step for event in data]) * 4096 * 24 / 1e6
        y = np.array([event.value for event in data])
        ax.plot(x, y, color=color, alpha=0.15)
        window = min(20, len(y))
        ax.plot(x[window - 1 :], np.convolve(y, np.ones(window) / window, mode="valid"), color=color, label=label)
        ax.set(title=title, xlabel="Million transitions")
        ax.grid(alpha=0.2)
    last = events.Scalars("Perf/total_fps")[-1]
    corrected_training.append(
        dict(
            family=family,
            directory=str(runs[-1]),
            last_iteration=last.step,
            transitions=(last.step + 1) * 4096 * 24,
            ambient_fluid_density=0,
        )
    )
axes[0, 0].legend(fontsize=8)
fig.suptitle("Repaired rigid-ground control: 4,096 environments; full commands throughout", y=1.02)
save(fig, "09-corrected-training.png")

corrected_cases = []
for name in (
    "rigid_gait_control",
    "rigid_gait_stand_validation",
    "rigid_gait_walk_validation",
    "rigid_gait_maximum_validation",
    "rigid_gait_commands_123",
    "rigid_gait_commands_456",
    "rigid_gait_commands_789",
    "material_rft_preview",
    "material_achieved_validation",
    "material_full_validation",
    "corrected_stance",
    "corrected_rigid_final",
    "corrected_rft_curriculum_nominal",
    "corrected_rft_curriculum_mpm",
    "corrected_zero_layer_policy",
    "corrected_thin_layer_005",
    "corrected_thin_layer_020",
    "corrected_thin_layer_050",
):
    if (ROOT / name / "summary.json").exists():
        summary, _ = read_case(name)
        corrected_cases.append(dict(case=name, **summary))

gait = {}
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, (name, title) in zip(
    axes,
    (
        ("corrected_rft_curriculum_nominal", "Final RFT-trained policy on RFT sand"),
        (
            "rigid_gait_control" if (ROOT / "rigid_gait_control/summary.json").exists() else "corrected_rigid_final",
            "Rigid-ground policy on a rigid floor",
        ),
    ),
    strict=True,
):
    if not (ROOT / name / "summary.json").exists():
        continue
    summary, data = read_case(name)
    labels = summary["body_labels"]
    poses = data["body_pose"]
    base = poses[:, next(i for i, label in enumerate(labels) if label.endswith("/base"))]
    local_feet = []
    for leg in ("RL", "RR"):
        foot = poses[:, next(i for i, label in enumerate(labels) if label.endswith(f"/{leg}_foot"))]
        displacement = foot[:, :3] - base[:, :3]
        xyz, scalar = -base[:, 3:6], base[:, 6:7]
        local = displacement + 2 * np.cross(xyz, np.cross(xyz, displacement) + scalar * displacement)
        local_feet.append(local[:, 1])
    separation = local_feet[0] - local_feet[1]
    crossed = separation < 0
    time = data["state"][:, 0, 0]
    gait[name] = dict(
        first_rear_foot_order_reversal_s=float(time[np.flatnonzero(crossed)[0]]) if crossed.any() else None,
        reversed_rear_foot_order_fraction=float(crossed.mean()),
        min_rear_foot_separation_m=float(separation.min()),
        mean_rear_foot_separation_m=float(separation.mean()),
    )
    ax.plot(time, separation, label="Left rear foot y − right rear foot y, in base frame")
    ax.axhline(0, color="red", linestyle="--", label="Left/right foot order reverses below zero")
    ax.set(title=f"{title}: rear feet crossed during {crossed.mean():.1%} of frames", ylabel="Separation [m]")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("Time [s]")
fig.suptitle("Rear-foot stance diagnostic; this does not certify collision-free or natural gait")
save(fig, "12-gait-diagnostics.png")
(OUT / "gait_diagnostics.json").write_text(json.dumps(gait, indent=2))

validation = [case for case in corrected_cases if case["case"] == "material_full_validation"]
if validation:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"Material width {case['validation_difficulty']:.0%}" for case in validation]
    axes[0].bar(labels, [1 - case["failure_fraction"] for case in validation])
    axes[0].set(title="Held-out 10 s episode completion", ylim=(0, 1.05))
    axes[1].bar(labels, [case["mean_velocity_error"] for case in validation])
    axes[1].axhline(0.1, color="grey", linestyle="--")
    axes[1].set(title="Held-out mean tracking error [m/s]")
    fig.suptitle("Frozen policy: 128 environments, full commands, seed 123")
    save(fig, "10-material-validation.png")

rigid_trials = (
    ("rigid_gait_stand_validation", "Stand"),
    ("rigid_gait_walk_validation", "Walk 0.30 m/s"),
    ("rigid_gait_maximum_validation", "Walk 0.50 m/s"),
    ("rigid_gait_commands_123", "Command bank 1"),
    ("rigid_gait_commands_456", "Command bank 2"),
    ("rigid_gait_commands_789", "Command bank 3"),
)
rigid_data = [(read_case(name)[0], label) for name, label in rigid_trials if (ROOT / name / "summary.json").exists()]
if rigid_data:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title, threshold in zip(
        axes,
        ("mean_velocity_error", "min_signed_foot_y", "mean_contact_slip_m_s"),
        ("Planar tracking error [m/s]", "Minimum foot clearance from centerline [m]", "Mean foot slip [m/s]"),
        (0.10, 0.01, 0.10),
        strict=True,
    ):
        ax.bar([label for _, label in rigid_data], [summary[key] for summary, _ in rigid_data])
        ax.axhline(threshold, color="red", linestyle="--")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Repaired rigid control: fixed commands for 20 s; three held-out banks of 128 trials for 10 s")
    save(fig, "13-rigid-control-validation.png")

layer_cases = (
    ("corrected_rigid_final", "Rigid"),
    ("corrected_zero_layer_policy", "0 mm"),
    ("corrected_thin_layer_005", "5 mm"),
    ("corrected_thin_layer_020", "20 mm"),
    ("corrected_thin_layer_050", "50 mm"),
)
if any((ROOT / name / "summary.json").exists() for name, _ in layer_cases):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in layer_cases:
        if not (ROOT / name / "summary.json").exists():
            continue
        summary, data = read_case(name)
        state = data["state"][:, 0]
        axes[0].plot(state[:, 0], state[:, 7], label=label)
        axes[1].plot(state[:, 0], state[:, 8], label=label)
    axes[0].set(title="Base height above free surface [m]", xlabel="Time [s]")
    axes[1].set(title="Tracking error [m/s]", xlabel="Time [s]")
    axes[0].legend()
    fig.suptitle("Same frozen rigid-ground policy: substrate and finite RFT layers")
    save(fig, "11-layer-limit.png")

(OUT / "results.json").write_text(
    json.dumps(
        dict(
            training=training,
            inference=cases,
            convergence_validation=convergence,
            corrected_training=corrected_training,
            corrected_inference=corrected_cases,
        ),
        indent=2,
    )
)
videos = render_videos(ROOT, corrected_cases, gait)
evaluation_path = ROOT / "training/evaluation_status.json"
evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
rigid_validation = "<p>Rigid-control validation is in progress.</p>"
if "rigid_control_accepted" in evaluation:
    verdict = (
        "Passed the specified rigid-control checks" if evaluation["rigid_control_accepted"] else "Validation failed"
    )
    labels = {
        "no_falls": "No falls across fixed commands, held-out trials and the command sequence",
        "foot_placement": "Feet stay on the assigned side of the body",
        "tracking": "Mean planar velocity error below 0.10 m/s",
        "standing": "Standing drift below 0.03 m/s",
        "contact_slip": "Mean foot slip below 0.10 m/s",
        "capacity": "No detected contact-buffer overflow",
        "every_foot_steps": "At least three steps with each foot during walking tests",
    }
    rows = "".join(
        f"<tr><td>{labels[key]}</td><td>{'Pass' if passed else 'Fail'}</td></tr>"
        for key, passed in evaluation["rigid_validation_gates"].items()
    )
    rigid_validation = f"<p><strong>{verdict}.</strong></p><table>{rows}</table>"
figures = "".join(
    f"<figure><h3>{title}</h3><p>{description}</p><img src='{name}' alt='{title}' loading='lazy'></figure>"
    for name, title, description in (
        (
            "09-corrected-training.png",
            "Learning curves",
            "The repaired rigid task uses self-collision, posture/contact rewards and foot-placement checks. "
            "It trains with 4,096 environments and the full velocity-command range from the start.",
        ),
        (
            "13-rigid-control-validation.png",
            "Rigid-ground validation",
            "Fixed standing, walking and maximum-speed tests, followed by 384 randomized-command trials. "
            "Each test checks falls, foot placement, tracking and contact slip.",
        ),
    )
    if (OUT / name).exists()
)
(OUT / "index.html").write_text(
    """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2: RFT training and MPM evaluation</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
video,img{width:100%;border-radius:6px}
.video-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
article,figure{background:white;padding:20px;margin:0 0 24px;border-radius:10px}figure{margin:20px 0}
h1,h2,h3{line-height:1.2}h3{margin-top:0}details{margin:12px 0}summary{cursor:pointer;color:#1d4ed8}
code{overflow-wrap:anywhere}a{color:#1d4ed8}
.outcome{background:#fff7ed;border-left:4px solid #c2410c;padding:16px}</style>
<h1>Go2: training with RFT, testing with MPM</h1>
<p>The goal is to train with an inexpensive granular approximation, then evaluate the same policy
with particle-based material simulation. MPM is used only after training.</p>
<p class="outcome"><strong>Priority: validate the rigid-ground control first.</strong>
The previous control had a crossed-leg gait. A repaired task adds self-collision, foot-placement checks
and motion/contact rewards. Earlier RFT/MPM results remain provisional pending task validation.</p>
<p><a href="../live/">Training status and live curves</a></p>
<h2>Final policy comparisons</h2>
<p>The rigid-control recording tests a command sequence including the 0.50 m/s training maximum.
The earlier RFT/MPM clips both use 0.30 m/s and share the same weights.
Green arrows show granular forces on robot bodies.</p>"""
    + videos
    + "<h2>Training and validation</h2>"
    + rigid_validation
    + figures
    + """<details><summary>Rear-leg gait measurements</summary>
<p>The traces measure signed lateral separation of the rear feet in the base frame.
Negative values indicate reversed left/right foot order. Low velocity error alone does not measure gait quality.
The rigid trace uses the repaired policy once its recording is available; the RFT trace is from the earlier task.</p>
<img src="12-gait-diagnostics.png" alt="Rear-foot separation for the final policies" loading="lazy">
</details>
<details><summary>Rigid-ground limit as sand thickness vanishes</summary>
<p>This earlier physics diagnostic applied the superseded rigid policy to RFT layers of 0, 5, 20 and 50 mm.
At zero thickness the granular force vanishes and the motion matches the rigid-floor rollout, including its gait defect.
The graph shows the layer comparison; positive-thickness confinement accuracy remains unvalidated.</p>
<img src="11-layer-limit.png" alt="Same rigid-ground policy with different sand-layer thicknesses" loading="lazy">
</details>
<details><summary>Experiment settings and data</summary>
<p>Both runs used 4,096 environments and seed 42. Commands were sampled at reset across the full range:
forward 0.1–0.5 m/s, lateral ±0.1 m/s, yaw ±0.3 rad/s, and 10% standing trials.</p>
<p>RFT material randomization covered density 1,400–1,800 kg/m³, internal friction 0.3–0.5,
surface/internal friction ratio 0.6–1.0, strength multiplier 0.8–1.2, and surface elevation ±0.02 m.</p>
<p>RFT includes an experimental near-rest bearing term. The MPM reference still needs resolution,
timestep and coupling-convergence checks. Force arrows are qualitative: RFT uses interval-average impulses,
while MPM uses the last coupled feedback sample.</p>
<p><a href="results.json">Full numerical records</a> · <a href="gait_diagnostics.json">Gait measurements</a></p>
</details>"""
)
print(f"Built {OUT / 'index.html'}")
