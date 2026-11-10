# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build graphs and a local video gallery from actual Go2 training/inference artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    ("go2_rft_nominal_matched", "Nominal material, matched commands", "tab:orange"),
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
fig.suptitle("Go2 RFT training — actual TensorBoard records; single training seed", y=1.02)
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
fig.suptitle("Frozen randomized-RFT policy: transfer discrepancy; MPM convergence not established", y=1.02)
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

(OUT / "results.json").write_text(json.dumps(dict(training=training, inference=cases), indent=2))
cards = []
case_labels = {
    "dr_rft_nominal": "Randomized policy — nominal RFT",
    "dr_mpm_deep": "Same randomized policy — nominal MPM",
    "nominal_rft_nominal": "Fixed-forward demonstration policy — RFT",
    "matched_rft_nominal": "Command-matched nominal policy — RFT",
    "matched_mpm_deep": "Same nominal policy — MPM",
    "dr_rft_weak": "Randomized policy — weaker RFT material",
    "dr_mpm_weak": "Same randomized policy — weaker MPM material",
    "dr_rft_strong": "Randomized policy — stronger RFT material",
    "dr_mpm_strong": "Same randomized policy — stronger MPM material",
}
for case in cases:
    name = case["case"]
    if (ROOT / name / "rollout.mp4").exists():
        cards.append(
            f"<article><h2>{html.escape(case_labels.get(name, name))}</h2>"
            f'<video controls preload="metadata" src="../{name}/rollout.mp4"></video>'
            f"<p>Failure: {case['first_failure_seconds']}; "
            f"velocity error: {case.get('mean_velocity_error', 0):.3f} m/s. "
            f"Execution error: {html.escape(str(case.get('execution_error')))}.</p></article>"
        )
stock = ROOT / "stock_go2_mpm_runway"
if (stock / "rollout.mp4").exists():
    cards.append(
        "<article><h2>Stock Go2 / MPM runway smoke</h2>"
        '<video controls preload="metadata" src="../stock_go2_mpm_runway/rollout.mp4"></video>'
        "<p>Separate pretrained 48-value policy; rigid-supported 8 cm strip. Integration showcase only.</p></article>"
    )
images = "".join(f'<img src="{p.name}" alt="{p.stem}">' for p in sorted(OUT.glob("*.png")))
(OUT / "index.html").write_text(
    """<!doctype html><meta charset="utf-8"><title>Go2 RFT / MPM investigation</title>
<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f1f5f9;color:#0f172a}
video,img{width:100%}article{background:white;padding:16px;margin:20px 0;border-radius:10px}</style>
<h1>Go2 RFT training and MPM inference</h1><p>Actual Newton physics and frozen-policy recordings.
RFT uses an experimental bearing extension. MPM convergence and transfer accuracy are not established.
Low posture and transfer failures are retained.</p>"""
    + "".join(cards)
    + images
)
print(f"Built {OUT / 'index.html'}")
