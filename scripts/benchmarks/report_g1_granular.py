# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish G1 training, held-out checks and paired rigid/RFT/MPM policy recordings."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from go2_granular_videos import PRIMARY_VIDEOS, render_videos, video_cases
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular/g1")


def material_curriculum(events: EventAccumulator, out: Path, iteration: int) -> str:
    """Show the achieved curriculum and sampled depths separately from the target range."""
    tags = events.Tags()["scalars"]
    if "Curriculum/difficulty" not in tags:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for axis, tag, label, scale in (
        (0, "Curriculum/difficulty", "Material difficulty", 1),
        (1, "Material/depth_mean", "Mean reset depth", 1000),
        (1, "Material/depth_max", "Maximum reset depth", 1000),
    ):
        if tag in tags:
            values = events.Scalars(tag)
            axes[axis].plot([v.step for v in values], [scale * v.value for v in values], label=label)
    axes[0].set(ylabel="Material difficulty", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Sampled layer depth [mm]")
    axes[1].axhline(118, color="grey", linestyle="--", label="Target maximum: 118 mm")
    for ax in axes:
        ax.set(xlabel="PPO iteration")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "material-curriculum.png", dpi=150)
    plt.close(fig)
    return (
        "<h3>RFT material curriculum</h3>"
        "<p>Difficulty controls both material-property spread and layer depth. The depth traces summarize resetting "
        "environments; the dashed line is the target maximum. Commands keep their full range throughout training.</p>"
        f"<img src='material-curriculum.png?v={iteration}' alt='Achieved material difficulty and reset layer depths'>"
    )


def publish() -> None:
    """Read completed outputs; never treat training reward as a convergence decision."""
    out = ROOT / "report"
    out.mkdir(parents=True, exist_ok=True)
    status_file = ROOT / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {"stage": "G1 setup and native checks"}
    metrics = (
        ("Train/mean_reward", "Training return"),
        ("Train/mean_episode_length", "Episode length [steps; max 500]"),
        ("Metrics/velocity_error", "Planar tracking error [m/s]"),
        ("Metrics/min_foot_clearance", "Foot clearance from body centerline [m]"),
        ("Perf/total_fps", "Training transitions/s"),
        ("Metrics/gait_failure_fraction", "Foot-placement failure fraction"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    has_training = False
    counts = []
    rows = []
    curriculum = ""
    for policy, label, color in (("rigid", "Rigid", "tab:blue"), ("rft", "RFT", "tab:orange")):
        runs = sorted(Path("logs/rsl_rl", f"g1_granular_{policy}_max1").glob("20*"))
        if not runs:
            rows.append(f"<tr><td>{label}</td><td>Not started</td><td>—</td><td>—</td></tr>")
            continue
        events = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0}).Reload()
        tags = events.Tags()["scalars"]
        count = events.Scalars("Perf/total_fps")[-1].step + 1 if "Perf/total_fps" in tags else 0
        counts.append(count)
        if policy == "rft":
            curriculum = material_curriculum(events, out, count)
        difficulty = (
            f"{events.Scalars('Curriculum/difficulty')[-1].value:.0%}" if "Curriculum/difficulty" in tags else "—"
        )
        gates = status.get("validation", {}).get(policy, {})
        failed = [key.replace("_", " ") for key, value in gates.items() if not value]
        verdict = "Passed" if gates and not failed else "Failed: " + ", ".join(failed) if gates else "Pending"
        rows.append(
            f"<tr><td>{label}</td><td>{count:,} / 1,500 iterations</td><td>{verdict}</td><td>{difficulty}</td></tr>"
        )
        for ax, (tag, _) in zip(axes.flat, metrics, strict=True):
            if tag in tags:
                has_training = True
                values = events.Scalars(tag)
                x = np.array([v.step for v in values])
                y = np.array([v.value for v in values])
                ax.plot(x, y, color=color, alpha=0.15)
                window = min(20, len(y))
                ax.plot(
                    x[window - 1 :], np.convolve(y, np.ones(window) / window, mode="valid"), color=color, label=label
                )
    for ax, (_, title) in zip(axes.flat, metrics, strict=True):
        ax.set(title=title, xlabel="PPO iteration")
        ax.grid(alpha=0.2)
    figures = ""
    if has_training:
        axes[0, 0].legend()
        fig.suptitle("G1 rigid and RFT training · 4,096 environments · seed 42")
        fig.tight_layout()
        fig.savefig(out / "training-comparison.png", dpi=150)
        figures = (
            "<h2>Training curves</h2><p>Rigid and RFT share each axis. Solid curves show 20-iteration "
            "moving averages; faint curves show raw measurements.</p>"
            f"<img src='training-comparison.png?v={sum(counts)}' alt='G1 rigid and RFT shared learning curves'>"
        )
    plt.close(fig)
    cases = []
    for name, _, _ in video_cases((1.0, 0.5)):
        path = ROOT / "evaluations" / name / "summary.json"
        if path.exists() and name in status.get("published_videos", []):
            cases.append(dict(case=name, **json.loads(path.read_text())))
    plots = []
    labels = {name: title for name, title, *_ in PRIMARY_VIDEOS}
    for speed in (0.5, 1.0):
        names = {name: base for name, base, command in video_cases((1.0, 0.5)) if command == speed}
        available = [case for case in cases if case["case"] in names]
        if not available:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for case in available:
            with np.load(ROOT / "evaluations" / case["case"] / "rollout.npz") as data:
                state = data["state"][:, 0]
                for ax, column in zip(axes, (4, 8, 7), strict=True):
                    ax.plot(state[:, 0], state[:, column], label=labels[names[case["case"]]])
        axes[0].axhline(speed, color="grey", linestyle="--")
        axes[0].legend(fontsize=6)
        axes[2].axhline(0.4, color="grey", linestyle=":")
        for ax, label in zip(
            axes,
            ("Forward speed [m/s]", "Planar tracking error [m/s]", "Height above initial surface [m]"),
            strict=True,
        ):
            ax.set(xlabel="Time [s]", ylabel=label)
            ax.grid(alpha=0.2)
        fig.suptitle(f"G1 frozen policies · {speed:.2f} m/s command")
        fig.tight_layout()
        filename = f"forward-{speed:g}.png"
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
        plots.append(
            f"<section><h3>Comparison at {speed:.2f} m/s</h3>"
            "<p>Velocities and commands use the robot frame. Curves include initial acceleration "
            "and time after a failure threshold.</p>"
            f"<img src='{filename}' alt='Speed, tracking and height at {speed:g} m/s'>"
        )
        folder = ROOT / f"materials_{speed:g}"
        if speed == 0.5 and (folder / "policy-and-support.png").exists():
            plots.append(
                "<h4>Material sensitivity</h4>"
                "<p>Both policies use the same material settings for 3 s. Support includes all rigid contacts "
                "and time after failure. These trials do not establish material equivalence.</p>"
                f"<img src='../materials_{speed:g}/policy-and-support.png' "
                "alt='Tracking and support for three materials'>"
            )
        plots.append("</section>")
    numerical_cases = [case for case in status.get("cases", []) if case["case"].startswith("mpm_startup_")]
    numerics = ""
    if numerical_cases:
        fig, axes = plt.subplots(4, 2, figsize=(12, 11), squeeze=False)
        numerical_rows = []
        for case in numerical_cases:
            speed = case["commands"][0][0]
            column = 0 if speed == 0.5 else 1
            passes, feedback = case["proxy_iterations"], case["proxy_relaxation"]
            label = f"Coupling passes {passes}, feedback {feedback:g}"
            with np.load(ROOT / "evaluations" / case["case"] / "rollout.npz") as data:
                state = data["state"][:, 0]
                values = (
                    state[:, 7],
                    state[:, 4],
                    data["metrics"][:, 0, 3],
                    data["support_wrenches"][:, [2, 5]].sum(-1),
                )
                for ax, value in zip(axes[:, column], values, strict=True):
                    ax.plot(state[:, 0], value, label=label)
            first = case["first_failure_seconds"]
            outcome = "None in 2 s" if first is None else f"{first:.2f} s"
            numerical_rows.append(
                f"<tr><td>{speed:.2f}</td><td>{passes}</td><td>{feedback:g}</td>"
                f"<td>{case['mean_velocity_error']:.3f}</td><td>{outcome}</td></tr>"
            )
        for column, speed in enumerate((0.5, 1.0)):
            axes[0, column].set_title(f"Forward command {speed:.2f} m/s")
            axes[0, column].axhline(0.4, color="grey", linestyle=":")
            if axes[0, column].get_legend_handles_labels()[0]:
                axes[0, column].legend(fontsize=8)
            for ax, label in zip(
                axes[:, column],
                ("Height above initial surface [m]", "Forward speed [m/s]", "Torso tilt [rad]", "Vertical support [N]"),
                strict=True,
            ):
                ax.set(xlabel="Time [s]", ylabel=label)
                ax.grid(alpha=0.2)
        fig.suptitle("Frozen G1 RFT policy: MPM coupling sensitivity · 2 s checks")
        fig.tight_layout()
        fig.savefig(out / "mpm-coupling.png", dpi=150)
        plt.close(fig)
        numerics = (
            "<h2 id='mpm-numerics'>MPM coupling sensitivity</h2>"
            "<p>Same frozen RFT checkpoint, prepared 5 × 3.2 × 0.118 m bed and seed 123. "
            "Only the coupling controls vary. Short tests do not establish 10 s stability or MPM convergence.</p>"
            "<table><tr><th>Command [m/s]</th><th>Coupling passes</th><th>Feedback factor</th>"
            "<th>Tracking error [m/s]</th><th>First failure threshold</th></tr>"
            + "".join(numerical_rows)
            + "</table><img src='mpm-coupling.png' alt='G1 height, speed, tilt and support versus coupling controls'>"
        )
    videos = (
        f"<h2>Policy recordings · {len(cases)} of 8 verified</h2>"
        + render_videos(ROOT / "evaluations", cases, {}, url_prefix="../evaluations/", speeds=(1.0, 0.5), robot="G1")
        if cases
        else "<h2>Policy recordings</h2><p>Final recordings follow training and held-out baseline validation.</p>"
    )
    page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1: rigid, RFT and MPM comparisons</title><style>
body{font:16px/1.5 system-ui;max-width:1200px;margin:24px auto;padding:20px;background:#f1f5f9;color:#0f172a}
article{background:white;padding:20px;border-radius:10px;margin:24px 0}video,img{width:100%;border-radius:6px}
.paired-videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
table{border-collapse:collapse}td,th{padding:10px;border:1px solid #cbd5e1}code{overflow-wrap:anywhere}
.status{background:#dbeafe;padding:16px}a,summary{color:#1d4ed8}</style>"""
    page += (
        '<h1>G1: rigid, RFT and MPM comparisons</h1><p><a href="/report/">Go2 comparison</a> · '
        '<a href="../STATUS.md">G1 status and settings</a></p>'
    )
    page += (
        f'<p class="status">{html.escape(status["stage"])}</p>'
        "<table><tr><th>Policy</th><th>Training</th><th>Held-out checks</th><th>Material difficulty</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    if status.get("active_case"):
        progress_url = json.dumps(f"../evaluations/{status['active_case']}/progress.json")
        page += (
            "<p id='evaluation-progress'>Initializing evaluation…</p><script>"
            f"const progressUrl={progress_url};"
            "setInterval(()=>fetch(progressUrl+'?t='+Date.now()).then(r=>r.ok?r.json():null).then(p=>{"
            "if(p)document.getElementById('evaluation-progress').textContent='Evaluation progress: '"
            "+Math.round(100*p.completed_steps/p.total_steps)+'% · '+p.simulated_seconds.toFixed(1)"
            "+' / '+p.total_simulated_seconds.toFixed(1)+' s simulated';}).catch(()=>{}),10000);</script>"
        )
    page += """<p>Two fresh policies use 4,096 environments, seed 42 and the same 1,500-iteration PPO budget.
Forward commands span 0.1–1.0 m/s from the first reset, with lateral ±0.1 m/s, yaw ±0.3 rad/s and 10% standing.
Only the RFT policy receives reset-randomized sand and a material curriculum.
Layer depth starts at 0–10 mm and widens to 0–118 mm with material properties as performance improves.
MPM is used only for inference.</p>
<p>The 37-joint G1 uses its minimal asset with shared thigh/calf capsule colliders,
plus its authored foot and torso colliders. Both policies use identical mechanics,
actuators, observations, rewards and self-collision. RFT integrates loads over those collision surfaces.</p>
<p>Each panel compares 1.0 and 0.5 m/s using the same frozen checkpoint and seed 123.
Sand starts 118 mm deep at density 1,600 kg/m³, internal friction 0.4 and surface/internal ratio 1.0.
MPM uses PIC, a 20 mm grid, four substeps and the same prepared 12 × 3.2 m bed for both policies and speeds.
The G1 MPM recordings use one coupling pass and feedback factor 1.0; the earlier Go2 main videos used 0.1.
Controlled G1 tests compare both feedback factors and four-pass refinement.</p>
<p><strong>MPM force convergence and material equivalence remain unvalidated.</strong> These are exploratory transfers.
Support plots include contacts after failure, including non-foot contacts.
A completed training budget is separate from passing gait checks.</p>"""
    page += figures + curriculum + videos + ("<h2>Comparisons</h2>" if plots else "") + "".join(plots) + numerics
    page += (
        "<script>setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused&&!v.ended))"
        "location.reload()},30000)</script>"
    )
    (out / "index.tmp").write_text(page)
    (out / "index.tmp").replace(out / "index.html")
    (out / "results.json").write_text(json.dumps(dict(status=status, cases=cases), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        publish()
        if not args.watch:
            break
        time.sleep(30)
