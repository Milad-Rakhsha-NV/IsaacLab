# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Refresh the local Go2 study dashboard from TensorBoard events and systemd job states.

Run alongside the two training jobs; reads logs only and uses no simulation/GPU resources.
Serve ``logs/go2_granular`` to view ``/live/`` in a browser.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from go2_granular_videos import render_videos
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular")
OUT = ROOT / "live"
RUNS = (
    (
        "rft",
        "RFT: 0–118 mm layers with material curriculum",
        "go2_rft_knee_depth_curriculum",
        "go2-rft-knee-depth-training.service",
        1500,
    ),
    ("rigid", "Validated rigid-ground control", "go2_rigid_gait_control", "go2-rigid-gait-training.service", 600),
)
TAGS = (
    ("Train/mean_reward", "Training return"),
    ("Train/mean_episode_length", "Episode length [control steps; maximum 500]"),
    ("Metrics/velocity_error", "Training tracking error [m/s]"),
    ("Curriculum/difficulty", "Material randomization width [0–1]"),
    ("Metrics/timeout_fraction", "Training episode completion fraction"),
    ("Perf/total_fps", "Training transitions/s"),
)


def refresh() -> None:
    """Publish an atomic snapshot; missing events mean a job has not started logging yet."""
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for key, label, family, unit, target in RUNS:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState,Result,ExecMainStatus"],
            capture_output=True,
            text=True,
            check=False,
        )
        state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        runs = sorted(Path("logs/rsl_rl", family).glob("20*"))
        series = {}
        checkpoint = None
        if runs:
            events = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0}).Reload()
            series = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
            checkpoints = list(runs[-1].glob("model_*.pt"))
            if checkpoints:
                checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1])).name
        last = {tag: values[-1].value for tag, values in series.items() if values}
        progress = series.get("Perf/total_fps", [])
        completed = progress[-1].step + 1 if progress else 0
        event_time = max((values[-1].wall_time for values in series.values() if values), default=None)
        rows.append(
            dict(
                key=key,
                label=label,
                unit=unit,
                state=state,
                directory=str(runs[-1]) if runs else None,
                completed_iterations=completed,
                target_iterations=target,
                transitions=completed * 4096 * 24,
                event_age_seconds=round(time.time() - event_time) if event_time else None,
                checkpoint=checkpoint,
                metrics=last,
            )
        )
        for ax, (tag, _) in zip(axes.flat, TAGS, strict=True):
            values = series.get(tag, [])
            if not values:
                continue
            x = np.array([value.step for value in values])
            y = np.array([value.value for value in values])
            line = ax.plot(x, y, alpha=0.2)[0]
            window = min(20, len(y))
            ax.plot(
                x[window - 1 :],
                np.convolve(y, np.ones(window) / window, mode="valid"),
                color=line.get_color(),
                label=label,
            )
    for ax, (_, title) in zip(axes.flat, TAGS, strict=True):
        ax.set(title=title, xlabel="PPO iteration")
        ax.grid(alpha=0.2)
    axes[0, 2].axhline(0.1, color="grey", linestyle="--", label="Material progression threshold")
    axes[1, 0].set_ylim(0, 1)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("4,096 environments per run · full commands from the start · no MPM training")
    fig.tight_layout()
    fig.savefig(OUT / "curves.tmp", format="png", dpi=130)
    plt.close(fig)
    (OUT / "curves.tmp").replace(OUT / "curves.png")
    evaluation_path = ROOT / "training/evaluation_status.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    rft_evaluation_path = ROOT / "training/rft-evaluation-status.json"
    if rft_evaluation_path.exists():
        rft_evaluation = json.loads(rft_evaluation_path.read_text())
        evaluation["stage"] = rft_evaluation["stage"]
        evaluation["cases"] = evaluation.get("cases", []) + rft_evaluation.get("cases", [])
        evaluation["rft_validation_gates"] = rft_evaluation.get("rft_validation_gates")
    for name in ("rigid_max_forward", "rft_max_forward", "mpm_max_forward"):
        path = ROOT / name / "summary.json"
        if path.exists():
            evaluation.setdefault("cases", []).append(dict(case=name, **json.loads(path.read_text())))
    # Keep the live snapshot compact; per-body/material manifests stay in the case summaries.
    evaluation["cases"] = [
        {
            key: case.get(key)
            for key in (
                "case",
                "checkpoint",
                "field",
                "simulated_seconds",
                "command_schedule",
                "commands",
                "substrate_depth",
                "bed_depth",
                "first_bed_exit_seconds",
                "first_failure_seconds",
                "failure_fraction",
                "mean_velocity_error",
            )
        }
        for case in {case["case"]: case for case in evaluation.get("cases", [])}.values()
    ]
    gait_path = ROOT / "report/gait_diagnostics.json"
    gait = json.loads(gait_path.read_text()) if gait_path.exists() else {}
    snapshot = dict(updated_utc=now.isoformat(), runs=rows, evaluation=evaluation, gait=gait)
    (OUT / "status.tmp").write_text(json.dumps(snapshot, indent=2))
    (OUT / "status.tmp").replace(OUT / "status.json")
    cards = []
    for row in rows:
        metrics = row["metrics"]
        difficulty = metrics.get("Curriculum/difficulty", 0)
        error = metrics.get("Metrics/velocity_error")
        error_text = f"{error:.3f} m/s" if error is not None else "Waiting for episodes"
        status = row["state"].get("ActiveState", "not started")
        if status == "inactive" and row["state"].get("Result") == "success":
            status = (
                "Training budget completed" if row["completed_iterations"] >= row["target_iterations"] else "Stopped"
            )
        if row["state"].get("Result") not in (None, "success"):
            status += f" ({row['state']['Result']})"
        material = f"{difficulty:.0%} of target randomization width" if row["key"] == "rft" else "Rigid plane"
        cards.append(
            f"<article><h2>{html.escape(row['label'])}</h2><p><b>{html.escape(status)}</b> · "
            f"{row['completed_iterations']:,} / {row['target_iterations']:,} iterations</p>"
            f"<progress value='{row['completed_iterations']}' max='{row['target_iterations']}'></progress>"
            f"<p>{row['transitions'] / 1e6:.1f} million transitions · Training error: {error_text}<br>{material}<br>"
            "</p><details><summary>Run details</summary>"
            f"<p>Last checkpoint: {row['checkpoint']} · Event age: {row['event_age_seconds']} s</p>"
            f"<a href='../training/{'rigid-gait' if row['key'] == 'rigid' else 'rft-knee-depth'}.log'>"
            "Training log</a></details></article>"
        )
    videos = render_videos(ROOT, evaluation.get("cases", []), gait)
    evaluation_text = html.escape(evaluation.get("stage", "Validation pipeline has not started"))
    verdict = ""
    if "rigid_control_accepted" in evaluation:
        verdict = (
            "<p><strong>The repaired rigid control passed the specified validation checks.</strong> "
            "The report contains the per-test results. The new RFT run uses the shared locomotion criteria.</p>"
            if evaluation["rigid_control_accepted"]
            else "<p><strong>The rigid control failed validation.</strong> Inspect the gates in the study report.</p>"
        )
    page = (
        '<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Go2 live training</title>'
        "<style>body{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1200px;margin:24px auto;"
        "padding:16px}section{display:flex;gap:20px;flex-wrap:wrap}article{background:white;padding:20px;"
        "border-radius:12px;flex:1;min-width:280px}progress,img,video{width:100%}a{color:#1d4ed8}"
        ".video-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}"
        ".video-grid article{min-width:0}code{overflow-wrap:anywhere}summary{cursor:pointer;color:#1d4ed8}</style>"
        f"<h1>Go2 live training</h1><p>Updates every 30 seconds while videos are paused. "
        f"Snapshot: {now:%Y-%m-%d %H:%M:%S} UTC.</p>"
        "<p>Both runs: 4,096 environments, seed 42, full commands from the first reset. "
        "Only RFT material randomization has a curriculum. MPM is used after training for inference.</p>"
        "<p><b>New RFT run:</b> self-collision, foot placement and contact/posture rewards match the rigid control. "
        "Depth is sampled over 0–118 mm above a rigid floor, capped at half the 236 mm knee-to-sole length. "
        "Material properties have a curriculum; depth and commands use their full ranges from the start.</p>"
        f"<section>{''.join(cards)}</section><img src='curves.png?t={int(time.time())}' alt='Live learning curves'>"
        f"<h2>Final policy comparisons</h2>{verdict}{videos}"
        f"<details><summary>Evaluation progress</summary><p>{evaluation_text}</p></details>"
        "<p>Material progression requires ≥95% episode completion and ≤0.10 m/s mean tracking error in "
        "two consecutive windows of at least 4,096 qualifying episodes. Training curves do not establish "
        "held-out convergence or MPM transfer.</p><p>Full target: density 1,400–1,800 kg/m³; friction 0.3–0.5; "
        "roughness 0.6–1.0; strength multiplier 0.8–1.2; layer depth 0–118 mm above a fixed floor. "
        "Commands: forward 0.1–0.5 m/s, lateral ±0.1 m/s, yaw ±0.3 rad/s; 10% standing.</p>"
        '<p><a href="../report/">Study report and graphs</a></p>'
        "<script>setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused&&!v.ended))"
        "location.reload()},30000)</script>"
    )
    (OUT / "index.tmp").write_text(page)
    (OUT / "index.tmp").replace(OUT / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        refresh()
        if args.once:
            break
        time.sleep(args.interval)
