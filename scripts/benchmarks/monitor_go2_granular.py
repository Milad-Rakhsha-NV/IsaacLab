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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular")
OUT = ROOT / "live"
RUNS = (
    ("rft", "RFT: material curriculum", "go2_rft_material_curriculum", "go2-rft-material-training.service", 1500),
    ("rigid", "Rigid-ground control", "go2_rigid_infinite", "go2-rigid-plane-training.service", 600),
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
    handles, _ = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("4,096 environments per run · full commands from the start · no MPM training")
    fig.tight_layout()
    fig.savefig(OUT / "curves.tmp", format="png", dpi=130)
    plt.close(fig)
    (OUT / "curves.tmp").replace(OUT / "curves.png")
    snapshot = dict(updated_utc=now.isoformat(), runs=rows)
    (OUT / "status.tmp").write_text(json.dumps(snapshot, indent=2))
    (OUT / "status.tmp").replace(OUT / "status.json")
    cards = []
    for row in rows:
        metrics = row["metrics"]
        difficulty = metrics.get("Curriculum/difficulty", 0)
        error = metrics.get("Metrics/velocity_error")
        error_text = f"{error:.3f} m/s" if error is not None else "Waiting for episodes"
        status = row["state"].get("ActiveState", "not started")
        if row["state"].get("Result") not in (None, "success"):
            status += f" ({row['state']['Result']})"
        material = f"{difficulty:.0%} of target randomization width" if row["key"] == "rft" else "Rigid plane"
        cards.append(
            f"<article><h2>{html.escape(row['label'])}</h2><p><b>{html.escape(status)}</b> · "
            f"{row['completed_iterations']:,} / {row['target_iterations']:,} iterations</p>"
            f"<progress value='{row['completed_iterations']}' max='{row['target_iterations']}'></progress>"
            f"<p>{row['transitions'] / 1e6:.1f} million transitions · {error_text}<br>{material}<br>"
            f"Last checkpoint: {row['checkpoint']} · Event age: {row['event_age_seconds']} s</p>"
            f"<a href='../training/{row['key']}.log'>Training log</a></article>"
        )
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Go2 live training</title>'
        "<style>body{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1200px;margin:24px auto;"
        "padding:16px}section{display:flex;gap:20px;flex-wrap:wrap}article{background:white;padding:20px;"
        "border-radius:12px;flex:1;min-width:280px}progress,img{width:100%}a{color:#1d4ed8}</style>"
        f"<h1>Go2 live training</h1><p>Updates every 30 seconds. Snapshot: {now:%Y-%m-%d %H:%M:%S} UTC.</p>"
        "<p>Both runs: 4,096 environments, seed 42, full commands from the first reset. "
        "Only RFT material randomization has a curriculum. MPM is used after training for inference.</p>"
        f"<section>{''.join(cards)}</section><img src='curves.png?t={int(time.time())}' alt='Live learning curves'>"
        "<p>Material progression requires ≥95% episode completion and ≤0.10 m/s mean tracking error in "
        "two consecutive windows of at least 4,096 qualifying episodes. Training curves do not establish "
        "held-out convergence or MPM transfer.</p><p>Full target: density 1,400–1,800 kg/m³; friction 0.3–0.5; "
        "roughness 0.6–1.0; strength multiplier 0.8–1.2; surface elevation ±0.02 m. "
        "Commands: forward 0.1–0.5 m/s, lateral ±0.1 m/s, yaw ±0.3 rad/s; 10% standing.</p>"
        '<p><a href="../report/">Videos and study report</a> · <a href="status.json">Raw live metrics</a></p>'
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
