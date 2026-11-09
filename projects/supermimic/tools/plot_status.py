# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Publish a Markdown status report and PNG plots from a running experiment's JSONL log."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock

TIMEZONE = ZoneInfo("America/Los_Angeles")
COLORS = ["#2563eb", "#ea580c", "#059669", "#9333ea"]


def read_records(path):
    """Tolerate an incomplete final write, but surface corrupted complete records."""
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                break
            raise ValueError(f"Invalid JSON in {path}, line {index + 1}") from None
    return records


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {remainder % 60:02d}s"


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def process_alive(pid, run):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Avoid mistaking a recycled Linux PID for this training process.
    command_path = Path(f"/proc/{pid}/cmdline")
    if command_path.exists():
        command = command_path.read_bytes()
        return b"projects.supermimic.train" in command and str(run).encode() in command
    return True


def series(ax, rows, key, label, color, smooth=True):
    data = [(row["iteration"], row[key]) for row in rows if key in row]
    if not data:
        ax.text(0.5, 0.5, "Waiting for samples", transform=ax.transAxes, ha="center")
        return
    x, y = np.array(data).T
    window = min(11, len(y)) if smooth else 1
    ax.plot(x, y, color=color, alpha=0.25 if window > 1 else 1, linewidth=1)
    ax.plot(
        x[window - 1 :], np.convolve(y, np.ones(window) / window, mode="valid"), color=color, label=label, linewidth=1.8
    )


def finish_figure(fig, path, title, timestamp):
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.07, ha="left")
    fig.text(0.99, 0.008, f"Updated {local_time(timestamp)}", ha="right", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    temporary = path.with_suffix(".tmp")
    fig.savefig(temporary, format="png", dpi=145, facecolor="white")
    temporary.replace(path)
    plt.close(fig)


def learning_plot(training, evaluations, directory, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    if evaluations:
        x = [r["iteration"] for r in evaluations]
        axes[0, 0].plot(x, [r["normalized_return"] for r in evaluations], "o-", color=COLORS[0])
        axes[0, 1].plot(x, [r["failed_clips"] for r in evaluations], "o-", color=COLORS[1])
        axes[0, 1].set_ylim(-0.5, evaluations[-1]["clips"] + 0.5)
    axes[0, 0].set(title="Simulator evaluation return · higher is better", ylabel="Normalized return", ylim=(0, 1))
    axes[0, 1].set(title="Failed evaluation windows · lower is better", ylabel="Failed motions")
    series(axes[1, 0], training, "sim_reward", "Simulator collection", COLORS[0])
    series(axes[1, 0], training, "imagined_reward", "Imagined rollouts", COLORS[1])
    axes[1, 0].set(title="Step rewards · different rollout distributions", ylabel="Mean reward / 20 ms step")
    axes[1, 0].legend(fontsize=9)
    series(axes[1, 1], training, "sim_failure_fraction", "Simulator step failure fraction", COLORS[1])
    series(axes[1, 1], training, "imagined_survival", "Survival to imagined horizon", COLORS[2])
    axes[1, 1].set(title="Termination and imagined survival", ylabel="Fraction", ylim=(-0.02, 1.02))
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("Full-run simulator iteration (warm start precedes iteration 0)")
        ax.grid(alpha=0.2)
    finish_figure(fig, directory / "learning.png", "SuperMimic · learning progress", timestamp)


def optimization_plot(training, directory, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key, title, color in (
        (axes[0, 0], "model_loss", "World-model autoregressive training loss", COLORS[0]),
        (axes[0, 1], "critic_loss", "Critic lambda-return fitting loss", COLORS[2]),
        (axes[1, 0], "actor_loss", "Actor objective (signed loss)", COLORS[3]),
    ):
        series(ax, training, key, key, color)
        ax.set_title(title)
        ax.set_ylabel("Loss")
    for ax, key in ((axes[0, 0], "model_loss"), (axes[0, 1], "critic_loss")):
        if any(row.get(key, 0) > 0 for row in training):
            ax.set_yscale("log")
    series(axes[1, 1], training, "model_gradient_norm", "World model (before clipping)", COLORS[0])
    series(axes[1, 1], training, "actor_gradient_norm", "Actor (before clipping)", COLORS[3])
    axes[1, 1].axhline(1, color="#64748b", linestyle="--", linewidth=1, label="Global norm clipping threshold")
    axes[1, 1].set(title="Gradient norms", ylabel="Global norm", yscale="log")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Full-run simulator iteration")
        ax.grid(alpha=0.2)
    finish_figure(fig, directory / "optimization.png", "SuperMimic · optimization diagnostics", timestamp)


def motion_plot(evaluations, directory, timestamp):
    if not evaluations:
        return
    first, latest = evaluations[0], evaluations[-1]
    scores = latest.get("per_clip", {})
    if not scores:
        return
    names = sorted(scores, key=scores.get)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.28)))
    ax.barh(
        y,
        [scores[n] for n in names],
        color=COLORS[0],
        alpha=0.85,
        label=f"Latest evaluation · iteration {latest['iteration']:,}",
    )
    ax.scatter(
        [first["per_clip"].get(n, np.nan) for n in names],
        y,
        color=COLORS[1],
        s=18,
        zorder=3,
        label="Full-run starting checkpoint",
    )
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1), xlabel="Normalized evaluation-window return · higher is better")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    finish_figure(fig, directory / "motions.png", "SuperMimic · per-motion tracking", timestamp)


def render(run, output, interval):
    timestamp = time.time()
    metadata = json.loads((run / "launch.json").read_text())
    config = json.loads((run / "config.json").read_text())["learner"]
    records = read_records(run / "metrics.jsonl")
    training = [r for r in records if r["phase"] == "training"]
    evaluations = [r for r in records if r["phase"] in ("evaluation", "finished") and "normalized_return" in r]
    latest = training[-1] if training else {"phase": "initializing", "iteration": 0, "time_unix": timestamp}
    last = records[-1] if records else latest
    target, iteration = metadata["target_iterations"], latest["iteration"]
    alive = process_alive(metadata["pid"], run)
    if last["phase"] == "failed":
        status = f"FAILED — {last.get('error', 'see training log')}"
    elif last["phase"] == "finished":
        status = "COMPLETED" if last.get("iteration", 0) >= target else "STOPPED"
        iteration = last.get("iteration", iteration)
    else:
        status = "RUNNING" if alive else "NO ACTIVE TRAINING PROCESS"
    recent = []
    for row in training:
        if recent and row["iteration"] < recent[-1]["iteration"]:
            recent = []
        recent.append(row)
    recent = [r for r in recent if latest["time_unix"] - r["time_unix"] <= 300]
    rate = 0
    if len(recent) >= 2:
        rate = (recent[-1]["iteration"] - recent[0]["iteration"]) / max(
            1, recent[-1]["time_unix"] - recent[0]["time_unix"]
        )
    eta = max(0, target - iteration) / rate if rate > 0 else None
    figures = run / "monitor"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    learning_plot(training, evaluations, figures, timestamp)
    optimization_plot(training, figures, timestamp)
    motion_plot(evaluations, figures, timestamp)
    output.parent.mkdir(parents=True, exist_ok=True)

    def link(path):
        return Path(os.path.relpath(path, output.parent)).as_posix()

    checkpoint = run / "last.pt"
    checkpoint_text = (
        f"{local_time(checkpoint.stat().st_mtime)} ({checkpoint.stat().st_size / 2**20:.1f} MiB)"
        if checkpoint.exists()
        else "Not yet written"
    )
    launch_time = datetime.fromisoformat(metadata["started_local"]).timestamp()
    lines = [
        "# SuperMimic training status",
        "",
        f"**{status}** · updated {local_time(timestamp)} · refresh every {interval:g} seconds",
        "",
        "| Metric | Current value |",
        "|---|---|",
        "| Experiment | LAFAN1, 40 G1 clips, fixed physics, Newton/MJWarp |",
        f"| Progress | {iteration:,} / {target:,} iterations ({100 * iteration / target:.2f}%) |",
        f"| Actor updates, including readiness | {latest.get('updates', 'pending'):,} |"
        if "updates" in latest
        else "| Actor updates | Pending |",
        f"| Training transitions after warm start | {iteration * config['num_envs']:,} |",
        f"| Time since launch | {duration(timestamp - launch_time)} |",
        f"| Recent throughput | {rate:.2f} simulator iterations/s |",
        f"| Estimated remaining time | {duration(eta) if eta is not None else 'Waiting for samples'} |",
        f"| Training log freshness | {duration(timestamp - last['time_unix'])} |",
        f"| Process | PID {metadata['pid']} · {'alive' if alive else 'not running'} |",
        f"| Latest checkpoint modified | {checkpoint_text} |",
        f"| Peak PyTorch GPU allocation | {latest.get('peak_allocated_gib', 0):.2f} GiB |",
        "",
    ]
    if evaluations:
        evaluation = evaluations[-1]
        evaluation_scope = (
            "complete clips"
            if evaluation.get("complete_clips")
            else f"{evaluation['horizon'] / 50:g}-second first-frame windows"
        )
        lines += [
            f"Latest evaluation: **return {evaluation['normalized_return']:.4f}**, "
            f"**{evaluation['failed_clips']} / {evaluation['clips']} failed windows**, "
            f"at iteration **{evaluation['iteration']:,}** ({local_time(evaluation['time_unix'])}).",
            "",
            f"Evaluation uses {evaluation_scope}. "
            "These are training-set coverage diagnostics. Iteration zero already includes readiness pretraining.",
            "",
        ]
    video_directory = run / "videos" / "all_clips"
    video_metadata_path = video_directory / "metadata.json"
    if video_metadata_path.exists():
        video_metadata = json.loads(video_metadata_path.read_text())
        videos = video_metadata["videos"]
        failed = sum(video["first_failure_seconds"] is not None for video in videos)
        lines += [
            "## Policy videos",
            "",
            f"**{len(videos)} / {len(video_metadata['requested_clips'])} videos available**, "
            f"from checkpoint **{video_metadata['checkpoint_iteration']:,}**. "
            f"Each requested rollout is {video_metadata['requested_duration_seconds']:g} seconds. "
            f"Tracking thresholds exceeded in **{failed} / {len(videos)}** completed recordings; "
            "failures are marked on screen and do not trigger resets.",
            "",
            f"[Open video gallery]({link(video_directory / 'index.html')}) · "
            f"[Browse individual videos]({link(video_directory / 'README.md')})",
            "",
        ]
        archive = video_directory / "supermimic_all_clips.zip"
        if archive.exists():
            lines += [f"[Download all videos]({link(archive)})", ""]
    lines += [
        "## Learning progress",
        "",
        f"![Learning progress]({link(figures / 'learning.png')})",
        "",
        "Raw training samples are faint; solid lines show an 11-sample moving mean where available. "
        "Simulator failure fraction is measured per control step, not per episode. "
        "Simulator and imagined rewards sample different state distributions; their gap is not a model-error estimate.",
        "",
        "## Optimization diagnostics",
        "",
        f"![Optimization diagnostics]({link(figures / 'optimization.png')})",
        "",
        "Model loss is the online autoregressive training loss, not held-out dynamics error. "
        "Actor loss includes a critic bootstrap and survival weighting, so it is not simulator tracking return. "
        "Gradient norms are logged before clipping.",
        "",
    ]
    if evaluations and evaluations[-1].get("per_clip"):
        lines += [
            "## Motion coverage",
            "",
            f"![Per-motion evaluation]({link(figures / 'motions.png')})",
            "",
            "Motions are sorted by their latest return. "
            "Orange points mark the checkpoint at the start of the full run.",
            "",
        ]
    lines += [
        "## Monitoring",
        "",
        "The background plotter rewrites this file and its figures once per minute. "
        "Reload the Markdown preview to see the latest images. "
        "ETA uses the last five minutes of logged progress "
        "and can change with evaluation or compilation costs.",
        "",
        f"[Metrics JSONL]({link(run / 'metrics.jsonl')}) · [Training log]({link(run / 'train.log')}) · "
        f"[Readiness report]({link(run / 'readiness.json')}) · [Launch metadata]({link(run / 'launch.json')})",
        "",
        "Refresh once from the IsaacLab checkout:",
        "",
        "```bash",
        f"uv run --no-sync python projects/supermimic/tools/plot_status.py --run {link(run)} --output {output.name}",
        "```",
        "",
    ]
    temporary = output.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(output)
    print(
        json.dumps({"status": status, "iteration": iteration, "output": str(output), "updated": local_time(timestamp)}),
        flush=True,
    )
    return alive and last["phase"] not in ("failed", "finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("projects/supermimic/runs/lafan1_full"))
    parser.add_argument("--output", type=Path, default=Path("status.md"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("Refresh interval must be at least five seconds")
    monitor = args.run.resolve() / "monitor"
    monitor.mkdir(exist_ok=True)
    if args.watch:
        (monitor / "plotter.pid").write_text(str(os.getpid()) + "\n")
    while True:
        with FileLock(str(monitor / "refresh.lock"), timeout=30):
            running = render(args.run.resolve(), args.output.resolve(), args.interval)
        if not args.watch or not running:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
