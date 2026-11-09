# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Snapshot experiment facts and create the tutorial's static vector figures."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def snapshot():
    root = HERE.parents[3]
    run = root / "projects/supermimic/runs/lafan1_full"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    evaluations = [row for row in rows if row["phase"] == "evaluation"]
    final = next(row for row in reversed(rows) if row["phase"] == "finished")
    with np.load(root / "projects/supermimic/data/lafan1.npz") as data:
        lengths = dict(zip(data["names"].tolist(), data["lengths"].tolist(), strict=True))
        samples = {}
        for name in ("walk1_subject1", "dance2_subject1"):
            index = data["names"].tolist().index(name)
            frame = int(data["starts"][index]) + 250
            # Canonical left knee is joint index 3, starting at state component 13.
            samples[name] = data["states"][frame : frame + 46, 16].tolist()
    return {
        "config": json.loads((run / "config.json").read_text()),
        "readiness": json.loads((run / "readiness.json").read_text()),
        "evaluations": evaluations,
        "final": final,
        "lengths": lengths,
        "joint_preview_samples": samples,
        "videos": json.loads((run / "videos/all_clips/validation.json").read_text()),
        "metrics_sha256": hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace the saved snapshot from local experiment files")
    args = parser.parse_args()
    path = HERE / "report_data.json"
    if args.refresh:
        path.write_text(json.dumps(snapshot(), indent=2) + "\n")
    data = json.loads(path.read_text())
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    evaluations = data["evaluations"]
    start = evaluations[0]["time_unix"]
    hours = [(row["time_unix"] - start) / 3600 for row in evaluations]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot(hours, [row["normalized_return"] for row in evaluations], color="#2563eb", lw=1.2)
    axes[0].set(
        ylabel="Normalized return",
        ylim=(0, 1),
        title="Actual simulator evaluation: five-second windows, 40 training clips",
    )
    axes[1].plot(hours, [row["failed_clips"] for row in evaluations], color="#ea580c", lw=1.2)
    axes[1].set(
        xlabel="Hours since first full-run evaluation (already warm-started)",
        ylabel="Failed windows",
        ylim=(-0.5, 40.5),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(directory / "training_results.pdf")
    plt.close(fig)
    final = data["final"]["per_clip"]
    names = sorted(final, key=final.get)
    fig, ax = plt.subplots(figsize=(7.0, 8.2), constrained_layout=True)
    ax.barh(names, [final[name] for name in names], color="#2563eb", height=0.7)
    ax.set(xlim=(0, 1), xlabel="Final normalized return: five-second first-frame evaluation")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(directory / "final_motion_returns.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 2.9), constrained_layout=True)
    time = np.arange(46) / 50
    for (name, values), color in zip(data["joint_preview_samples"].items(), ("#2563eb", "#ea580c"), strict=True):
        ax.plot(time, values, color=color, lw=1.5, label=name)
        ax.scatter(time[::5], np.array(values)[::5], color=color, s=24, zorder=3)
    ax.set(xlabel="Look-ahead relative to current reference frame (seconds)", ylabel="Reference left knee (rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(directory / "goal_preview.pdf")
    plt.close(fig)
    rows = [
        r"\begin{longtable}{@{}p{0.57\linewidth}rr@{}}",
        r"\toprule Motion & Clip seconds & Final return\\\midrule\endhead",
    ]
    for name in sorted(final):
        escaped = name.replace("_", r"\_")
        rows.append(f"{escaped} & {data['lengths'][name] / 50:.2f} & {final[name]:.4f} " + r"\\")
    rows.extend([r"\bottomrule", r"\end{longtable}"])
    (HERE / "motion_table.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps({"figures": 3, "clips": len(final), "final_iteration": data["final"]["iteration"]}))


if __name__ == "__main__":
    main()
