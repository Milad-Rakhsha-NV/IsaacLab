# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the research report from its included result snapshots; no simulation or training."""

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "data/imitation_summary.json").read_text())
PPO = json.loads((ROOT / "data/ppo_comparison.json").read_text())
COLORS = {"teacher": "#526477", "attention": "#007C83", "graph": "#CF7B35", "ppo": "#A7B3BE"}


def evaluation(key):
    return RESULTS[key] if key == "teacher" else RESULTS[key]["evaluation"]


def make_tables():
    generated = ROOT / "generated"
    generated.mkdir(exist_ok=True)
    rows = []
    for label, three, four in [
        ("Teacher MLPs", "teacher", "teacher"),
        ("MLP imitation control", "mlp_joint", "mlp_joint"),
        ("Graph, separate per body", "graph_ant_3", "graph_ant_4"),
        ("Graph, shared imitation", "graph_joint", "graph_joint"),
        ("Attention, shared imitation", "attention_joint", "attention_joint"),
    ]:
        a, b = evaluation(three)["ant_3"], evaluation(four)["ant_4"]
        rows.append(
            f"{label} & {a['mean_speed_m_s']:.3f} & {100 * a['mean_survival']:.1f}"
            f" & {b['mean_speed_m_s']:.3f} & {100 * b['mean_survival']:.1f} \\\\"
        )
    (generated / "source_results.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for label, key in [
        ("MLP imitation", "mlp_joint"),
        ("Shared graph", "graph_joint"),
        ("Shared attention", "attention_joint"),
    ]:
        a, b = RESULTS[key]["test"]["ant_3"], RESULTS[key]["test"]["ant_4"]
        rows.append(
            f"{label} & {a['raw_rmse']:.3f} & {100 * a['relative_rmse']:.1f}"
            f" & {b['raw_rmse']:.3f} & {100 * b['relative_rmse']:.1f} \\\\"
        )
    (generated / "imitation_errors.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for population in ("canonical", "diverse"):
        for label, routing in [("Attention", "attention"), ("Graph", "message_passing"), ("Hybrid pooling", "pooled")]:
            d = PPO[f"{routing}_{population}_seed42"]
            cells = [str(2 if population == "canonical" else 8), label]
            cells.extend(
                f"{d[body]['mean_speed_m_s']:.3f} / {100 * d[body]['mean_survival']:.1f}"
                for body in ("ant_3", "ant_4", "dev_5")
            )
            rows.append(" & ".join(cells) + " \\\\")
    (generated / "ppo_results.tex").write_text("\n".join(rows) + "\n")
    fields = json.loads((ROOT / "data/static_fields.json").read_text())
    assert sum(fields.values()) == 104
    rows = []
    offset = 0
    for name, width in fields.items():
        rows.append(f"\\path{{{name}}} & {width} & {offset}--{offset + width - 1} \\\\")
        offset += width
    (generated / "static_fields.tex").write_text("\n".join(rows) + "\n")
    transfer = json.loads((ROOT / "data/transfer_summary.json").read_text())["results"]
    rows = []
    for body, label in (("dev_5", "Five-leg Ant"), ("go2", "Unitree Go2"), ("a1", "Unitree A1")):
        for control in ("attention", "graph"):
            cell = transfer[body][control]
            rows.append(
                f"{label} & {control.title()} & {cell['progress_speed_m_s']['mean']:.3f}"
                f" & {100 * cell['survival']['mean']:.1f} & {cell['length_s']['mean']:.3f} \\\\"
            )
    (generated / "transfer_results.tex").write_text("\n".join(rows) + "\n")
    learning = json.loads((ROOT / "data/go2_learning_summary.json").read_text())
    rows = []
    for body, label, robot in (
        ("ant_3", "Three-leg Ant", "ant_sources"),
        ("ant_4", "Four-leg Ant", "ant_sources"),
        ("go2", "Go2 (new source)", "go2"),
        ("a1", "A1 (development)", "a1"),
        ("dev_5", "Five-leg Ant (development)", "dev_5"),
    ):
        shared = learning["students"]["shared_ant_go2"]["evaluations"][robot]["results"][body]
        if body in ("ant_3", "ant_4"):
            ref = learning["ant_references"]["teacher"][body]
            teacher_cell = f"{ref['mean_speed_m_s']:.3f} / {100 * ref['mean_survival']:.1f}"
        elif body == "go2":
            ref = learning["teacher1500"]["results"][body]
            teacher_cell = f"{ref['progress_speed_m_s']['mean']:.3f} / {100 * ref['survival']['mean']:.1f}"
        else:
            teacher_cell = "---"
        solo_cell = "---"
        if body in ("go2", "a1"):
            solo = learning["students"]["go2_only"]["evaluations"][robot]["results"][body]
            solo_cell = f"{solo['progress_speed_m_s']['mean']:.3f} / {100 * solo['survival']['mean']:.1f}"
        shared_cell = f"{shared['progress_speed_m_s']['mean']:.3f} / {100 * shared['survival']['mean']:.1f}"
        rows.append(f"{label} & {teacher_cell} & {solo_cell} & {shared_cell} \\\\")
    (generated / "go2_learning_results.tex").write_text("\n".join(rows) + "\n")


def make_figures():
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    methods = [
        ("Teacher", evaluation("teacher"), COLORS["teacher"]),
        ("Distilled attention", evaluation("attention_joint"), COLORS["attention"]),
        ("Distilled graph", evaluation("graph_joint"), COLORS["graph"]),
        ("Attention: scratch PPO", PPO["attention_canonical_seed42"], COLORS["ppo"]),
    ]
    for ax, body, title in zip(axes, ("ant_3", "ant_4"), ("Three-leg source", "Four-leg source")):
        for i, (label, data, color) in enumerate(methods):
            mean = data[body]["mean_speed_m_s"]
            low, high = data[body]["speed_range_m_s"]
            ax.barh(i, mean, color=color, height=0.62)
            ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="none", ecolor="#172C3B", capsize=3, lw=1)
            ax.text(mean + 0.12, i, f"{mean:.2f}", va="center", fontsize=9)
        ax.set_yticks(range(4), [m[0] for m in methods] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(0, 6.4)
        ax.set_xlabel("Forward speed (m/s)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#DBE2E8", alpha=0.7)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures/source_speed.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    for ax, body, title in zip(axes, ("ant_3", "ant_4"), ("Three-leg source", "Four-leg source")):
        for i, (label, data, color) in enumerate(methods[:3]):
            failure = 100 * (1 - data[body]["mean_survival"])
            ax.barh(i, failure, color=color, height=0.6)
            ax.text(failure + 0.13, i, f"{failure:.2f}%", va="center", fontsize=9)
        ax.set_yticks(range(3), [m[0] for m in methods[:3]] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(0, 8)
        ax.set_xlabel("Episodes ending before the time limit (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#DBE2E8", alpha=0.7)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures/source_failure.pdf", bbox_inches="tight")
    plt.close(fig)


def make_go2_diagnostic_figure():
    reports = json.loads((ROOT / "data/go2_diagnostics.json").read_text())["reports"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, name, title in zip(axes, ("go2_only", "shared_ant_go2"), ("Go2-only student", "Shared Ant+Go2 student")):
        values = [reports[f"audit_{name}_{behavior}"]["raw_rmse"] for behavior in ("teacher", "student")]
        ax.bar([0, 1], values, color=[COLORS["attention"], COLORS["graph"]], width=0.55)
        for i, value in enumerate(values):
            ax.text(i, value + 0.008, f"{value:.3f}", ha="center")
        ax.set_xticks([0, 1], ["Teacher-visited states", "Student-visited states"])
        ax.set_ylim(0, 0.33)
        ax.set_ylabel("Raw action RMSE against teacher")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#DBE2E8", alpha=0.7)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures/go2_rollout_error.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    make_tables()
    make_figures()
    make_go2_diagnostic_figure()
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=.build",
            "cross_embodiment_report.tex",
        ],
        cwd=ROOT,
        check=True,
    )
    destination = ROOT / "cross_embodiment_report.pdf"
    shutil.copy2(build / destination.name, destination)
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
