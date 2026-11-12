# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continue matched G1 RFT policies at full material difficulty with recorded profiles."""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/go2_granular/g1/refinement")
INITIAL = Path("logs/rsl_rl/g1_granular_rft_max1/2026-11-12_01-41-04/model_2998.pt")
OFFSET = 2998


def unit_state(unit):
    """Read the durable training process outcome."""
    lines = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    return dict(line.split("=", 1) for line in lines.splitlines() if "=" in line)


def publish(status):
    """Keep live learning curves separate from the immutable baseline comparison."""
    ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    rows = []
    for name, record in status.get("runs", {}).items():
        runs = sorted(Path("logs/rsl_rl", record["family"]).glob("20*"))
        progress = 0
        if runs:
            record["run"] = str(runs[-1])
            events = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0}).Reload()
            tags = events.Tags()["scalars"]
            for ax, tag, label in zip(
                axes,
                ("Train/mean_reward", "Metrics/velocity_error", "Train/mean_episode_length"),
                ("Training return", "Planar error [m/s]", "Episode length [steps]"),
                strict=True,
            ):
                if tag in tags:
                    values = events.Scalars(tag)
                    ax.plot([v.step - OFFSET for v in values], [v.value for v in values], label=name)
                ax.set(xlabel="Additional PPO iteration", ylabel=label)
                ax.grid(alpha=0.2)
            if "Perf/total_fps" in tags:
                progress = events.Scalars("Perf/total_fps")[-1].step + 1 - OFFSET
        record["completed_iterations"] = progress
        rows.append(
            f"<tr><td>{name}</td><td>{progress} / {record['iterations']}</td>"
            f"<td>{html.escape(record.get('stage', 'pending'))}</td></tr>"
        )
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "retraining.png", dpi=150)
    plt.close(fig)
    (ROOT / "training-status.tmp").write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "training-status.tmp").replace(ROOT / "training-status.json")
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:30px auto;padding:20px}"
        "img{width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>"
        '<h1>G1 full-range training continuations</h1><p><a href="./">Force investigation</a></p>'
        "<p>Both start from the same original RFT checkpoint, with a reset optimizer, seed 42 and "
        "4,096 environments. Full reset-randomized material ranges apply from the first episode. "
        "Commands keep 0.1–1.0 m/s throughout. MPM supplies no training transitions.</p>"
        "<p>The control keeps the original force model and randomization. The matched variant is an exploratory "
        "combination of force-informed parameters and wider material-response ranges; it does not isolate each "
        "individual parameter change. Held-out locomotion and MPM evaluations follow training.</p>"
        "<table><tr><th>Variant</th><th>Additional iterations</th><th>Status</th></tr>"
        + "".join(rows)
        + '</table><img src="retraining.png">'
    )
    (ROOT / "training.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "matched"), required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--python_environment", type=Path, help="Separate uv environment for the corrected Newton revision."
    )
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text()) if args.profile else {}
    if args.variant == "matched" and not profile:
        parser.error("The matched run requires its recorded force-informed profile")
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / ("smoke-training-status.json" if args.smoke else "training-status.json")
    status = json.loads(path.read_text()) if path.exists() else dict(runs={})
    label = args.variant + ("-smoke" if args.smoke else "")
    family = "g1_rft_refinement_" + args.variant + ("_smoke" if args.smoke else "")
    unit = "g1-rft-refinement-" + label + "-training.service"
    log = (ROOT / (label + "-training.log")).resolve()
    overrides = ["env.curriculum_enabled=False", "env.material_difficulty=1.0"]
    for group, prefix in (("environment", "env."), ("solver", "env.sim.physics.solver_cfg.")):
        for key, value in profile.get(group, {}).items():
            overrides.append(prefix + key + "=" + json.dumps(value, separators=(",", ":")))
    command = [
        shutil.which("uv"),
        "run",
        "--no-sync",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Velocity-Flat-G1-Granular-RFT-Direct",
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.iterations),
        "--seed",
        "42",
        "--experiment_name",
        family,
        "--checkpoint",
        str(INITIAL),
        *overrides,
    ]
    if args.python_environment:
        command = ["env", f"UV_PROJECT_ENVIRONMENT={args.python_environment.resolve()}", *command]
    record = dict(
        family=family,
        iterations=args.iterations,
        num_envs=args.num_envs,
        initial_checkpoint=str(INITIAL),
        initial_sha256=hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
        optimizer_restored=False,
        command=command,
        profile=profile,
        python_environment=str(args.python_environment) if args.python_environment else None,
        stage="Training",
    )
    status["runs"][label] = record
    state = unit_state(unit)
    if state.get("ActiveState") not in ("active", "activating"):
        launch = [
            "systemd-run",
            "--user",
            "--unit",
            unit.removesuffix(".service"),
            "--property",
            f"WorkingDirectory={Path.cwd()}",
            "--property",
            f"StandardOutput=append:{log}",
            "--property",
            "StandardError=inherit",
            *command,
        ]
        subprocess.run(launch, check=True)
    while unit_state(unit).get("ActiveState") in ("active", "activating"):
        if args.smoke:
            path.write_text(json.dumps(status, indent=2) + "\n")
        else:
            publish(status)
        time.sleep(30)
    state = unit_state(unit)
    if state.get("Result") != "success" or state.get("ExecMainStatus") != "0":
        record["stage"] = "Failed: inspect training log"
        publish(status)
        raise RuntimeError(state)
    run = sorted(Path("logs/rsl_rl", family).glob("20*"))[-1]
    checkpoint = run / f"model_{OFFSET + args.iterations - 1}.pt"
    assert checkpoint.exists(), checkpoint
    record.update(stage="Training complete; held-out validation pending", run=str(run), checkpoint=str(checkpoint))
    if args.smoke:
        path.write_text(json.dumps(status, indent=2) + "\n")
    else:
        publish(status)


if __name__ == "__main__":
    main()
