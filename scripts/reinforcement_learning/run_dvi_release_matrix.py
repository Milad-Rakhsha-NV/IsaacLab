# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Resume the release-3 DVI solver campaign and record 64-world 4K playback.

Run with the editable DVI packages in the active Python environment. Training
uses the archived seed, environment count and PPO budgets. Metrics, commands
and logs are kept separately from the directories containing only videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dvi_release_metrics import world_velocity_metrics
from validate_dvi_release import RECIPES, recipe_overrides

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "logs/dvi_release3_matrix/2026-10-04"
VIDEOS = ROOT / "videos/dvi_release3"
REPORT = ROOT / "release3_dvi_integration_status.md"
EXPORT = Path("/home/mrakhsha/Documents/DVI/Newton-DVI-tech-report/results/reinforcement_learning/release3_2026-10-04")
EXPERIMENT = "release3_dvi_matrix_20261004"
OLD_RUNS = {
    "ant": "2026-10-02_17-41-17_ant_jacobi10/model_999.pt",
    "humanoid": "2026-10-02_17-44-34_humanoid_jacobi10/model_999.pt",
    "anymal_c": "2026-10-02_18-18-01_anymal_c_jacobi20/model_499.pt",
    "go2": "2026-10-02_19-55-05_go2_paper_mdp_jacobi15/model_499.pt",
    "h1": "2026-10-02_18-22-30_h1_jacobi10/model_999.pt",
    "g1": "2026-10-02_18-22-30_g1_jacobi15/model_1499.pt",
    "dr_legs": "2026-10-02_18-25-04_dr_legs_jacobi10/model_999.pt",
}
LOCK = threading.RLock()
STATE = {}


def jacobi_checkpoint(task):
    """Return the current full-budget Jacobi checkpoint for each robot."""
    if task == "go2":
        return ROOT / "logs/rsl_rl" / EXPERIMENT / "2026-10-04_20-22-37_go2_jacobi_coupling2_no_post/model_499.pt"
    return ROOT / "logs/rsl_rl/release3_dvi_validation" / OLD_RUNS[task]


def update(key, **values):
    """Atomically save campaign progress and regenerate its report section."""
    with LOCK:
        STATE.setdefault(key, {}).update(values, updated=datetime.now().isoformat())
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "campaign.json"
        path.with_suffix(".tmp").write_text(json.dumps(STATE, indent=2) + "\n")
        path.with_suffix(".tmp").replace(path)
        EXPORT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, EXPORT / path.name)
        lines = [
            "## October 4–5 full solver campaign",
            "",
            "All new training uses seed 42 and 4096 environments with the archived PPO budgets.",
            "Go2 uses coupling=2 and post-stabilization=false, joint_alpha=0, "
            "and the release standing-height objective.",
            "This Go2 comparison changes two solver settings together as requested; "
            "it is not a single-variable ablation.",
            "APGD uses contact tolerance 1e-4; P-SPG-FB uses 1e-4 for Ant/Humanoid and the task default elsewhere.",
            "Contact iteration budgets remain 10/10/20/15/10/15/10 in the environment order below.",
            "The other six Jacobi policies reuse the completed October 2 training; "
            "Go2 is retrained for 500 iterations.",
            "Playback uses 64 worlds, 3840×2160 H.264, fixed forward commands (0.5 m/s; DR Legs 0.2), and 1200 steps.",
            "Videos retain physical world positions and use a stationary lower oblique camera, "
            "tightly framing the initial formation. "
            "Robots move relative to the fixed ground; no per-robot tracking offsets are applied.",
            "Camera position, angle and zoom remain fixed throughout each recording. "
            "Fast robots can leave the close view; all 64 worlds continue to simulate.",
            "The earlier recentered movies were rejected and replaced; "
            f"[rendering correction]({EXPORT / 'VIDEO_RENDERING_FIX.md'}).",
            "Direct Ant/Humanoid keep their native forward goal. "
            "Commands, checkpoints and raw metrics are in campaign.json.",
            "The table uses world-X speed for direct Ant/Humanoid and body-forward speed for velocity-command tasks.",
            "A separate 4096-world playback measures finite native state and behavior. "
            "Reward scale is not a parity criterion.",
            "Joint-anchor separation is measured over the first world's trajectory; "
            "peaks above 0.02 m are flagged for inspection. "
            "This does not establish a merge regression or sample every world's joint drift.",
            "H1/G1 drift is also observed on the retained paper branches; "
            f"[measurements and limits]({EXPORT / 'PHYSICAL_STATE_LIMITATIONS.md'}).",
            "Each new solver/task result uses the final checkpoint of one seed-42 run. "
            "Earlier training jobs overlapped; following the October 5 correction, "
            "training runs strictly one at a time. "
            "Wall times are not solver throughput comparisons.",
            "Release-3 direct Ant/Humanoid add foot-wrench observations; DVI currently leaves these channels zero. "
            f"See [direct-task compatibility and retained-branch checks]({EXPORT / 'DIRECT_TASK_COMPATIBILITY.md'}).",
            "Videos only: `videos/dvi_release3/2026-10-04_initial_64env/` and "
            "`videos/dvi_release3/2026-10-04_all_solvers_64env/`.",
            "The initial-policy directory now uses the retrained Go2 Jacobi checkpoint. "
            "Its earlier crouching diagnostic clip is retained with the rejected recordings.",
            "",
            "| Run | Status | Forward m/s (4096) | Height m | Premature ends / all ends | Observations | Video |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        rows = []
        for solver in ("jacobi", "apgd", "pspg"):
            for task in RECIPES:
                key = f"{solver}_{task}"
                run = STATE.get(key, {})
                metrics = run.get("metrics4096", {})
                speed = metrics.get(
                    "mean_world_forward_speed_last_half"
                    if task in ("ant", "humanoid")
                    else "mean_forward_speed_last_half"
                )
                height = metrics.get("median_root_height")
                notes = run.get("issues", [])
                lines.append(
                    f"| {key} | {run.get('status', 'pending')} | {speed:.3f}"
                    if speed is not None
                    else f"| {key} | {run.get('status', 'pending')} | —"
                )
                lines[-1] += f" | {height:.3f}" if height is not None else " | —"
                lines[-1] += (
                    f" | {metrics.get('premature_terminations', '—')} / {metrics.get('episode_ends', '—')} "
                    f"| {'; '.join(notes)} |"
                )
                video = run.get("video")
                lines[-1] += f" [4K]({video}) |" if video and run.get("status") == "complete" else " — |"
                rows.append(
                    {
                        "run": key,
                        "status": run.get("status", "pending"),
                        "checkpoint": run.get("checkpoint", ""),
                        "speed": speed,
                        "height": height,
                        "max_joint_anchor_error_first_world_m": metrics.get("max_joint_anchor_error_first_world_m"),
                        "early_ends": metrics.get("premature_terminations"),
                        "episode_ends": metrics.get("episode_ends"),
                        "issues": "; ".join(notes),
                        "video": run.get("video", ""),
                    }
                )
        text = "\n".join(lines) + "\n"
        (EXPORT / "README.md").write_text(text)
        with (EXPORT / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        marker = "<!-- release3-oct4-campaign -->"
        original = REPORT.read_text().split(marker)[0].rstrip()
        REPORT.write_text(original + "\n\n" + marker + "\n" + text)


def run_command(key, command, log):
    """Run one subprocess and persist its exact command before execution."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        commands = list(STATE.get(key, {}).get("commands", []))
    commands.append({"argv": command, "log": str(log)})
    update(key, commands=commands)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {log}")


def evaluate(task, solver, checkpoint, initial=False):
    """Record all worlds and evaluate full training-scale behavior."""
    key = f"initial_{task}" if initial else f"{solver}_{task}"
    directory = OUTPUT / ("initial" if initial else "evaluations")
    video_dir = VIDEOS / ("2026-10-04_initial_64env" if initial else "2026-10-04_all_solvers_64env")
    video = video_dir / f"{task if initial else key}.mp4"
    recipe = RECIPES[task]
    physics = recipe_overrides(recipe, solver)
    update(key, status="evaluating", checkpoint=str(checkpoint), video=str(video), physics_overrides=physics)
    for count in [64] if initial else [64, 4096]:
        metrics = directory / f"{key}_{count}.json"
        if initial and task == "ant" and (directory / "ant.json").exists():
            metrics = directory / "ant.json"
        if count == 64 and metrics.exists():
            cached = json.loads(metrics.read_text())
            recorded = cached.get("video") or {}
            if recorded.get("renderer_version") != "stationary-world-oblique-v4" or cached.get("checkpoint") != str(
                checkpoint.resolve()
            ):
                metrics.replace(metrics.with_suffix(".previous_renderer.json"))
                if metrics.with_suffix(".npz").exists():
                    metrics.with_suffix(".npz").replace(metrics.with_suffix(".previous_renderer.npz"))
        if not metrics.exists() or (count == 64 and not video.exists()):
            command = [
                sys.executable,
                str(ROOT / "scripts/reinforcement_learning/evaluate_dvi_release.py"),
                "--task",
                recipe.task,
                "--checkpoint",
                str(checkpoint),
                "--num_envs",
                str(count),
                "--visualizer",
                "none",
                "--metrics",
                str(metrics),
                "--forward-speed",
                "0.2" if task == "dr_legs" else "0.5",
                *physics,
            ]
            if count == 64:
                command += [
                    "--video-file",
                    str(video),
                    "--video-layout",
                    "grid",
                    "--video-width",
                    "3840",
                    "--video-height",
                    "2160",
                ]
            run_command(key, command, directory / f"{key}_{count}.console.log")
        values = json.loads(metrics.read_text())
        if "mean_world_forward_speed_last_half" not in values:
            import numpy as np  # noqa: PLC0415

            with np.load(metrics.with_suffix(".npz")) as traces:
                values.update(world_velocity_metrics(traces["root_pose"], traces["root_velocity"]))
            metrics.write_text(json.dumps(values, indent=2) + "\n")
        update(key, **{f"metrics{count}": values})
    if not initial:
        values = STATE[key]["metrics4096"]
        issues = []
        if values["premature_terminations"]:
            issues.append("premature terminations observed")
        if values["upright_fraction"] < 0.995:
            issues.append(f"upright fraction {values['upright_fraction']:.4f}")
        anchor_error = values.get("max_joint_anchor_error_first_world_m")
        if anchor_error is not None and anchor_error > 0.02:
            issues.append(f"first-world joint-anchor separation peaks at {anchor_error:.3f} m")
        if task == "go2" and values["median_root_height"] < 0.28:
            issues.append("crouching: median root height below 0.28 m")
        command_speed = values["forward_command"]
        if command_speed and abs(values["mean_forward_speed_last_half"] - command_speed) > command_speed * 0.25:
            issues.append("mean forward tracking error exceeds 25%")
        update(key, issues=issues)
    update(key, status="complete")


def train(task, solver, evaluation_queue):
    """Resume a completed final checkpoint or run the full archived budget."""
    key = f"{solver}_{task}"
    try:
        if STATE.get(key, {}).get("status") == "complete":
            return
        candidates = sorted((ROOT / "logs/rsl_rl" / EXPERIMENT).glob(f"*_{key}"))
        checkpoint = candidates[-1] / f"model_{RECIPES[task].iterations - 1}.pt" if candidates else None
        if checkpoint is None or not checkpoint.exists():
            update(key, status="training", iterations=RECIPES[task].iterations, num_envs=4096, seed=42)
            command = [
                sys.executable,
                str(ROOT / "scripts/reinforcement_learning/validate_dvi_release.py"),
                "--task",
                task,
                "--phase",
                "train",
                "--solver",
                solver,
                "--experiment-name",
                EXPERIMENT,
                "--run-name",
                key,
                "--output",
                str(OUTPUT / "training" / key),
            ]
            run_command(key, command, OUTPUT / "training" / f"{key}.launcher.log")
            candidates = sorted((ROOT / "logs/rsl_rl" / EXPERIMENT).glob(f"*_{key}"))
            checkpoint = candidates[-1] / f"model_{RECIPES[task].iterations - 1}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing final checkpoint: {checkpoint}")
        update(key, status="trained", checkpoint=str(checkpoint))
        evaluation_queue.put((task, solver, checkpoint, False))
    except Exception as error:
        update(key, status="failed", issues=[str(error)])


def main():
    """Keep training and rendering bounded and independently resumable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initial", "matrix", "videos"], required=True)
    parser.add_argument("--workers", type=int, choices=[1], default=1, help="Training runs strictly one at a time")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = OUTPUT / "campaign.json"
    if manifest.exists():
        STATE.update(json.loads(manifest.read_text()))
    if args.phase == "videos":
        for key, run in list(STATE.items()):
            if run.get("status") != "complete" or not run.get("checkpoint"):
                continue
            solver, task = key.split("_", 1)
            checkpoint = jacobi_checkpoint(task) if solver == "initial" else Path(run["checkpoint"])
            evaluate(task, "jacobi" if solver == "initial" else solver, checkpoint, solver == "initial")
        return
    if args.phase == "initial":
        for task in RECIPES:
            if STATE.get(f"initial_{task}", {}).get("status") == "complete":
                continue
            try:
                evaluate(task, "jacobi", jacobi_checkpoint(task), True)
            except Exception as error:
                update(f"initial_{task}", status="failed", issues=[str(error)])
        return
    evaluation_queue = queue.Queue()

    def evaluator():
        while True:
            job = evaluation_queue.get()
            if job is None:
                return
            task, solver, checkpoint, initial = job
            try:
                evaluate(task, solver, checkpoint, initial)
            except Exception as error:
                update(f"initial_{task}" if initial else f"{solver}_{task}", status="failed", issues=[str(error)])

    worker = threading.Thread(target=evaluator)
    worker.start()
    for task in ["go2", *[name for name in RECIPES if name != "go2"]]:
        key = f"jacobi_{task}"
        if STATE.get(key, {}).get("status") == "complete":
            continue
        checkpoint = jacobi_checkpoint(task)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluation_queue.put((task, "jacobi", checkpoint, False))
    for task in RECIPES:
        if STATE.get(f"initial_{task}", {}).get("status") != "complete":
            evaluation_queue.put((task, "jacobi", jacobi_checkpoint(task), True))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(train, task, solver, evaluation_queue) for task in RECIPES for solver in ("apgd", "pspg")
        ]
        for future in futures:
            future.result()
    evaluation_queue.put(None)
    worker.join()
    failures = [key for key, value in STATE.items() if value.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"Failed runs: {failures}")


if __name__ == "__main__":
    main()
