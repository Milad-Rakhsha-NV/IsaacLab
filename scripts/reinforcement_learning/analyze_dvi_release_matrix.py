# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export measured training curves and behavior from the release-3 campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_dvi_release_matrix import EXPORT, OUTPUT, RECIPES, ROOT  # noqa: E402

LABELS = {"jacobi": "Jacobi", "apgd": "APGD", "pspg": "P-SPG-FB"}
COLORS = {"jacobi": "#1875aa", "apgd": "#e68827", "pspg": "#278c57"}


def capture_final_source():
    """Preserve the working source without changing either repository's index."""
    snapshot = EXPORT / "source_snapshot/final_source"
    snapshot.mkdir(parents=True, exist_ok=True)
    manifest = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "repositories": {}}
    for name in ("isaaclab-dvi", "newton-dvi"):
        repository = ROOT.parent / name

        def git(*args):
            return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True).stdout

        patch = git("diff", "--binary", "HEAD")
        (snapshot / f"{name}.patch").write_bytes(patch)
        manifest["repositories"][name] = {
            "head": git("rev-parse", "HEAD").decode().strip(),
            "branch": git("branch", "--show-current").decode().strip(),
            "status": git("status", "--short").decode(),
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "index_diff_sha256": hashlib.sha256(git("diff", "--cached", "--binary", "HEAD")).hexdigest(),
        }
    for relative in (
        "newton/tests/test_dvi_target_layout.py",
        "changelog/+dvi-coordinate-targets-0b46d7a3.fixed.md",
    ):
        source = ROOT.parent / "newton-dvi" / relative
        destination = snapshot / "newton-dvi" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_training_configuration(key, checkpoint, recipe):
    """Compare saved training parameters with the requested solver recipe."""
    solver, task = key.split("_", 1)
    # BaseLoader reads the saved Python tuple tags as data, without
    # importing or constructing objects from the configuration file.
    cfg = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    physics = cfg["sim"]["physics"]["solver_cfg"]
    expected_solver = {"jacobi": "sparse_jacobi", "apgd": "sparse_apgd", "pspg": "sparse_pspg"}[solver]
    if physics["contact_solver_type"] != expected_solver or int(cfg["scene"]["num_envs"]) != 4096:
        raise RuntimeError(f"Wrong solver or training scale: {key}")
    if int(cfg["seed"]) != 42 or int(cfg["sim"]["physics"]["num_substeps"]) != recipe.substeps:
        raise RuntimeError(f"Wrong seed or substeps: {key}")
    if int(physics["contact_max_iterations"]) != recipe.contacts:
        raise RuntimeError(f"Wrong contact iteration budget: {key}")
    if physics["cache_factorization"] != "true" or physics["actuator_integration"] != "semi_implicit":
        raise RuntimeError(f"Wrong actuation or factorization settings: {key}")
    if physics["contact_friction_projection"] != "tangential":
        raise RuntimeError(f"Wrong friction projection: {key}")
    if solver == "apgd" or (solver == "pspg" and task in ("ant", "humanoid")):
        if float(physics["contact_tolerance"]) != 1e-4:
            raise RuntimeError(f"Wrong contact tolerance: {key}")
    elif solver == "pspg" and physics["contact_tolerance"] != "null":
        raise RuntimeError(f"Changed task-default P-SPG-FB tolerance: {key}")
    if int(physics["coupling_iterations"]) != recipe.coupling:
        raise RuntimeError(f"Wrong coupling iterations: {key}")
    if (physics["post_stabilize_joints"] == "true") != recipe.post_stabilize:
        raise RuntimeError(f"Wrong post-stabilization: {key}")


def verify_deliverables(state, curves):
    """Check final budgets, saved native-state traces, and movie metadata."""
    expected = {f"{solver}_{task}" for solver in LABELS for task in RECIPES}
    if not all(state.get(key, {}).get("status") == "complete" for key in expected):
        return
    verified = []
    for key in sorted(expected):
        run = state[key]
        task = key.split("_", 1)[1]
        recipe = RECIPES[task]
        checkpoint = Path(run["checkpoint"])
        if checkpoint.name != f"model_{recipe.iterations - 1}.pt":
            raise RuntimeError(f"Wrong final checkpoint: {key}, {checkpoint}")
        archived = EXPORT / "raw" / key / checkpoint.name
        digest = archived.with_suffix(archived.suffix + ".sha256").read_text().strip()
        if hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Wrong archived checkpoint checksum: {key}")
        points = curves[key]["Train/mean_reward"]
        if points["iteration"][-1] != recipe.iterations - 1:
            raise RuntimeError(f"Incomplete training curve: {key}")
        verify_training_configuration(key, checkpoint, recipe)
        for count in (64, 4096):
            metrics = run[f"metrics{count}"]
            if not metrics["finite_native_state"] or metrics["num_envs"] != count or metrics["steps"] != 1200:
                raise RuntimeError(f"Incomplete native-state playback: {key}, {count}")
            with np.load(OUTPUT / "evaluations" / f"{key}_{count}.npz") as traces:
                for field in ("root_pose", "root_velocity", "projected_gravity", "body_pose"):
                    if not np.isfinite(traces[field]).all():
                        raise RuntimeError(f"Non-finite trace: {key}, {count}, {field}")
                if traces["root_pose"].shape[:2] != (1200, count):
                    raise RuntimeError(f"Incomplete pose trace: {key}, {count}")
        recording = run["metrics64"]["video"]
        if recording["visible_worlds"] != 64 or recording["renderer_version"] != "stationary-world-oblique-v4":
            raise RuntimeError(f"Wrong movie layout: {key}")
        if recording["root_centered_grid"] or recording["world_offsets"] != "zero":
            raise RuntimeError(f"Movie hides physical translation: {key}")
        if not recording["camera_stationary"]:
            raise RuntimeError(f"Movie moves the camera: {key}")
        reader = imageio_ffmpeg.read_frames(run["video"])
        metadata = next(reader)
        reader.close()
        if metadata["size"] != (3840, 2160):
            raise RuntimeError(f"Wrong movie resolution: {key}")
        if abs(metadata["duration"] - run["metrics64"]["duration_seconds"]) > 0.1:
            raise RuntimeError(f"Incomplete movie duration: {key}")
        verified.append({"run": key, "checkpoint": str(checkpoint), "movie": run["video"], "metadata": metadata})
    for directory in (ROOT / "videos/dvi_release3").glob("2026-10-04_*_64env"):
        if any(path.suffix != ".mp4" or not path.is_file() for path in directory.iterdir()):
            raise RuntimeError(f"Video directory contains other artifacts: {directory}")
        count = 7 if "initial" in directory.name else 21
        if len(list(directory.iterdir())) != count:
            raise RuntimeError(f"Incomplete video directory: {directory}")
    for task in RECIPES:
        metric = OUTPUT / "initial" / f"initial_{task}_64.json"
        if task == "ant" and not metric.exists():
            metric = OUTPUT / "initial/ant.json"
        initial = json.loads(metric.read_text())
        if initial["checkpoint"] != state[f"jacobi_{task}"]["checkpoint"]:
            raise RuntimeError(f"Initial video uses an older policy: {task}")
        if not initial["video"]["camera_stationary"]:
            raise RuntimeError(f"Initial video moves the camera: {task}")
        reader = imageio_ffmpeg.read_frames(initial["video"]["path"])
        metadata = next(reader)
        reader.close()
        if metadata["size"] != (3840, 2160):
            raise RuntimeError(f"Wrong initial movie resolution: {task}")
        if abs(metadata["duration"] - initial["duration_seconds"]) > 0.1:
            raise RuntimeError(f"Incomplete initial movie duration: {task}")
        verified.append(
            {
                "run": f"initial_{task}",
                "checkpoint": initial["checkpoint"],
                "movie": initial["video"]["path"],
                "metadata": metadata,
                "same_latest_jacobi": True,
            }
        )
    (EXPORT / "deliverable_verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    capture_final_source()


def main():
    """Read actual TensorBoard scalars; preserve archived paper datasets."""
    state = json.loads((OUTPUT / "campaign.json").read_text())
    curves, summaries = {}, []
    for key, run in state.items():
        if key.startswith("initial_"):
            continue
        checkpoint = run.get("checkpoint")
        if checkpoint:
            directory = Path(checkpoint).parent
        else:
            candidates = sorted((OUTPUT.parents[1] / "rsl_rl/release3_dvi_matrix_20261004").glob(f"*_{key}"))
            if not candidates:
                continue
            directory = candidates[-1]
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        curves[key] = {}
        summary = {"run": key, "directory": str(directory), "status": run.get("status", "pending")}
        for tag in tags:
            events = accumulator.Scalars(tag)
            values = np.asarray([event.value for event in events])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite training scalar: {key}, {tag}")
            curves[key][tag] = {"iteration": [event.step for event in events], "value": values.tolist()}
            summary[f"{tag}/mean_last_50"] = float(values[-50:].mean())
        if "Train/mean_reward" in curves[key]:
            summary["final_logged_iteration"] = curves[key]["Train/mean_reward"]["iteration"][-1]
        summaries.append(summary)
        if run.get("status") == "complete":
            raw = EXPORT / "raw" / key
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "source_run_directory.txt").write_text(str(directory) + "\n")
            for filename in ("env.yaml", "agent.yaml"):
                source = directory / "params" / filename
                if source.exists():
                    (raw / "params").mkdir(exist_ok=True)
                    shutil.copy2(source, raw / "params" / filename)
            if checkpoint:
                digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                shutil.copy2(checkpoint, raw / Path(checkpoint).name)
                (raw / (Path(checkpoint).name + ".sha256")).write_text(digest + "\n")
            logs = sorted((OUTPUT / "training" / key).glob("*/*.log"))
            if key == "jacobi_go2":
                logs = sorted((OUTPUT / "go2_fix").glob("*/*.log"))
            if logs:
                shutil.copy2(logs[-1], raw / "run.log")
                manifest = logs[-1].with_suffix(".json")
                if manifest.exists():
                    shutil.copy2(manifest, raw / "launch_manifest.json")
            (raw / "commands.json").write_text(json.dumps(run.get("commands", []), indent=2) + "\n")
            refresh = OUTPUT / "stationary_video_refresh" / f"{key}.json"
            if refresh.exists():
                shutil.copy2(refresh, raw / "stationary_video_command.json")
            for count in (64, 4096):
                if f"metrics{count}" in run:
                    metrics = run[f"metrics{count}"]
                    source = OUTPUT / "evaluations" / f"{key}_{count}.json"
                    if source.exists():
                        recorded = json.loads(source.read_text())
                        if recorded.get("checkpoint") == checkpoint:
                            metrics = recorded
                    (raw / f"metrics{count}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    snapshot = EXPORT / "source_snapshot"
    shutil.copytree(OUTPUT / "source_snapshot", snapshot / "initial", dirs_exist_ok=True)
    tooling = snapshot / "final_tooling"
    tooling.mkdir(parents=True, exist_ok=True)
    for filename in (
        "analyze_dvi_release_matrix.py",
        "run_dvi_release_matrix.py",
        "validate_dvi_release.py",
        "evaluate_dvi_release.py",
        "dvi_release_metrics.py",
    ):
        shutil.copy2(ROOT / "scripts/reinforcement_learning" / filename, tooling / filename)
    (snapshot / "README.md").write_text(
        "The initial/ directory preserves the source snapshot taken at campaign launch. "
        "The final_tooling/ directory contains the current launcher, evaluator, and analysis helpers. "
        "After all deliverables pass verification, final_source/ records both working-tree patches, "
        "repository revisions, and the untracked Newton regression test and changelog fragment. "
        "Recorded evaluator checksums distinguish the video renderer revisions. "
        "Earlier flat copies are retained as intermediate exports.\n"
    )
    shutil.copytree(OUTPUT / "go2_fix/playback_ablations", EXPORT / "go2_playback_ablations", dirs_exist_ok=True)
    shutil.copytree(OUTPUT / "paper_branch_comparisons", EXPORT / "paper_branch_comparisons", dirs_exist_ok=True)
    shutil.copytree(OUTPUT / "stationary_video_refresh", EXPORT / "stationary_video_refresh", dirs_exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    (EXPORT / "training_curves.json").write_text(json.dumps(curves) + "\n")
    (EXPORT / "training_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fields = sorted({field for row in summaries for field in row})
    with (EXPORT / "training_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    for tag, filename, ylabel in [
        ("Train/mean_reward", "training_reward", "Mean episode reward"),
        ("Train/mean_episode_length", "training_episode_length", "Mean episode length (steps)"),
    ]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for axis, task in zip(axes.flat, RECIPES):
            for solver, label in LABELS.items():
                points = curves.get(f"{solver}_{task}", {}).get(tag)
                if points:
                    values = np.asarray(points["value"])
                    window = min(25, len(values))
                    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
                    axis.plot(points["iteration"][window - 1 :], smoothed, color=COLORS[solver], label=label)
            axis.set_title(task.replace("_", " "))
            axis.set_xlabel("PPO iteration")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(RECIPES) :]:
            axis.axis("off")
        fig.suptitle("Isaac Lab release 3 / Newton 1.6 DVI: seed 42, 4096 environments\n25-iteration moving mean")
        fig.savefig(EXPORT / f"{filename}.png", dpi=220)
        fig.savefig(EXPORT / f"{filename}.pdf")
        plt.close(fig)
    verify_deliverables(state, curves)
    print(f"Exported {len(curves)} measured training series to {EXPORT}")


if __name__ == "__main__":
    main()
