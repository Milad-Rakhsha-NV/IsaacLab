# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify all eight Go2 recordings and finish their paired-command report."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from go2_granular_videos import PRIMARY_VIDEOS, video_cases
from imageio_ffmpeg import count_frames_and_secs
from monitor_go2_granular import ROOT, load_study
from report_go2_materials import policy_report
from run_go2_speed_comparison import write_json
from validate_go2_training import job_state


def main() -> None:
    """Reuse unchanged 1 m/s physics traces and validate every final movie and plot."""
    while job_state("go2-finish-half-speed.service").get("ActiveState") in ("active", "activating"):
        time.sleep(30)
    state = job_state("go2-finish-half-speed.service")
    if state.get("Result") != "success" or state.get("ExecMainStatus") != "0":
        raise RuntimeError(f"Half-speed recordings did not complete: {state}")
    study = load_study()
    root = ROOT / study["case_directory"]
    status_path = ROOT / study["status_file"]
    status = json.loads(status_path.read_text())
    status["stage"] = "All eight Go2 videos are available; finishing half-speed material plots"
    write_json(status_path, status)
    subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
    material_root = root / "materials_cmd_0p5"
    checkpoints = {key: Path(folder) / "model_1499.pt" for key, folder in study["run_directories"].items()}
    for phase in ("materials", "reference"):
        command = [
            sys.executable,
            "scripts/benchmarks/evaluate_go2_material_bank.py",
            "--phase",
            phase,
            "--output_root",
            str(material_root),
            "--speed",
            "0.5",
            "--max_forward_command",
            "1.0",
            "--bed_length",
            "5",
            "--bed_width",
            "1.6",
            "--checkpoint_rigid",
            str(checkpoints["rigid"]),
            "--checkpoint_rft",
            str(checkpoints["rft"]),
        ]
        with (root / f"{phase}_cmd_0p5.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = policy_report(material_root, material_root, speed=0.5)
    write_json(material_root / "results.json", result)
    labels = {name: title for name, title, *_ in PRIMARY_VIDEOS}
    verified = []
    verified_rows = []
    table = [
        "| Training / evaluation terrain | Forward command | Tracking error | First failure threshold |",
        "|---|---:|---:|---|",
    ]
    for name, base, speed in video_cases((1.0, 0.5)):
        folder = root / name
        summary = json.loads((folder / "summary.json").read_text())
        frames, seconds = count_frames_and_secs(folder / "rollout.mp4")
        assert frames == 500 and abs(seconds - 10) < 0.1
        assert summary["steps"] == 500 and summary["execution_error"] is None
        assert summary["seed"] == 123
        assert summary["particles_finite"] and summary["max_rigid_overflow"] == 0
        if summary["field"] == "mpm":
            assert summary["first_bed_exit_seconds"] is None
            assert summary["proxy_iterations"] == 1 and summary["proxy_relaxation"] == 0.1
            for key, expected_value in (
                ("density", 1600.0),
                ("internal_friction", 0.4),
                ("roughness", 1.0),
                ("bed_depth", 0.118),
                ("voxel_size", 0.02),
                ("substeps", 4),
                ("mpm_iterations", 50),
            ):
                assert summary[key] == expected_value, (name, key, summary[key])
        expected = checkpoints[summary["policy_training_terrain"]].resolve()
        assert Path(summary["checkpoint"]).resolve() == expected
        with np.load(folder / "rollout.npz") as data:
            np.testing.assert_array_equal(data["commands"], np.broadcast_to([speed, 0, 0], data["commands"].shape))
        failure = summary["first_failure_seconds"]
        outcome = "None in 10 s" if failure is None else f"{failure:.2f} s"
        table.append(f"| {labels[base]} | {speed:.2f} m/s | {summary['mean_velocity_error']:.3f} m/s | {outcome} |")
        verified.append(name)
        verified_rows.append(dict(case=name, **summary))
    study = load_study()
    study.update(
        published_videos=verified,
        videos_ready=True,
        half_speed_material_comparison_ready=True,
        mpm_recheck_ready=True,
        reused_maximum_command_trajectories=True,
        go2_report_complete=True,
    )
    write_json(ROOT / "current-study.json", study)
    status = json.loads(status_path.read_text())
    status["cases"] = [row for row in status["cases"] if row["case"] not in verified] + verified_rows
    status["stage"] = "Go2 complete: eight videos, both command plots, material comparisons and coupling checks"
    write_json(status_path, status)
    lines = [
        "# Go2 paired-command comparison",
        "",
        status["stage"],
        "",
        "[Report](http://localhost:18010/report/)",
        "",
        "Both policies trained with 4,096 environments for 1,500 iterations, seed 42, "
        "and forward commands 0.1–1.0 m/s.",
        "The same frozen checkpoints are evaluated at 1.0 and 0.5 m/s. "
        "Every video is 10 s and has 500 frames at 50 FPS.",
        "All eight recordings use evaluation seed 123 and fixed material parameters; "
        "these are single-seed comparisons.",
        "The 1.0 m/s videos were regenerated from saved trajectories with new banners; 0.5 m/s uses new rollouts.",
        "",
        *table,
        "",
        "MPM uses one coupling pass and feedback relaxation 0.1 in all main videos. The 1.0 m/s bed is 12 × 1.6 m;",
        "the 0.5 m/s bed is 12 × 3.2 m to avoid a boundary exit. Both policies share a bed at each command.",
        "The width difference limits a strict comparison across speeds. "
        "Sand depth, density and friction are unchanged.",
        "",
        "An excluded narrow-bed 0.5 m/s trial reached the bed margin at 7.2 s before its 9.38 s clearance failure.",
        "That fall is not counted as a material-induced failure.",
        "",
        "The immediate 1.0 m/s collapse is sensitive to coupling numerics. Finer time steps did not remove it,",
        "but four coupling passes or full feedback avoided the failure threshold in 2 s tests. This is not a",
        "10 s convergence result. The video falls cannot be attributed solely to the RFT policy or material law.",
        "",
        "MPM force convergence and material equivalence remain unvalidated. "
        "RFT standing and slip acceptance checks remain failed.",
        "G1 training is stopped while this Go2 report is reviewed.",
        "",
    ]
    (root / "STATUS.md").write_text("\n".join(lines))
    document = Path("GO2_RFT_STATUS.md")
    previous = document.read_text()
    marker = "## Previous study: forward commands up to 0.5 m/s"
    history = previous[previous.index(marker) :] if marker in previous else ""
    if history:
        history_lines = history.splitlines()
        history = (
            "\n".join(
                [history_lines[0], *["#" + line if line.startswith("## ") else line for line in history_lines[1:]]]
            )
            + "\n"
        )
    current = "\n".join(lines).replace(
        "# Go2 paired-command comparison", "# Go2: rigid, RFT and MPM policy comparisons"
    )
    current += (
        "\n## Training and material settings\n\n"
        "RFT reached material difficulty 1.0. Reset-based randomization covers density 1,400–1,800 kg/m³, "
        "internal friction 0.3–0.5, surface/internal friction ratio 0.6–1.0 and log-uniform strength 0.8–1.2. "
        "Layers span 0–118 mm from the first episode over a fixed substrate, half Go2's approximately 236 mm "
        "knee-to-sole length. Material distributions widen with performance; commands have no curriculum. "
        "Lateral commands span ±0.1 m/s, yaw rates ±0.3 rad/s, and 10% of resets request standing.\n\n"
        "Both checkpoints are `model_1499.pt` in `logs/rsl_rl/go2_rigid_max1/2026-11-11_08-33-41/` "
        "and `logs/rsl_rl/go2_rft_max1/2026-11-11_08-33-41/`. MPM is excluded from training. "
        "The report includes training curves, two command-tracking figures, three-material plots at both commands, "
        "and the controlled MPM startup comparison. Material trials last 3 s and use 5 × 1.6 m beds; "
        "support includes contacts after failure. The older force-convergence investigation remains applicable "
        "and has not passed its acceptance gate.\n\n"
        "The rigid policy passed the held-out checks. RFT standing drift and loaded-foot slip still fail. "
        "A completed training budget does not establish policy convergence or material equivalence.\n\n"
        "## Access and artifacts\n\n"
        "[Live report](http://localhost:18010/report/) · [Generated status](http://localhost:18010/max1/STATUS.md)\n\n"
        "Keep the existing SSH tunnel running: `ssh -o ExitOnForwardFailure=yes -N "
        "-L 18010:127.0.0.1:8010 horde@mrakhsha-41eslp`. Generated videos, state traces, summaries and figures "
        "remain under ignored `logs/go2_granular/` paths. The 1.0 m/s original movies are preserved beside "
        "their retitled copies; their numerical trajectories are unchanged.\n\n"
    )
    checks = json.loads((root / "mpm-recheck/results.json").read_text())["cases"]
    current += (
        "## Controlled MPM startup check at 1.0 m/s\n\n"
        "These 2 s trials use the same checkpoint, seed 123 and prepared 5 × 1.6 × 0.118 m bed. "
        "The physics frame is 5 ms. Only the listed numerical control changes.\n\n"
        "| Substeps | Coupling passes | Feedback factor | Tracking error | First failure threshold |\n"
        "|---:|---:|---:|---:|---|\n"
    )
    for row in checks:
        s = row["summary"]
        first = s["first_failure_seconds"]
        outcome = "None in 2 s" if first is None else f"{first:.2f} s"
        current += (
            f"| {row['substeps']} | {row['coupling_passes']} | {s.get('proxy_relaxation', 0.1):g} | "
            f"{s['mean_velocity_error']:.3f} m/s | {outcome} |\n"
        )
    current += (
        "\nThese results demonstrate coupling sensitivity. They do not establish a converged MPM reference, "
        "accurate 1.0 m/s tracking, or 10 s stability for the alternative settings.\n\n"
    )
    document.write_text(current + history)
    write_json(root / "paired-video-verification.json", dict(cases=verified, completed_unix=time.time()))
    subprocess.run([sys.executable, "scripts/benchmarks/report_go2_granular.py"], check=True)
    subprocess.run([sys.executable, "scripts/benchmarks/monitor_go2_granular.py", "--once"], check=True)
    write_json(
        root / "followup-complete.json",
        dict(completed_unix=time.time(), videos=verified, numerical_reference_accepted=False),
    )


if __name__ == "__main__":
    main()
