# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared explanations for the four final-policy comparisons in granular studies."""

from __future__ import annotations

import html
from pathlib import Path

PRIMARY_VIDEOS = (
    (
        "rigid_max_forward",
        "Rigid-trained policy on a rigid floor",
        "Rigid ground",
        "Rigid floor",
        "The rigid-ground policy is evaluated at the requested forward commands. "
        "This is the rigid-surface reference for the sand recordings.",
        "The same rigid-trained weights are also tested in the rigid-policy-on-MPM clip. "
        "Comparing those two clips shows how this baseline transfers from a floor to sand.",
    ),
    (
        "rft_max_forward",
        "RFT-trained policy on RFT sand",
        "RFT sand with material randomization",
        "RFT sand",
        "Go2 was trained with the validated locomotion settings and reset-randomized RFT materials. "
        "Sand depth is sampled over 0–118 mm above a rigid floor, half the knee-to-sole length. "
        "Density, friction and strength ranges widen with performance.",
        "This tests the final RFT checkpoint in its training simulator at fixed nominal material properties. "
        "Standing, tracking, slip and gait acceptance checks are reported separately.",
    ),
    (
        "mpm_max_forward",
        "RFT-trained policy on MPM sand",
        "RFT sand with material randomization",
        "MPM sand",
        "The exact same new RFT-trained policy is evaluated in a particle-based MPM bed after training. "
        "No MPM particles or MPM training are used to learn the controller.",
        "This is an exploratory transfer test of the current checkpoint. Bed depth, density, friction and "
        "command match the RFT recording. Material calibration and numerical convergence "
        "are still needed to attribute differences to the force model.",
    ),
    (
        "rigid_mpm_max_forward",
        "Rigid-trained policy on MPM sand",
        "Rigid ground",
        "MPM sand",
        "The exact same rigid-ground policy as in the rigid-floor clip is tested in the particle-based MPM bed. "
        "It received no RFT training, MPM training or sand-specific fine-tuning.",
        "The bed, material, command and simulation settings match the RFT-policy-on-MPM clip. "
        "This provides a rigid-trained baseline alongside the RFT-trained policy under the same inference conditions.",
    ),
)


def video_cases(speeds: tuple[float, ...] = (1.0,)) -> list[tuple[str, str, float]]:
    """Name speed variants consistently across recording, dashboards and plots."""
    return [
        (name if index == 0 else f"{name}_cmd_{speed:g}".replace(".", "p"), name, speed)
        for name, *_ in PRIMARY_VIDEOS
        for index, speed in enumerate(speeds)
    ]


def describe_progress(stage: str, speeds: tuple[float, ...]) -> str:
    """Use terrain roles and commands instead of internal recording identifiers."""
    titles = {row[0]: row[1] for row in PRIMARY_VIDEOS}
    for name, base, speed in sorted(video_cases(speeds), key=lambda row: len(row[0]), reverse=True):
        stage = stage.replace(name.replace("_", " "), f"{titles[base]} ({speed:.2f} m/s)")
    return stage


def render_videos(
    root: Path,
    cases: list[dict],
    gait: dict,
    *,
    url_prefix: str = "../",
    speeds: tuple[float, ...] = (1.0,),
    robot: str = "Go2",
) -> str:
    """Group matched commands under each training/evaluation terrain combination."""
    by_name = {case["case"]: case for case in cases}
    cards = []
    variants = video_cases(speeds)
    for base, title, training, terrain, description, interpretation in PRIMARY_VIDEOS:
        if robot != "Go2":
            description = description.replace("Go2", robot).replace(", half the knee-to-sole length", "")
        if robot == "G1" and base == "rft_max_forward":
            description = (
                "G1 starts from the accepted rigid-ground policy and fine-tunes in reset-randomized RFT materials. "
                "The material curriculum widens layer depth from 0–10 mm to 0–118 mm above a rigid floor "
                "and expands the density, friction and strength ranges."
            )
        clips = []
        for name, role, speed in variants:
            if role != base:
                continue
            clip = root / name / "rollout.mp4"
            if name not in by_name or not clip.exists():
                clips.append(f"<div><h4>Command: {speed:.2f} m/s forward</h4><p>Recording pending.</p></div>")
                continue
            case = by_name[name]
            command = case["commands"][0]
            version = int(clip.stat().st_mtime)
            duration = case["simulated_seconds"]
            failure = case["first_failure_seconds"]
            outcome = (
                f"No failure threshold reached in {duration:g} s"
                if failure is None
                else f"First failure threshold: {failure:.2f} s"
            )
            depth = case.get("substrate_depth") if case["field"] == "rft" else case.get("bed_depth")
            material_text = (
                f"Initial sand depth: {1000 * depth:g} mm. "
                if case["field"] != "rigid" and depth is not None
                else "This clip has no sand. "
            )
            domain = f"MPM bed: {case['bed_length']:g} × {case['bed_width']:g} m. " if case["field"] == "mpm" else ""
            clips.append(
                f"<div><h4>Command: {command[0]:.2f} m/s forward</h4>"
                f"<video controls preload='metadata' src='{url_prefix}{name}/rollout.mp4?v={version}' "
                f"poster='{url_prefix}{name}/preview.jpg?v={version}' "
                f"aria-label='{html.escape(title)} at {speed:g} m/s'></video>"
                f"<p>Mean planar tracking error: <strong>{case['mean_velocity_error']:.3f} m/s</strong>. {outcome}.</p>"
                f"<p>{domain}</p>"
                "<details><summary>Recording details and numerical data</summary>"
                f"<p>{duration:g} s; lateral command {command[1]:.2f} m/s; yaw {command[2]:.2f} rad/s. "
                f"{material_text}Failure means low body clearance, excessive tilt or nonfinite position. "
                "The rollout continues without resets; metrics include time after failure. "
                "Green arrows show granular body forces.</p>"
                f"<p>Checkpoint: <code>{html.escape(str(case['checkpoint']))}</code></p>"
                f"<p><a href='{url_prefix}{name}/summary.json'>Evaluation data</a></p></details></div>"
            )
        cards.append(
            f"<article><h3>{html.escape(title)}</h3>"
            f"<p><strong>Training:</strong> {html.escape(training)}. "
            f"<strong>Evaluation:</strong> {html.escape(terrain)}.</p>"
            f"<p>{html.escape(description)}</p>"
            "<div class='paired-videos'>" + "".join(clips) + "</div>"
            f"<p>{html.escape(interpretation)}</p></article>"
        )
    return "<div class='video-grid'>" + "".join(cards) + "</div>"
