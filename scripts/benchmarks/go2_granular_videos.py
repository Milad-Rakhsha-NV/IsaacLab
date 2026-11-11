# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared explanations for the four final-policy videos in the Go2 study."""

from __future__ import annotations

import html
from pathlib import Path

PRIMARY_VIDEOS = (
    (
        "rigid_max_forward",
        "Rigid-trained policy on a rigid floor",
        "Rigid ground",
        "Rigid floor",
        "The rigid-ground policy is evaluated at the maximum forward command from training. "
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


def render_videos(root: Path, cases: list[dict], gait: dict, *, url_prefix: str = "../") -> str:
    """Describe only the final comparison clips; retain provenance in collapsed details."""
    by_name = {case["case"]: case for case in cases}
    cards = []
    for name, title, training_terrain, evaluation_terrain, description, interpretation in PRIMARY_VIDEOS:
        if name not in by_name or not (root / name / "rollout.mp4").exists():
            cards.append(f"<article><h3>{html.escape(title)}</h3><p>Recording pending.</p></article>")
            continue
        version = int((root / name / "rollout.mp4").stat().st_mtime)
        case = by_name[name]
        duration = case["simulated_seconds"]
        failure = case["first_failure_seconds"]
        outcome = (
            f"No failure threshold reached during {duration:g} s"
            if failure is None
            else f"Failure threshold reached at {failure:.2f} s"
        )
        diagnostic = gait.get(name)
        gait_text = ""
        if diagnostic and diagnostic["reversed_rear_foot_order_fraction"] > 0:
            gait_text = (
                f" Rear feet cross at {diagnostic['first_rear_foot_order_reversal_s']:.2f} s and their "
                f"left/right order stays reversed in {diagnostic['reversed_rear_foot_order_fraction']:.1%} of frames."
            )
        elif diagnostic:
            gait_text = f" Mean rear-foot spacing: {100 * diagnostic['mean_rear_foot_separation_m']:.1f} cm."
        checkpoint = html.escape(str(case["checkpoint"]))
        command = case.get("commands", [])
        commands = "Command history is included in the evaluation data."
        if command and len(command) == 1 and not case.get("command_schedule"):
            forward, lateral, yaw = command[0]
            limit = case.get("training_forward_command_range", (0.1, 0.5))[1]
            maximum = " (training maximum)" if abs(forward - limit) < 1e-6 else ""
            commands = f"Forward command: {forward:.2f} m/s{maximum}; lateral: {lateral:.2f} m/s; yaw: {yaw:.2f} rad/s."
        depth = case.get("substrate_depth") if case["field"] == "rft" else case.get("bed_depth")
        layer = f"Initial sand depth: {1000 * depth:g} mm. " if case["field"] != "rigid" and depth is not None else ""
        boundary = ""
        if case["field"] == "mpm":
            exit_time = case.get("first_bed_exit_seconds")
            boundary = (
                f" The body entered the bed-edge margin at {exit_time:.2f} s."
                if exit_time is not None
                else " The body stayed within the bed's interior margin."
            )
        cards.append(
            f"<article><h3>{html.escape(title)}</h3>"
            f"<p><strong>Policy trained on:</strong> {html.escape(training_terrain)}<br>"
            f"<strong>Terrain simulated in this video:</strong> {html.escape(evaluation_terrain)}</p>"
            f"<p>{html.escape(description)}</p>"
            f"<p><strong>{commands} {layer}</strong></p>"
            f"<video controls preload='metadata' src='{url_prefix}{name}/rollout.mp4?v={version}' "
            f"poster='{url_prefix}{name}/preview.jpg?v={version}' aria-label='{html.escape(title)}'></video>"
            f"<p><strong>{outcome}.</strong> Mean planar velocity error: "
            f"{case['mean_velocity_error']:.3f} m/s.{gait_text}{boundary}</p>"
            f"<p>{html.escape(interpretation)}</p>"
            "<details><summary>Recording details</summary>"
            f"<p>Duration: {duration:g} s. "
            "Failure means body height below 0.16 m above the initial surface, excessive tilt "
            "(projected gravity z greater than −0.3), or nonfinite position. The rollout continues without resets. "
            "Errors cover the full recording, including any interval after a failure. "
            "Green arrows show granular body forces; the rigid-floor clip has no granular load.</p>"
            f"<p>Saved policy: <code>{checkpoint}</code></p>"
            f"<p><a href='{url_prefix}{name}/summary.json'>Evaluation data</a></p></details></article>"
        )
    main = (
        "<div class='video-grid'>" + "".join(cards) + "</div>"
        if cards
        else "<p>The four maximum-forward-command recordings are being generated.</p>"
    )
    return main
