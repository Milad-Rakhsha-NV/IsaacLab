# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared explanations for the three final-policy videos in the Go2 study."""

from __future__ import annotations

import html
from pathlib import Path

PRIMARY_VIDEOS = (
    (
        "corrected_rft_curriculum_nominal",
        "RFT-trained policy on RFT sand",
        "Go2 was trained with randomized RFT material properties. This video evaluates the final policy "
        "in a fixed RFT field with density 1,600 kg/m³ and friction coefficient 0.4, without a rigid floor.",
        "This is the reference behavior for the MPM comparison. The rear stance is narrow; "
        "avoiding a fall does not establish a good gait.",
    ),
    (
        "corrected_rft_curriculum_mpm",
        "Same RFT-trained policy on MPM sand",
        "The exact same final policy is evaluated with a particle-based MPM sand bed instead of RFT. "
        "Density, friction and the forward command match the RFT video. The bed is 0.4 m deep.",
        "This tests transfer without retraining. The robot fails in this rollout. The MPM setup "
        "still needs numerical convergence checks before it can serve as a validated reference.",
    ),
    (
        "corrected_rigid_final",
        "Rigid-ground policy on a rigid floor",
        "A separate policy was trained and evaluated on flat rigid ground using the same command range, "
        "observations and rewards. It was never trained in sand.",
        "This is the flat-ground control. Its rear legs cross even though it tracks velocity closely. "
        "The gait is unacceptable; self-collision and leg-posture constraints need correction.",
    ),
)


def render_videos(root: Path, cases: list[dict], gait: dict) -> str:
    """Describe only the final comparison clips; retain provenance in collapsed details."""
    by_name = {case["case"]: case for case in cases}
    cards = []
    for name, title, description, interpretation in PRIMARY_VIDEOS:
        if name not in by_name or not (root / name / "rollout.mp4").exists():
            continue
        case = by_name[name]
        duration = case["simulated_seconds"]
        failure = case["first_failure_seconds"]
        outcome = f"No fall detected during {duration:g} s" if failure is None else f"Fall detected at {failure:.2f} s"
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
        cards.append(
            f"<article><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p>"
            f"<video controls preload='metadata' src='../{name}/rollout.mp4' "
            f"poster='../{name}/preview.jpg' aria-label='{html.escape(title)}'></video>"
            f"<p><strong>{outcome}.</strong> Mean planar velocity error: "
            f"{case['mean_velocity_error']:.3f} m/s.{gait_text}</p>"
            f"<p>{html.escape(interpretation)}</p>"
            "<details><summary>Recording details</summary>"
            f"<p>Fixed forward command: 0.30 m/s; lateral/yaw commands: zero. Duration: {duration:g} s. "
            "Errors cover the full recording, including any interval after a fall. "
            "Green arrows show granular body forces; the rigid-floor clip has no granular load.</p>"
            f"<p>Saved policy: <code>{checkpoint}</code></p>"
            f"<p><a href='../{name}/summary.json'>Evaluation data</a></p></details></article>"
        )
    if not cards:
        return "<p>Final-policy recordings will appear here after training and evaluation finish.</p>"
    return "<div class='video-grid'>" + "".join(cards) + "</div>"
