# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure world-space progress separately from body-frame velocity tracking."""

import numpy as np


def world_velocity_metrics(poses: np.ndarray, body_velocities: np.ndarray) -> dict[str, float]:
    """Rotate body velocities using Newton's xyzw root poses and measure the last half."""
    start = len(poses) // 2
    rotations = poses[start:, ..., 3:7]
    velocities = body_velocities[start:]
    cross = 2.0 * np.cross(rotations[..., :3], velocities)
    world = velocities + rotations[..., 3:4] * cross + np.cross(rotations[..., :3], cross)
    return {
        "mean_world_forward_speed_last_half": float(world[..., 0].mean()),
        "median_world_forward_speed_last_half": float(np.median(world[..., 0])),
        "mean_world_lateral_speed_last_half": float(np.abs(world[..., 1]).mean()),
    }
