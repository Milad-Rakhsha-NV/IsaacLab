# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Differentiable XYZW rotations and the 71-component G1 state contract."""

import torch

JOINT_NAMES = (
    [
        f"{side}_{j}_joint"
        for side in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{j}_joint"
        for side in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
)
# State: world root-link position, XYZW orientation, world root-link linear/angular
# velocity, named joint positions, named joint velocities. Environment origins removed.
POS, QUAT, VEL, OMEGA, Q, QD = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13), slice(13, 42), slice(42, 71)


def conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def multiply(a, b):
    av, bv = a[..., :3], b[..., :3]
    return torch.cat(
        (
            a[..., 3:] * bv + b[..., 3:] * av + torch.cross(av, bv, dim=-1),
            a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True),
        ),
        -1,
    )


def rotate(q, v):
    qv, v = torch.broadcast_tensors(q[..., :3], v)
    cross = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., 3:] * cross + torch.cross(qv, cross, dim=-1)


def exp_rotation(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * v, torch.cos(angle * 0.5)), -1)


def log_rotation(q):
    q = torch.where(q[..., 3:] < 0, -q, q)
    n = q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, q[..., 3:].clamp_min(1e-8))
    scale = torch.where(n > 1e-6, angle / n.clamp_min(1e-8), 2 * torch.ones_like(n))
    return q[..., :3] * scale


def heading(q):
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)), -1)


def features(s):
    inv = conjugate(heading(s[..., QUAT]))
    return torch.cat(
        (
            s[..., 2:3],
            multiply(inv, s[..., QUAT]),
            rotate(inv, s[..., VEL]),
            rotate(inv, s[..., OMEGA]),
            s[..., Q],
            s[..., QD],
        ),
        -1,
    )


def state_delta(s, successor):
    inv = conjugate(s[..., QUAT])
    return torch.cat(
        (
            rotate(inv, successor[..., POS] - s[..., POS]),
            log_rotation(multiply(inv, successor[..., QUAT])),
            rotate(inv, successor[..., VEL] - s[..., VEL]),
            rotate(inv, successor[..., OMEGA] - s[..., OMEGA]),
            successor[..., Q] - s[..., Q],
            successor[..., QD] - s[..., QD],
        ),
        -1,
    )


def integrate(s, delta):
    quat = multiply(s[..., QUAT], exp_rotation(delta[..., 3:6]))
    quat = torch.nn.functional.normalize(quat, dim=-1)
    return torch.cat(
        (
            s[..., POS] + rotate(s[..., QUAT], delta[..., :3]),
            quat,
            s[..., VEL] + rotate(s[..., QUAT], delta[..., 6:9]),
            s[..., OMEGA] + rotate(s[..., QUAT], delta[..., 9:12]),
            s[..., Q] + delta[..., 12:41],
            s[..., QD] + delta[..., 41:],
        ),
        -1,
    )
