# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Differentiable rigid-tree kinematics, proprioception and tracking objective."""

import torch
from torch import nn

from .math_utils import (
    JOINT_NAMES,
    OMEGA,
    POS,
    QD,
    QUAT,
    VEL,
    Q,
    conjugate,
    exp_rotation,
    log_rotation,
    multiply,
    rotate,
)

TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


class Kinematics(nn.Module):
    """Tree extracted from the actual Newton model; no simulator calls in forward()."""

    def __init__(self, parents, joints, xp, xc, axes, tracked):
        super().__init__()
        self.parents, self.joints, self.tracked = parents, joints, tracked
        self.register_buffer("xp", xp)
        self.register_buffer("xc", xc)
        self.register_buffer("axes", axes)
        depths = [0]
        for parent in parents:
            depths.append(depths[parent] + 1)
        self.levels = []
        for depth in range(1, max(depths) + 1):
            ids = [i for i in range(1, len(depths)) if depths[i] == depth]
            label = str(depth)
            self.register_buffer("ids_" + label, torch.tensor(ids, device=xp.device))
            self.register_buffer("parents_" + label, torch.tensor([parents[i - 1] for i in ids], device=xp.device))
            self.register_buffer("joints_" + label, torch.tensor([joints[i - 1] for i in ids], device=xp.device))
            self.levels.append(label)

    @classmethod
    def from_model(cls, model, device):
        # Use world zero, then prune finger subtrees: the G1 asset has fixed hands.
        names = [name.rsplit("/", 1)[-1] for name in model.body_label]
        root = names.index("pelvis")
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        kinds = model.joint_type.numpy()
        qd_start = model.joint_qd_start.numpy()
        axis = model.joint_axis.numpy()
        xp_all, xc_all = model.joint_X_p.numpy(), model.joint_X_c.numpy()
        labels = [name.rsplit("/", 1)[-1] for name in model.joint_label]
        selected = [root]
        parents, joints, xps, xcs, axes = [], [], [], [], []
        for i, c in enumerate(child):
            if c == root or parent[i] not in selected:
                continue
            label = labels[i]
            if label not in JOINT_NAMES:
                # None of the tracked chains includes an unactuated intermediate joint.
                continue
            if kinds[i] != 1:
                raise ValueError(f"Expected revolute G1 joint: {label}")
            parents.append(selected.index(parent[i]))
            joints.append(JOINT_NAMES.index(label))
            selected.append(int(c))
            xps.append(xp_all[i])
            xcs.append(xc_all[i])
            axes.append(axis[qd_start[i]])
        body_names = [names[i] for i in selected]
        if len(joints) != 29:
            raise ValueError(f"Expected 29 G1 joints, got {len(joints)}: {body_names}")
        import numpy as np

        return cls(
            parents,
            joints,
            torch.tensor(np.array(xps), device=device),
            torch.tensor(np.array(xcs), device=device),
            torch.tensor(np.array(axes), device=device),
            [body_names.index(name) for name in TRACKED_BODIES],
        )

    def forward(self, s):
        count = len(self.parents) + 1
        positions = s.new_zeros(*s.shape[:-1], count, 3)
        quats = s.new_zeros(*s.shape[:-1], count, 4)
        velocities, omegas = torch.zeros_like(positions), torch.zeros_like(positions)
        root = self.xp.new_zeros(1, dtype=torch.long)
        positions = positions.index_copy(-2, root, s[..., POS].unsqueeze(-2))
        quats = quats.index_copy(-2, root, s[..., QUAT].unsqueeze(-2))
        velocities = velocities.index_copy(-2, root, s[..., VEL].unsqueeze(-2))
        omegas = omegas.index_copy(-2, root, s[..., OMEGA].unsqueeze(-2))
        for level in self.levels:
            ids = self.get_buffer("ids_" + level)
            parent = self.get_buffer("parents_" + level)
            joint = self.get_buffer("joints_" + level)
            xp, xc, axis = self.xp[ids - 1], self.xc[ids - 1], self.axes[ids - 1]
            p, r = positions.index_select(-2, parent), quats.index_select(-2, parent)
            v, w = velocities.index_select(-2, parent), omegas.index_select(-2, parent)
            rp = multiply(r, xp[..., 3:].expand_as(r))
            offset = rotate(r, xp[..., :3])
            origin = p + offset
            rotation = multiply(rp, exp_rotation(s[..., Q].index_select(-1, joint)[..., None] * axis))
            rc = multiply(rotation, conjugate(xc[..., 3:]).expand_as(r))
            pc = origin - rotate(rc, xc[..., :3])
            wc = w + rotate(rp, axis) * s[..., QD].index_select(-1, joint)[..., None]
            vc = v + torch.cross(w, offset, dim=-1) - torch.cross(wc, origin - pc, dim=-1)
            positions = positions.index_copy(-2, ids, pc)
            quats = quats.index_copy(-2, ids, rc)
            velocities = velocities.index_copy(-2, ids, vc)
            omegas = omegas.index_copy(-2, ids, wc)
        return tuple(items[..., self.tracked, :] for items in (positions, quats, velocities, omegas))


class TrackingTask(nn.Module):
    def __init__(self, kinematics, default_q, limits):
        super().__init__()
        self.kinematics = kinematics
        self.register_buffer("default_q", default_q)
        self.register_buffer("limits", limits)
        self.register_buffer(
            "obs_noise", default_q.new_tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
        )

    def observation(self, state, previous, noise=False):
        gravity = torch.zeros_like(state[..., POS])
        gravity[..., 2] = -1
        inv = conjugate(state[..., QUAT])
        obs = torch.cat(
            (
                state[..., Q] - self.default_q,
                state[..., QD],
                rotate(inv, state[..., OMEGA]),
                rotate(inv, gravity),
                previous,
            ),
            -1,
        )
        if noise:
            obs = obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise
        return obs

    def termination(self, state, ref, soft=False):
        z = torch.zeros_like(state[..., POS])
        z[..., 2] = -1
        tilt = (rotate(conjugate(state[..., QUAT]), z)[..., 2] - rotate(conjugate(ref[..., QUAT]), z)[..., 2]).abs()
        height = (state[..., 2] - ref[..., 2]).abs()
        if soft:
            # Smooth survival permits policy gradients near the hard simulator thresholds.
            ph = torch.sigmoid((height - 0.25) * 100)
            pt = torch.sigmoid((tilt - 0.8) * 30)
            return 1 - (1 - ph) * (1 - pt)
        return (height > 0.25) | (tilt > 0.8)

    def reward(self, state, reference, action, previous, ref_kin=None):
        pos, quat, vel, omega = self.kinematics(state)
        rp, rq, rv, rw = self.kinematics(reference) if ref_kin is None else ref_kin
        orientation = log_rotation(multiply(quat, conjugate(rq))).square().sum(-1).mean(-1)
        terms = (
            torch.exp(-(pos - rp).square().sum(-1).mean(-1) / 0.3**2)
            + torch.exp(-orientation / 0.4**2)
            + torch.exp(-(vel - rv).square().sum(-1).mean(-1))
            + torch.exp(-(omega - rw).square().sum(-1).mean(-1) / 3.14**2)
            + 0.5 * torch.exp(-(state[..., POS] - reference[..., POS]).square().sum(-1) / 0.3**2)
            + 0.5
            * torch.exp(
                -log_rotation(multiply(state[..., QUAT], conjugate(reference[..., QUAT]))).square().sum(-1) / 0.4**2
            )
        )
        violation = (state[..., Q] - self.limits[:, 1]).clamp_min(0) + (self.limits[:, 0] - state[..., Q]).clamp_min(0)
        return 0.02 * (terms - 0.1 * (action - previous).square().sum(-1) - 2 * violation.sum(-1))
