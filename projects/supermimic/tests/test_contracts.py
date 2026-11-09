# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Protect data conventions, episode isolation, kinematics and analytic gradients."""

import numpy as np
import pytest
import torch

from projects.supermimic.learner import frozen
from projects.supermimic.math_utils import Q, exp_rotation, integrate, rotate, state_delta
from projects.supermimic.motions import MotionLibrary, prepare
from projects.supermimic.networks import WorldModel
from projects.supermimic.replay import Replay
from projects.supermimic.task import Kinematics

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_csv_conversion_and_clip_end(tmp_path):
    # Independent physical trajectory: 1 m/s translation, 1 rad/s yaw,
    # and a 2 rad/s first joint, sampled at the source's documented 30 Hz.
    t = np.arange(61) / 30
    raw = np.zeros((61, 36))
    raw[:, 0] = t
    raw[:, 2] = 0.8
    raw[:, 5] = np.sin(t / 2)
    raw[:, 6] = np.cos(t / 2)
    raw[:, 7] = 2 * t
    np.savetxt(tmp_path / "a.csv", raw, delimiter=",")
    raw[:, 0] += 100
    np.savetxt(tmp_path / "b.csv", raw, delimiter=",")
    output = tmp_path / "motions.npz"
    prepare(tmp_path, output, "test-physical-trajectory")
    library = MotionLibrary(output, "cpu")
    ref = library.reference(torch.tensor([0]), torch.tensor([50]))[0]
    torch.testing.assert_close(ref[:3], torch.tensor([1.0, 0.0, 0.8]))
    torch.testing.assert_close(ref[7:13], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]), atol=2e-5, rtol=1e-5)
    assert ref[42] == pytest.approx(2)
    future = library.future(torch.tensor([0]), torch.tensor([100]))
    assert (future[0, :, 0] == 2).all(), "Future references must stop before the next clip"


def test_replay_preserves_terminal_successor_and_isolates_history():
    replay = Replay(5, 1, "cpu")
    for step in range(6):
        ep = 0 if step < 3 else 1
        s = torch.full((1, 71), float(ep))
        successor = torch.full_like(s, 999 if step == 2 else ep)
        action = torch.full((1, 29), float(step))
        replay.add(s, successor, action, action, torch.tensor([ep]), torch.tensor([ep]), torch.tensor([step]))
    torch.manual_seed(2)
    states, actions, successors = replay.windows(100, 2)
    pairs = set(tuple(int(v) for v in pair) for pair in actions[:, :, 0].tolist())
    assert pairs == {(1, 2), (3, 4), (4, 5)}
    assert (successors[:, -1, 0] == 999).any(), "Falling transitions must be retained"
    sample = replay.starts(100, 4)
    assert (sample["history_states"][..., 0] == sample["state"][:, None, 0]).all()


def test_kinematics_world_velocities_and_rotation_zero_gradient():
    # One hinge one meter from a translating root: +Z rotation moves the child
    # endpoint around the hinge and contributes omega cross radius to velocity.
    xp = torch.tensor([[1.0, 0, 0, 0, 0, 0, 1]])
    xc = torch.tensor([[-1.0, 0, 0, 0, 0, 0, 1]])
    fk = Kinematics([0], [0], xp, xc, torch.tensor([[0.0, 0, 1]]), [0, 1])
    s = torch.zeros(1, 71)
    s[:, 6], s[:, 7], s[:, 13], s[:, 42] = 1, 3, torch.pi / 2, 2
    pos, _, vel, omega = fk(s)
    torch.testing.assert_close(pos[0, 1], torch.tensor([1.0, 1, 0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vel[0, 1], torch.tensor([1.0, 0, 0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(omega[0, 1], torch.tensor([0.0, 0, 2]))
    v = torch.zeros(1, 3, requires_grad=True)
    rotate(exp_rotation(v), torch.tensor([[1.0, 0, 0]])).sum().backward()
    torch.testing.assert_close(v.grad, torch.tensor([[0.0, -1.0, 1.0]]))


def test_frozen_world_model_retains_multistep_action_gradient():
    torch.manual_seed(5)
    model = WorldModel((16, 16)).double()
    state = torch.zeros(1, 71, dtype=torch.float64)
    state[:, 6] = 1

    def objective(a):
        s = state
        for _ in range(3):
            s = model(s, a)
        return s[:, Q].square().sum()

    action = torch.randn(1, 29, dtype=torch.float64, requires_grad=True)
    with frozen(model):
        loss = objective(action)
        loss.backward()
        eps = 1e-4
        offset = torch.zeros_like(action)
        offset[0, 0] = eps
        finite_difference = (objective(action.detach() + offset) - objective(action.detach() - offset)) / (2 * eps)
    assert action.grad.abs().max() > 0
    torch.testing.assert_close(action.grad[0, 0], finite_difference, rtol=1e-4, atol=1e-12)
    assert all(p.grad is None for p in model.parameters())
    delta = torch.randn(1, 70, dtype=torch.float64) * 0.01
    torch.testing.assert_close(state_delta(state, integrate(state, delta)), delta, atol=1e-10, rtol=1e-8)
