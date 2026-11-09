# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation must balance bodies and preserve the task's raw-action semantics."""

import pytest
import torch

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_imitation import balanced_action_mse, descriptor_roundoff


def test_imitation_balances_bodies_and_ignores_padding_without_clipping_targets():
    """A larger body or duplicated trajectory cannot outweigh another body; saturation still matters."""
    prediction = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[2.0, 1000.0, -1000.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask = torch.tensor([[True, False, False], [True, True, True], [True, True, True]])
    bodies = torch.tensor([0, 1, 1])
    loss = balanced_action_mse(prediction, target, mask, bodies)
    # Body 0 contributes 4, body 1 contributes 1, independent of row and joint counts.
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(prediction.grad[0], torch.tensor([-2.0, 0.0, 0.0]))
    subset = torch.tensor([0, 1])
    torch.testing.assert_close(
        balanced_action_mse(prediction[subset], target[subset], mask[subset], bodies[subset]), loss
    )
    assert balanced_action_mse(prediction, target.clamp(-1, 1), mask, bodies) < loss


def test_teacher_mechanics_tolerate_transform_roundoff_but_reject_physics_or_topology_changes():
    """Numerical zero residuals cannot hide a changed mass or mechanical relation."""
    reference = {"ant": {"static": torch.tensor([0.0, 2.0]), "edges": torch.tensor([0, 1], dtype=torch.uint8)}}
    actual = {"ant": {key: value.clone() for key, value in reference["ant"].items()}}
    actual["ant"]["static"][0] = -1.4e-17
    assert descriptor_roundoff(actual, reference)["ant/static"] < 1e-12
    actual["ant"]["static"][1] = 2.01
    with pytest.raises(ValueError, match="ant/static"):
        descriptor_roundoff(actual, reference)
    actual["ant"]["static"][1] = 2.0
    actual["ant"]["edges"][1] = 0
    with pytest.raises(ValueError, match="ant/edges"):
        descriptor_roundoff(actual, reference)
