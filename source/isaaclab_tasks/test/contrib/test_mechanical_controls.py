# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The source controls isolate specialist means and explicit inertial information."""

import torch
from tensordict import TensorDict

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_controls import (
    MechanicalSpecialists,
    MechanicalWithoutInertia,
)
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import (
    DYNAMIC_WIDTH,
    STATIC_SLICES,
    STATIC_WIDTH,
)


def test_specialist_update_and_normalization_do_not_change_another_body_mean():
    """Learning from one body's rows cannot update another body's mean or input statistics."""
    torch.manual_seed(12)
    obs = TensorDict(
        {
            "critic": torch.randn(6, 7),
            "embodiment": torch.tensor([0, 0, 0, 1, 1, 1]),
            "action_mask": torch.tensor([[True, True, False]] * 3 + [[True, True, True]] * 3),
        },
        batch_size=[6],
    )
    actor = MechanicalSpecialists(obs, {}, "actor", 3, [5, 7], [2, 3]).eval()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    # A previously active head has Adam momentum; empty rows must leave its gradients absent.
    actor(obs[:3]).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = actor(obs).detach().clone()
    actor.update_normalization(obs[3:])
    actor(obs[3:]).square().sum().backward()
    optimizer.step()
    after = actor(obs).detach()
    torch.testing.assert_close(after[:3], before[:3], rtol=0, atol=0)
    assert not torch.equal(after[3:], before[3:])
    assert torch.equal(after[:3, 2], torch.zeros(3))
    actor(obs, stochastic_output=True)
    log_prob = actor.get_output_log_prob(after)
    arbitrary_padding = after.clone()
    arbitrary_padding[:3, 2] = 10000
    torch.testing.assert_close(actor.get_output_log_prob(arbitrary_padding), log_prob)


def test_inertial_ablation_blocks_explicit_information_without_mutating_shared_observations():
    """Body/rotor inertia and the total-mass side channel cannot reach this actor; geometry still can."""
    torch.manual_seed(23)
    obs = TensorDict(
        {
            "global": torch.randn(2, 24),
            "static": torch.randn(2, 4, STATIC_WIDTH),
            "dynamic": torch.randn(2, 4, DYNAMIC_WIDTH),
            "edges": torch.zeros(2, 4, 4, dtype=torch.uint8),
            "node_mask": torch.ones(2, 4, dtype=torch.bool),
            "action_nodes": torch.tensor([[1, 3], [1, 3]]),
            "action_mask": torch.ones(2, 2, dtype=torch.bool),
            "critic": torch.randn(2, 7),
        },
        batch_size=[2],
    )
    original = obs.clone()
    actor = MechanicalWithoutInertia(obs, {}, "actor", 2).eval()
    expected = actor(obs).detach()
    changed = obs.clone()
    for field in ("body_mass", "body_com", "body_inertia", "joint_armature"):
        changed["static"][..., STATIC_SLICES[field]] += 100
    changed["global"][:, 18] += 1000
    torch.testing.assert_close(actor(changed), expected, rtol=0, atol=0)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)
    # Gradients independently enforce exclusion and prevent an always-constant actor passing the test.
    obs["static"].requires_grad_()
    obs["global"].requires_grad_()
    actor(obs).sum().backward()
    for field in ("body_mass", "body_com", "body_inertia", "joint_armature"):
        assert torch.count_nonzero(obs["static"].grad[..., STATIC_SLICES[field]]) == 0
    assert torch.count_nonzero(obs["global"].grad[:, 18]) == 0
    assert obs["static"].grad[..., STATIC_SLICES["shape_scale"]].abs().sum() > 0
