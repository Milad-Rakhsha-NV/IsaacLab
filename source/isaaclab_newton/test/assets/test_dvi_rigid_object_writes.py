# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check selective rigid-object resets against native maximal-coordinate state."""

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView

from isaaclab.physics import PhysicsManager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("selection", ["index32", "index64", "mask"])
def test_rigid_object_reset_updates_native_state_and_preserves_peers(monkeypatch, selection):
    """Reset one cube without reconstructing other objects' maximal body state."""
    source = newton.ModelBuilder()
    body = source.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    source.add_articulation([source.add_joint_free(body)], label="Cube")
    builder = newton.ModelBuilder()
    for _ in range(3):
        builder.add_world(source)
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    for name, value in {
        "_model": model,
        "_state_0": state,
        "_control": model.control(),
        "_world_reset_mask": wp.zeros(3, dtype=wp.int32, device="cpu"),
        "_fk_reset_mask": wp.zeros(3, dtype=wp.bool, device="cpu"),
        "_model_changes": set(),
        "_solver": None,
    }.items():
        monkeypatch.setattr(NewtonManager, name, value)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    asset = RigidObject.__new__(RigidObject)
    asset._device = "cpu"
    asset._check_shapes = True
    asset._root_view = ArticulationView(model, "Cube", exclude_joint_types=[newton.JointType.FREE])
    asset._data = RigidObjectData(asset._root_view, "cpu")
    asset._ALL_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    asset._ALL_ENV_MASK = wp.ones(3, dtype=wp.bool, device="cpu")
    poses = state.body_q.numpy()
    poses[[0, 2], 2] = [0.1, 0.3]
    state.body_q.assign(poses)
    velocities = state.body_qd.numpy()
    velocities[[0, 2], 0] = [0.2, 0.4]
    state.body_qd.assign(velocities)
    target_pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
    target_velocity = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    if selection == "mask":
        mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
        asset.write_root_pose_to_sim_mask(root_pose=target_pose.repeat(3, 1), env_mask=mask)
        asset.write_root_velocity_to_sim_mask(root_velocity=target_velocity.repeat(3, 1), env_mask=mask)
    else:
        indices = torch.tensor([1], dtype=torch.int32 if selection == "index32" else torch.int64)
        asset.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=indices)
        asset.write_root_velocity_to_sim_index(root_velocity=target_velocity, env_ids=indices)
    NewtonManager.forward()
    np.testing.assert_allclose(state.body_q.numpy()[1], target_pose.numpy()[0])
    np.testing.assert_allclose(state.body_qd.numpy()[1], target_velocity.numpy()[0])
    np.testing.assert_array_equal(state.body_q.numpy()[[0, 2]], poses[[0, 2]])
    np.testing.assert_array_equal(state.body_qd.numpy()[[0, 2]], velocities[[0, 2]])
