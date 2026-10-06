# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise asset writers against real replicated Newton state buffers without Kit."""

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView

from isaaclab.physics import PhysicsManager

pytestmark = pytest.mark.unit


@pytest.fixture(params=["cpu", "cuda:0"])
def bound_articulation(request, monkeypatch):
    """Bind an asset to three real worlds; USD import is outside this writer test."""
    device = request.param
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    source = newton.ModelBuilder()
    root, tip = [source.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
    source.add_articulation([source.add_joint_free(root), source.add_joint_revolute(root, tip)], label="Robot")
    builder = newton.ModelBuilder()
    for _ in range(3):
        builder.add_world(source)
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    for name, value in {
        "_model": model,
        "_state_0": state,
        "_control": model.control(),
        "_world_reset_mask": wp.zeros(3, dtype=wp.int32, device=device),
        "_fk_reset_mask": wp.zeros(model.articulation_count, dtype=wp.bool, device=device),
        "_model_changes": set(),
    }.items():
        monkeypatch.setattr(NewtonManager, name, value)
    monkeypatch.setattr(PhysicsManager, "_device", device)
    asset = Articulation.__new__(Articulation)
    asset._device = device
    asset._check_shapes = True
    asset._root_view = ArticulationView(model, "Robot", exclude_joint_types=[newton.JointType.FREE])
    asset._data = ArticulationData(asset._root_view, device)
    asset.data._apply_ordering_maps_after_resolve()
    asset._ALL_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    asset._ALL_ENV_MASK = wp.ones(3, dtype=wp.bool, device=device)
    asset._ALL_JOINT_INDICES = wp.array([0], dtype=wp.int32, device=device)
    asset._ALL_JOINT_MASK = wp.ones(1, dtype=wp.bool, device=device)
    asset._ALL_BODY_INDICES = wp.array([0, 1], dtype=wp.int32, device=device)
    asset._ALL_BODY_MASK = wp.ones(2, dtype=wp.bool, device=device)
    yield asset, model, state


@pytest.mark.parametrize("selection", ["index", "mask"])
def test_joint_reset_preserves_other_world_body_state(bound_articulation, selection):
    """Reset one world without projecting its peers' maximal body state through FK."""
    asset, model, state = bound_articulation
    # A maximal-coordinate solver can leave bodies off the FK manifold. Its peer
    # reset must preserve those actual poses and velocities, not reconstruct them.
    poses = state.body_q.numpy()
    poses[4:, 2] += 0.03
    state.body_q.assign(poses)
    velocities = state.body_qd.numpy()
    velocities[4:, 0] = 0.7
    state.body_qd.assign(velocities)
    if selection == "index":
        asset.write_joint_state_to_sim_index(
            position=torch.tensor([[0.25]], device=asset.device),
            velocity=torch.zeros((1, 1), device=asset.device),
            env_ids=[1],
        )
    else:
        asset.write_joint_state_to_sim_mask(
            position=torch.full((3, 1), 0.25, device=asset.device),
            velocity=torch.zeros((3, 1), device=asset.device),
            env_mask=wp.array([False, True, False], dtype=wp.bool, device=asset.device),
        )
    NewtonManager.forward()
    np.testing.assert_array_equal(state.body_q.numpy()[4:], poses[4:])
    np.testing.assert_array_equal(state.body_qd.numpy()[4:], velocities[4:])
    np.testing.assert_allclose(asset.data.joint_pos.warp.numpy().ravel(), [0.0, 0.25, 0.0])


@pytest.mark.parametrize("selection", ["index", "mask"])
def test_scalar_drive_writes_update_native_model_buffers(bound_articulation, selection):
    """Explicit actuator initialization zeros drives through native Warp arrays."""
    asset, model, _ = bound_articulation
    if selection == "index":
        asset.write_joint_stiffness_to_sim_index(stiffness=12.0, env_ids=[1])
        asset.write_joint_damping_to_sim_index(damping=3.0, env_ids=[1])
    else:
        mask = wp.array([False, True, False], dtype=wp.bool, device=asset.device)
        asset.write_joint_stiffness_to_sim_mask(stiffness=12.0, env_mask=mask)
        asset.write_joint_damping_to_sim_mask(damping=3.0, env_mask=mask)
    np.testing.assert_array_equal(model.joint_target_ke.numpy().reshape(3, -1)[:, -1], [0.0, 12.0, 0.0])
    np.testing.assert_array_equal(model.joint_target_kd.numpy().reshape(3, -1)[:, -1], [0.0, 3.0, 0.0])


def test_forward_preserves_solver_state_without_pending_writes(bound_articulation):
    """Observation and render refreshes leave unmodified maximal body state intact."""
    _, _, state = bound_articulation
    poses = state.body_q.numpy()
    poses[:, 2] += 0.03
    velocities = state.body_qd.numpy()
    velocities[:, 0] = 0.7
    state.body_q.assign(poses)
    state.body_qd.assign(velocities)
    NewtonManager.forward()
    np.testing.assert_array_equal(state.body_q.numpy(), poses)
    np.testing.assert_array_equal(state.body_qd.numpy(), velocities)


@pytest.mark.parametrize("selection", ["index", "mask"])
def test_com_pose_writes_accept_release_event_data(bound_articulation, selection):
    """COM randomization accepts seven-component poses and updates selected bodies."""
    asset, model, _ = bound_articulation
    coms = torch.zeros((3, 2, 7), device=asset.device)
    coms[..., :3] = torch.tensor([0.1, 0.2, 0.3], device=asset.device)
    coms[..., 6] = 1.0
    if selection == "index":
        asset.set_coms_index(coms=coms[1:2, 1:2], env_ids=[1], body_ids=[1])
    else:
        asset.set_coms_mask(
            coms=coms,
            env_mask=wp.array([False, True, False], dtype=wp.bool, device=asset.device),
            body_mask=wp.array([False, True], dtype=wp.bool, device=asset.device),
        )
    expected = np.zeros((6, 3))
    expected[3] = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(model.body_com.numpy(), expected, atol=1e-7)
