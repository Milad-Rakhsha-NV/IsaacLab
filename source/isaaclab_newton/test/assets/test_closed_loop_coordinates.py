# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop views expose release coordinate widths, including quaternion joints."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.articulation.closed_loop_view import ClosedLoopView
from isaaclab_newton.physics import NewtonManager


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_replicated_closed_loop_coordinate_widths(device, monkeypatch):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    source = newton.ModelBuilder()
    links = [source.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(4)]
    source.add_articulation(
        [
            source.add_joint_free(links[0]),
            source.add_joint_revolute(links[0], links[1]),
            source.add_joint_prismatic(links[1], links[2]),
            source.add_joint_fixed(links[2], links[3]),
        ]
    )
    source.add_joint_ball(links[3], links[1])
    builder = newton.ModelBuilder()
    for _ in range(3):
        builder.add_world(source)
    model = builder.finalize(device=device)
    state = model.state()
    coordinates = state.joint_q.numpy().reshape(3, -1)
    coordinates[:, 7:9] = [0.1, 0.2]
    target = np.array([0.2, -0.3, 0.4])
    coordinates[:, -4:] = np.asarray(
        wp.quat_from_axis_angle(wp.vec3(target / np.linalg.norm(target)), float(np.linalg.norm(target)))
    )
    state.joint_q.assign(coordinates.ravel())
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_state_0", state)
    monkeypatch.setattr(NewtonManager, "_control", model.control())
    view = ClosedLoopView(model)
    data = ArticulationData(view, device)
    assert view.joint_coord_counts == [1, 1, 0, 4]
    assert view.joint_dof_counts == [1, 1, 0, 3]
    assert sum(view.joint_coord_counts) == view.joint_coord_count == 6
    assert sum(view.joint_dof_counts) == view.joint_dof_count == 5
    assert data.joint_pos.shape == (3, 5)
    assert data.joint_vel.shape == (3, 5)
    np.testing.assert_allclose(data.joint_pos.warp.numpy(), np.tile([0.1, 0.2, *target], (3, 1)), atol=1e-6)
