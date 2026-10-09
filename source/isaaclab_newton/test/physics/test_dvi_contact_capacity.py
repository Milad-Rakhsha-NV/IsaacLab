# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every replicated world's ground contact must reach the DVI solve."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import DVISolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_newton.physics.dvi_manager import NewtonDVIManager

from isaaclab.physics import PhysicsManager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("capacity", [None, 2048], ids=["automatic", "explicit"])
def test_all_worlds_receive_contact_response(monkeypatch, capacity):
    """A batch larger than the native solver fallback must not lose tail contacts."""
    source = newton.ModelBuilder()
    body = source.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
    source.add_shape_sphere(body, radius=0.1)
    source.add_articulation([source.add_joint_free(body)])
    builder = newton.ModelBuilder()
    for world in range(1024):
        builder.add_world(source, xform=wp.transform(wp.vec3(float(world), 0.0, 0.0), wp.quat_identity()))
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    cfg = NewtonCfg(
        solver_cfg=DVISolverCfg(contact_max_iterations=30, coupling_iterations=1),
        collision_cfg=None if capacity is None else NewtonCollisionPipelineCfg(rigid_contact_max=capacity),
        use_cuda_graph=False,
    )
    for name, value in {
        "_model": model,
        "_state_0": state_in,
        "_state_1": state_out,
        "_control": model.control(),
        "_solver": None,
        "_collision_pipeline": None,
        "_contacts": None,
        "_collision_cfg": None,
        "_needs_collision_pipeline": False,
        "_use_single_state": False,
        "_num_substeps": 1,
        "_solver_dt": 1.0 / 60.0,
        "_graph": None,
        "_usdrt_stage": None,
    }.items():
        monkeypatch.setattr(NewtonManager, name, value)
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(PhysicsManager, "_sim", None)

    NewtonDVIManager.initialize_solver()
    NewtonDVIManager._step_solver(state_in, state_out, NewtonManager._control, 1.0 / 60.0)

    assert int(NewtonManager._contacts.rigid_contact_count.numpy()[0]) == 1024
    # All slightly penetrating spheres must be pushed upward, including those
    # beyond contact 1000. Without the capacity repair, 24 spheres fall instead.
    height_change = state_out.body_q.numpy()[:, 2] - state_in.body_q.numpy()[:, 2]
    assert np.all(height_change > 0.0), f"{np.count_nonzero(height_change <= 0.0)} worlds lost contact response"
