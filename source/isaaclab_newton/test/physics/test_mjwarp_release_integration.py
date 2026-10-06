# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Protect MJWarp construction and contact reporting on the merged native manager."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager, NewtonMJWarpManager

from isaaclab.physics import PhysicsManager
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mjwarp_manager_constructs_and_steps_native_model(monkeypatch, device):
    """Resolved cfg metadata must not prevent native MJWarp construction or contacts."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    body = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.end_world()
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    cfg = MJWarpSolverCfg(iterations=3, ls_iterations=2)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_cfg=cfg))
    monkeypatch.setattr(PhysicsManager, "_device", device)
    for name in ("_solver", "_contacts", "_use_single_state", "_needs_collision_pipeline"):
        monkeypatch.setattr(NewtonManager, name, getattr(NewtonManager, name))
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_state_0", state)
    NewtonMJWarpManager._build_solver(model, cfg)
    NewtonMJWarpManager._initialize_contacts()
    assert NewtonManager._contacts is not None
    for _ in range(10):
        NewtonManager._solver.step(state, state, model.control(), None, 0.01)
    assert np.isfinite(state.body_q.numpy()).all()
    assert 0.8 < state.body_q.numpy()[body, 2] < 0.99
    # Reset only solver-owned warm-start buffers; preserve the authored state.
    joint_q, joint_qd = state.joint_q.numpy(), state.joint_qd.numpy()
    NewtonManager._solver.mjw_data.qacc_warmstart.fill_(1.0)
    NewtonMJWarpManager._reset_solver_internals(wp.array([True, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(state.joint_q.numpy(), joint_q)
    np.testing.assert_array_equal(state.joint_qd.numpy(), joint_qd)
    assert not NewtonManager._solver.mjw_data.qacc_warmstart.numpy().any()
