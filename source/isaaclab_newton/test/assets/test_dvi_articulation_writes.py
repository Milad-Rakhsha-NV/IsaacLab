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
from isaaclab_newton.physics import DVISolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.dvi_manager import NewtonDVIManager
from newton.selection import ArticulationView

from isaaclab.physics import PhysicsManager

pytestmark = pytest.mark.unit


@pytest.fixture(params=["cpu", "cuda:0"])
def bound_articulation(request, monkeypatch):
    """Bind an asset to three real worlds; USD import is outside this writer test."""
    device, fixed = request.param if isinstance(request.param, tuple) else (request.param, False)
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    source = newton.ModelBuilder()
    root, tip = [source.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
    root_joint = source.add_joint_fixed(-1, root) if fixed else source.add_joint_free(root)
    hinge = source.add_joint_revolute(root, tip, axis=(0.0, 0.0, 1.0))
    source.add_articulation([root_joint, hinge], label="Robot")
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
        "_solver": None,
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


@pytest.mark.parametrize(
    "bound_articulation, selection",
    [
        (("cpu", False), "index32"),
        (("cpu", True), "index64"),
        (("cuda:0", False), "mask"),
        (("cuda:0", True), "warp64"),
    ],
    indirect=["bound_articulation"],
)
def test_root_pose_writes_refresh_derived_state(bound_articulation, selection):
    """Refresh link/COM caches on both root types without projecting unselected worlds."""
    asset, model, state = bound_articulation
    coms = torch.zeros((3, 2, 3), device=asset.device)
    coms[:, 0, 0] = 0.2
    asset.set_coms_index(coms=coms)
    # Keep a peer off the FK manifold to detect a global instead of selected reset.
    poses = state.body_q.numpy()
    poses[4:, 2] += 0.03
    state.body_q.assign(poses)
    for location in ("link", "com"):
        _ = asset.data.root_com_pose_w.torch.clone()
        _ = asset.data.heading_w.torch.clone()
        quat = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)] if location == "link" else [0.0, 0.0, 0.0, 1.0]
        pose = torch.tensor([[1.0, 2.0, 3.0, *quat]], dtype=torch.float32, device=asset.device)
        if selection == "mask":
            writer = getattr(asset, f"write_root_{location}_pose_to_sim_mask")
            writer(
                root_pose=pose.repeat(3, 1), env_mask=wp.array([False, True, False], dtype=wp.bool, device=asset.device)
            )
        else:
            ids = (
                wp.array([1], dtype=wp.int64, device=asset.device)
                if selection == "warp64"
                else torch.tensor(
                    [1], dtype=torch.int32 if selection == "index32" else torch.int64, device=asset.device
                )
            )
            getattr(asset, f"write_root_{location}_pose_to_sim_index")(root_pose=pose, env_ids=ids)
        NewtonManager.forward()
        expected_link = [1.0, 2.0, 3.0, *quat] if location == "link" else [0.8, 2.0, 3.0, *quat]
        expected_com = [1.0, 2.2, 3.0, *quat] if location == "link" else [1.0, 2.0, 3.0, *quat]
        np.testing.assert_allclose(state.body_q.numpy()[2], expected_link, atol=1e-6)
        np.testing.assert_allclose(asset.data.root_com_pose_w.warp.numpy()[1], expected_com, atol=1e-6)
        np.testing.assert_allclose(
            asset.data.heading_w.warp.numpy()[1], np.pi / 2 if location == "link" else 0.0, atol=1e-6
        )
        np.testing.assert_array_equal(state.body_q.numpy()[4:], poses[4:])


@pytest.mark.parametrize("selection", ["index", "mask"])
def test_velocity_only_writes_reach_native_body_state(bound_articulation, selection):
    """Apply root and joint velocity writes without requiring a simultaneous pose reset."""
    asset, _, state = bound_articulation
    coms = torch.zeros((3, 2, 3), device=asset.device)
    coms[:, 0, 0] = 0.2
    asset.set_coms_index(coms=coms)
    # Prime cached link velocities before each write at the same simulation time.
    mask = wp.array([False, True, False], dtype=wp.bool, device=asset.device)
    for location, velocity, expected in (
        ("com", [1.0, 2.0, 3.0, 0.0, 0.0, 2.0], [1.0, 2.0, 3.0, 0.0, 0.0, 2.0]),
        ("link", [4.0, 5.0, 6.0, 0.0, 0.0, 3.0], [4.0, 5.6, 6.0, 0.0, 0.0, 3.0]),
    ):
        _ = asset.data.root_link_vel_w.torch.clone()
        data = torch.tensor([velocity], device=asset.device)
        if selection == "index":
            getattr(asset, f"write_root_{location}_velocity_to_sim_index")(root_velocity=data, env_ids=[1])
        else:
            getattr(asset, f"write_root_{location}_velocity_to_sim_mask")(
                root_velocity=data.repeat(3, 1), env_mask=mask
            )
        NewtonManager.forward()
        np.testing.assert_allclose(state.body_qd.numpy()[2], expected, atol=1e-6)
        expected_link = np.array(expected)
        expected_link[1] -= 0.2 * expected[5]
        np.testing.assert_allclose(asset.data.root_link_vel_w.warp.numpy()[1], expected_link, atol=1e-6)
    if selection == "index":
        asset.write_joint_velocity_to_sim_index(velocity=torch.tensor([[0.5]], device=asset.device), env_ids=[1])
    else:
        asset.write_joint_velocity_to_sim_mask(velocity=torch.full((3, 1), 0.5, device=asset.device), env_mask=mask)
    NewtonManager.forward()
    np.testing.assert_allclose(state.body_qd.numpy()[3, 5], 3.5, atol=1e-6)
    np.testing.assert_array_equal(state.body_qd.numpy()[[0, 1, 4, 5]], 0.0)


@pytest.mark.parametrize("bound_articulation", [("cpu", True)], indirect=True)
def test_fixed_root_has_no_velocity_dofs(bound_articulation):
    """Keep a fixed root stationary while allowing its hinge to move after a velocity write."""
    asset, _, state = bound_articulation
    for location in ("com", "link"):
        for selection in ("index", "mask"):
            kwargs = (
                {"env_ids": [1]}
                if selection == "index"
                else {"env_mask": wp.array([False, True, False], dtype=wp.bool, device=asset.device)}
            )
            count = 1 if selection == "index" else 3
            getattr(asset, f"write_root_{location}_velocity_to_sim_{selection}")(
                root_velocity=torch.ones((count, 6), device=asset.device), **kwargs
            )
            np.testing.assert_array_equal(asset.data.root_com_vel_w.warp.numpy(), 0.0)
    asset.write_joint_velocity_to_sim_index(velocity=torch.tensor([[0.5]], device=asset.device), env_ids=[1])
    NewtonManager.forward()
    np.testing.assert_allclose(state.body_qd.numpy()[3, 5], 0.5)
    np.testing.assert_allclose(asset.data.body_com_vel_w.warp.numpy()[1, 1, 5], 0.5)
    np.testing.assert_array_equal(state.body_qd.numpy()[[0, 1, 2, 4, 5]], 0.0)


@pytest.mark.parametrize("bound_articulation", ["cpu"], indirect=True)
@pytest.mark.parametrize("substeps", [1, 2, 3])
def test_eager_dvi_steps_publish_current_asset_state(bound_articulation, monkeypatch, substeps):
    """Keep asset bindings current after each eager physics step, including odd substep counts."""
    asset, model, _ = bound_articulation
    cfg = NewtonCfg(solver_cfg=DVISolverCfg(), num_substeps=substeps, use_cuda_graph=False)
    for name, value in {
        "_state_1": model.state(),
        "_collision_pipeline": None,
        "_contacts": None,
        "_collision_cfg": None,
        "_needs_collision_pipeline": False,
        "_use_single_state": False,
        "_num_substeps": substeps,
        "_solver_dt": 0.01,
        "_graph": None,
        "_usdrt_stage": None,
        "_newton_frame_transform_sensors": [],
        "_newton_imu_sensors": [],
        "_report_contacts": False,
    }.items():
        monkeypatch.setattr(NewtonManager, name, value)
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg)
    monkeypatch.setattr(PhysicsManager, "_sim", None)
    NewtonDVIManager.initialize_solver()
    before = asset.data.root_link_pose_w.warp.numpy().copy()
    for _ in range(3):
        NewtonDVIManager._simulate_physics_only()
        state = NewtonManager.get_state_0()
        np.testing.assert_allclose(asset.data.root_link_pose_w.warp.numpy(), state.body_q.numpy()[::2], atol=1e-6)
        np.testing.assert_allclose(asset.data.root_com_vel_w.warp.numpy(), state.body_qd.numpy()[::2], atol=1e-6)
    assert np.all(asset.data.root_link_pose_w.warp.numpy()[:, 2] < before[:, 2])


@pytest.mark.parametrize("bound_articulation", ["cuda:0"], indirect=True)
def test_velocity_reset_mask_changes_on_graph_replay(bound_articulation):
    """Replay velocity resets with a changing world mask without rewriting peer states."""
    asset, _, state = bound_articulation
    mask = wp.zeros(3, dtype=wp.bool, device=asset.device)
    velocity = wp.zeros(3, dtype=wp.spatial_vector, device=asset.device)
    asset.write_root_com_velocity_to_sim_mask(root_velocity=velocity, env_mask=mask)
    NewtonManager.forward()
    with wp.ScopedCapture(device=asset.device) as capture:
        asset.write_root_com_velocity_to_sim_mask(root_velocity=velocity, env_mask=mask)
        NewtonManager.forward()
    for world, speed in ((1, 0.7), (2, -0.4)):
        selected = np.zeros(3, dtype=bool)
        selected[world] = True
        mask.assign(selected)
        values = np.zeros((3, 6), dtype=np.float32)
        values[:, 0] = speed
        velocity.assign(values)
        before = state.body_qd.numpy()
        wp.capture_launch(capture.graph)
        expected = before.copy()
        expected[2 * world : 2 * world + 2, 0] = speed
        np.testing.assert_allclose(state.body_qd.numpy(), expected, atol=1e-6)


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
