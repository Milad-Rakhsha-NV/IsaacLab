# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry contracts for selective RFT surfaces, independent of rigid integration."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from isaaclab_newton.physics.rft_solver_cfg import RFTSurfaceResolutionCfg

pytestmark = pytest.mark.kitless


def test_contact_surface_resolution_preserves_shape_and_targets():
    """Refine a foot and distal cylinder while preserving authored geometry and the torso."""
    builder = newton.ModelBuilder()
    foot = builder.add_body(mass=1.0, label="/Robot/FL_foot")
    calf = builder.add_body(mass=1.0, label="/Robot/FL_calf")
    torso = builder.add_body(mass=1.0, label="/Robot/base")
    offset = wp.transform(wp.vec3(0.01, -0.02, 0.03), wp.quat_rpy(0.3, -0.2, 0.5))
    builder.add_shape_sphere(body=foot, radius=0.022, xform=offset, label="/Robot/FL_foot/sphere")
    builder.add_shape_cylinder(body=calf, radius=0.011, half_height=0.0325, label="/Robot/FL_calf/lower")
    builder.add_shape_box(body=torso, hx=0.2, hy=0.1, hz=0.05, label="/Robot/base/box")
    model = builder.finalize(device="cpu")
    masses, scales = model.body_mass.numpy().copy(), model.shape_scale.numpy().copy()
    surfaces, _ = collision_surfaces(
        model,
        ".*",
        12,
        resolutions=[
            RFTSurfaceResolutionCfg(shape_pattern=".*/FL_foot/.*", sphere_subdivisions=2),
            dict(shape_pattern=".*/FL_calf/lower", segments=16, axial_segments=3),
        ],
    )
    meshes = dict(surfaces)
    sphere = meshes[foot]
    assert len(sphere.indices) // 3 == 128
    local = np.array([wp.transform_point(wp.transform_inverse(offset), wp.vec3(*v)) for v in sphere.vertices])
    np.testing.assert_allclose(np.linalg.norm(local, axis=1), 0.022, rtol=1e-5, atol=1e-8)
    triangles = local[sphere.indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0)
    cylinder = meshes[calf]
    radial = np.linalg.norm(cylinder.vertices[:, :2], axis=-1)
    np.testing.assert_allclose(radial[radial > 1e-6], 0.011, rtol=1e-5)
    assert np.max(np.abs(cylinder.vertices[:, 2])) == pytest.approx(0.0325)
    triangles = cylinder.vertices[cylinder.indices.reshape(-1, 3)]
    assert np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=-1).max() < 0.023
    assert len(meshes[torso].indices) // 3 == 12
    np.testing.assert_allclose(np.max(np.abs(meshes[torso].vertices), axis=0), (0.2, 0.1, 0.05))
    np.testing.assert_array_equal(model.body_mass.numpy(), masses)
    np.testing.assert_array_equal(model.shape_scale.numpy(), scales)


def test_resolution_cache_distinguishes_identical_colliders():
    """Identical sphere geometry can have different per-collider integration budgets."""
    builder = newton.ModelBuilder()
    for name in ("foot_a", "foot_b"):
        body = builder.add_body(mass=1.0, label=name)
        builder.add_shape_sphere(body=body, radius=0.022, label=name)
    model = builder.finalize(device="cpu")
    surfaces, _ = collision_surfaces(
        model,
        ".*",
        12,
        resolutions=[
            dict(shape_pattern="foot_a", sphere_subdivisions=2),
            dict(shape_pattern="foot_b", sphere_subdivisions=1),
        ],
    )
    assert [len(mesh.indices) // 3 for _, mesh in surfaces] == [128, 32]


@pytest.mark.parametrize(
    "patterns, message",
    [
        (["missing"], "matched no selected collider"),
        (["foot", ".*"], "Multiple RFT resolution rules"),
    ],
)
def test_resolution_rules_reject_missing_or_conflicting_targets(patterns, message):
    """An asset rename or overlapping selector must not silently change the mesh profile."""
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, label="foot")
    builder.add_shape_sphere(body=body, radius=0.022, label="foot")
    model = builder.finalize(device="cpu")
    with pytest.raises(ValueError, match=message):
        collision_surfaces(
            model, ".*", 12, resolutions=[dict(shape_pattern=p, sphere_subdivisions=2) for p in patterns]
        )
