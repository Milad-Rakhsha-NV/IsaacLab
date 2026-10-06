# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve DVI import and self-collision settings across clone-plan replication."""

import newton
import pytest
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from isaaclab_newton.cloner.replicate import _configure_collision_shapes
from isaaclab_newton.physics import NewtonCfg

from pxr import Usd, UsdGeom, UsdPhysics


@pytest.mark.parametrize("collapse", [False, True])
def test_fixed_joint_collapse_reaches_usd_importer(collapse):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("base", "foot"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        shape = UsdGeom.Cube.Define(stage, f"/Robot/{name}/shape")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed")
    joint.CreateBody0Rel().SetTargets(["/Robot/base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/foot"])
    builder = build_source_builders(
        stage, ["/Robot"], newton.ModelBuilder, [], collapse_fixed_joints=collapse, load_visual_shapes=False
    )["/Robot"]
    model = builder.finalize(device="cpu")
    assert model.body_count == (1 if collapse else 2)
    assert model.shape_count == 2
    assert len(set(model.shape_body.numpy())) == model.body_count


@pytest.mark.parametrize("hops,disable", [(0, False), (2, False), (0, True)])
def test_self_collision_filters_survive_world_replication(hops, disable):
    source = newton.ModelBuilder()
    links = [source.add_link(mass=1.0) for _ in range(4)]
    for link in links:
        source.add_shape_sphere(link, radius=0.1)
    joints = [source.add_joint_free(links[0])]
    for parent, child in zip(links[:-1], links[1:], strict=True):
        joints.append(source.add_joint_revolute(parent, child, collision_filter_parent=False))
    source.add_articulation(joints)
    before = {tuple(sorted(pair)) for pair in source.shape_collision_filter_pairs}
    cfg = NewtonCfg(jointed_self_collision_filter_hops=hops, disable_robot_self_collisions=disable)
    _configure_collision_shapes(source, cfg)
    expected = {
        (left, right) for left in range(4) for right in range(left + 1, 4) if disable or right - left <= hops
    } | before
    replicated = newton.ModelBuilder()
    for _ in range(3):
        replicated.begin_world()
        replicated.add_builder(source)
        replicated.end_world()
    model = replicated.finalize(device="cpu")
    actual = {tuple(sorted(pair)) for pair in model.shape_collision_filter_pairs}
    assert actual == {(left + 4 * world, right + 4 * world) for world in range(3) for left, right in expected}
