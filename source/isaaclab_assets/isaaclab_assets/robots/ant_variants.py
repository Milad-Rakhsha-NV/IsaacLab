# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible three- and five-legged derivatives of the stock USD Ant.

Identical stock front-left branches are distributed uniformly around the torso: 120 degrees
for three legs and 72 degrees for five, starting at +X. Link geometry, density, joint limits
and actuator settings are inherited; all ankle coordinates use the same bend convention.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Literal

from filelock import FileLock

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import retrieve_file_path

from .ant import ANT_CFG


def build_ant_variant(num_legs: Literal[3, 5], source_path: str | None = None, cache_dir: str | None = None) -> str:
    """Generate a self-contained USD in a content-addressed cache and return its path.

    Resolves the stock asset with the normal Isaac Lab asset routing. Source files are never
    modified. The cache key includes the composed source and generator revision, so source
    changes cannot silently reuse an old derivative. No network work occurs at module import.
    """
    if num_legs not in (3, 5):
        raise ValueError("Only three- and five-legged derivatives are supported; use ANT_CFG for four legs.")
    source_path = source_path or ANT_CFG.spawn.usd_path
    source = Usd.Stage.Open(retrieve_file_path(source_path))
    if source is None or source.GetDefaultPrim().GetPath() != Sdf.Path("/ant"):
        raise ValueError("Expected the stock Ant asset with default prim /ant.")
    # Edit a private session layer, not a shared stage or any authored source layer.
    source.SetEditTarget(source.GetSessionLayer())
    for prim in list(source.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    layer = source.Flatten(False)
    layer.customLayerData = {}
    layer.documentation = ""
    source_fingerprint = hashlib.sha256(layer.ExportToString().encode()).hexdigest()
    generator_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(f"{source_fingerprint}:{generator_fingerprint}".encode()).hexdigest()[:20]
    directory = Path(cache_dir or Path(tempfile.gettempdir()) / "isaaclab_ant_variants") / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"ant_{num_legs}.usda"
    with FileLock(str(output) + ".lock"):
        if output.exists():
            return str(output)
        stage = Usd.Stage.Open(layer)
        required = [
            f"/ant/{prefix}{name}"
            for prefix in ("", "joints/")
            for name in ("front_left_leg", "front_left_foot", "right_back_leg", "right_back_foot")
        ] + [f"/ant/torso/{group}/aux_{index}_geom" for group in ("collisions", "visuals") for index in (1, 4)]
        if any(not stage.GetPrimAtPath(path) for path in required):
            raise ValueError(
                "Stock Ant branch layout changed; required source links, joints or hip shapes are missing."
            )
        stage.RemovePrim("/physicsScene")
        # Keep an immutable branch template while replacing every stock attachment. A single
        # template gives every radial leg identical local axes, ankle signs, limits and mechanics.
        template = Sdf.Layer.CreateAnonymous()
        Sdf.CopySpec(layer, "/ant", template, "/ant")
        for prefix in ("front_left", "front_right", "left_back", "right_back"):
            for suffix in ("leg", "foot"):
                stage.RemovePrim(f"/ant/{prefix}_{suffix}")
                stage.RemovePrim(f"/ant/joints/{prefix}_{suffix}")
        for group in ("collisions", "visuals"):
            for index in range(1, 5):
                stage.RemovePrim(f"/ant/torso/{group}/aux_{index}_geom")
        for index in range(num_legs):
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 360.0 * index / num_legs - 45.0)
            transform = Gf.Matrix4d().SetRotate(rotation)
            for suffix in ("leg", "foot"):
                old, new = f"front_left_{suffix}", f"radial_{index}_{suffix}"
                Sdf.CopySpec(template, f"/ant/{old}", layer, f"/ant/{new}")
                body = UsdGeom.Xformable(stage.GetPrimAtPath(f"/ant/{new}"))
                matrix = body.GetLocalTransformation()
                body.MakeMatrixXform().Set(matrix * transform)
                Sdf.CopySpec(template, f"/ant/joints/{old}", layer, f"/ant/joints/{new}")
                joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"/ant/joints/{new}"))
                parent = "torso" if suffix == "leg" else f"radial_{index}_leg"
                joint.GetBody0Rel().SetTargets([f"/ant/{parent}"])
                joint.GetBody1Rel().SetTargets([f"/ant/{new}"])
                if suffix == "leg":
                    # Only the hip's torso frame changes; child-local joint frames stay fixed.
                    position = joint.GetLocalPos0Attr().Get()
                    joint.GetLocalPos0Attr().Set(Gf.Vec3f(rotation.TransformDir(Gf.Vec3d(position))))
                    joint.GetLocalRot0Attr().Set(Gf.Quatf(rotation.GetQuat()) * joint.GetLocalRot0Attr().Get())
            for group in ("collisions", "visuals"):
                old = f"/ant/torso/{group}/aux_1_geom"
                new = f"/ant/torso/{group}/aux_{index + 1}_geom"
                Sdf.CopySpec(template, old, layer, new)
                shape = UsdGeom.Xformable(stage.GetPrimAtPath(new))
                matrix = shape.GetLocalTransformation()
                shape.MakeMatrixXform().Set(matrix * transform)
        # Refresh stock semantic inventories rather than retaining dangling or incomplete relationships.
        root = stage.GetDefaultPrim()
        bodies = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
        if len(bodies) != 1 + 2 * num_legs or len(joints) != 2 * num_legs:
            raise ValueError("Stock Ant topology changed; re-audit the variant generator before using this source.")
        root.GetRelationship("isaac:physics:robotLinks").SetTargets(bodies)
        root.GetRelationship("isaac:physics:robotJoints").SetTargets(joints)
        layer.customLayerData = {
            "sourceAsset": source_path,
            "sourceFingerprint": source_fingerprint,
            "generatorFingerprint": generator_fingerprint,
            "buildFingerprint": fingerprint,
            "generator": "isaaclab_assets.robots.ant_variants:v2-radial",
            "numLegs": num_legs,
        }
        temporary = directory / f"ant_{num_legs}.tmp.usda"
        layer.Export(str(temporary))
        temporary.replace(output)
    return str(output)


def spawn_ant_variant(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Generate the derivative lazily, then use the standard USD spawn/clone lifecycle."""
    path = build_ant_variant(cfg.num_legs, source_path=cfg.usd_path)
    return sim_utils.spawn_from_usd(
        prim_path, replace(cfg, usd_path=path), translation=translation, orientation=orientation, **kwargs
    )


@configclass
class AntVariantUsdCfg(sim_utils.UsdFileCfg):
    """USD spawner retaining the source Ant path and the requested structural variant."""

    func = spawn_ant_variant
    num_legs: Literal[3, 5] = 3


def ant_variant_cfg(num_legs: Literal[3, 5]) -> ArticulationCfg:
    """Return a stock-mechanics articulation recipe with the appropriate reset joint names."""
    if num_legs not in (3, 5):
        raise ValueError("Expected 3 or 5 legs.")
    cfg = replace(ANT_CFG)
    cfg.spawn = AntVariantUsdCfg(
        **{key: value for key, value in vars(cfg.spawn).items() if key != "func"}, num_legs=num_legs
    )
    cfg.init_state.joint_pos = {".*_leg": 0.0, ".*_foot": 0.785398}
    return cfg
