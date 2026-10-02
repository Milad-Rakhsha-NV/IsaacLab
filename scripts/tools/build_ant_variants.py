# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export reproducible three-/five-legged Ant USD assets without launching simulation."""

import argparse
import shutil
from pathlib import Path

from isaaclab_assets.robots.ant_variants import ant_variant_cfg, build_ant_variant


def preview_assets(output_dir: Path):
    """Plot orthographic USD collision geometry at nominal joint positions (no learned motion)."""
    import math
    import re

    import matplotlib
    import numpy as np

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_assets.robots.ant import ANT_CFG

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for column, legs in enumerate((3, 4, 5)):
        cfg = ANT_CFG if legs == 4 else ant_variant_cfg(legs)
        path = retrieve_file_path(cfg.spawn.usd_path) if legs == 4 else str(output_dir / f"ant_{legs}.usda")
        stage = Usd.Stage.Open(path)
        transforms = UsdGeom.XformCache()
        posed = {"/ant/torso": Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.6))}
        pending = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        while pending:
            remaining = []
            for joint in pending:
                parent, child = str(joint.GetBody0Rel().GetTargets()[0]), str(joint.GetBody1Rel().GetTargets()[0])
                if parent not in posed:
                    remaining.append(joint)
                    continue
                angle = next(
                    value
                    for name, value in cfg.init_state.joint_pos.items()
                    if re.fullmatch(name, joint.GetPrim().GetName())
                )
                frame0 = Gf.Matrix4d().SetRotate(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
                frame0.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
                frame1 = Gf.Matrix4d().SetRotate(Gf.Quatd(joint.GetLocalRot1Attr().Get()))
                frame1.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
                motion = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(angle)))
                posed[child] = frame1.GetInverse() * motion * frame0 * posed[parent]
            if len(remaining) == len(pending):
                raise ValueError("Cannot resolve the asset's kinematic tree.")
            pending = remaining
        for body_path, world in posed.items():
            body = stage.GetPrimAtPath(body_path)
            body_inverse = transforms.GetLocalToWorldTransform(body).GetInverse()
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                matrix = transforms.GetLocalToWorldTransform(prim) * body_inverse * world
                if prim.IsA(UsdGeom.Capsule):
                    shape = UsdGeom.Capsule(prim)
                    radius, half_length = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get() / 2
                    direction = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[
                        shape.GetAxisAttr().Get()
                    ]
                    endpoints = np.array([matrix.Transform(direction * sign * half_length) for sign in (-1, 1)])
                elif prim.IsA(UsdGeom.Sphere):
                    radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    endpoints = np.array([matrix.Transform(Gf.Vec3d(0))] * 2)
                else:
                    continue
                color = "#e58b2b" if "torso_geom" in prim.GetName() else "#277da8"
                if "radial_4" in body_path or "aux_5" in prim.GetName():
                    color = "#b54757"
                for row, coordinates in enumerate(((0, 1), (0, 2))):
                    ax = axes[row, column]
                    a, b = endpoints[:, coordinates]
                    delta = b - a
                    normal = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1e-12) * radius
                    ax.add_patch(Polygon([a + normal, b + normal, b - normal, a - normal], color=color, alpha=0.9))
                    for center in (a, b):
                        ax.add_patch(Circle(center, radius, color=color, alpha=0.9))
        for row in (0, 1):
            ax = axes[row, column]
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim((-1.05, 1.05) if row == 0 else (-0.05, 1.05))
            ax.set_xlabel("X / forward (m)")
            ax.set_ylabel("Y (m)" if row == 0 else "Z (m)")
            ax.grid(alpha=0.2)
            if row == 1:
                ax.axhline(0, color="#465363", linewidth=1)
        axes[0, column].set_title(f"{legs} legs · {2 * legs} joints")
    fig.suptitle(
        "Ant morphology baselines — nominal reset pose\n"
        "USD collision geometry; radial spacing: 120° / 72°; fifth branch in red",
        fontsize=15,
    )
    fig.savefig(output_dir / "morphologies.png", dpi=180)
    fig.savefig(output_dir / "morphologies.svg")
    plt.close(fig)


def main():
    """Materialize cached derivatives in a user-selected directory for inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/ant_variants"))
    parser.add_argument("--preview", action="store_true", help="Also plot nominal collision geometry as PNG/SVG.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for legs in (3, 5):
        source = build_ant_variant(legs)
        destination = args.output_dir / f"ant_{legs}.usda"
        shutil.copyfile(source, destination)
        print(destination.resolve())
    if args.preview:
        preview_assets(args.output_dir)


if __name__ == "__main__":
    main()
