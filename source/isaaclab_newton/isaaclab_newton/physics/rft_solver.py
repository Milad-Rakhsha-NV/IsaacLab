# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind all selected collision surfaces to an RFT–MJWarp shared-model solver."""

from __future__ import annotations

import re

import numpy as np
import warp as wp
from newton import Axis, Contacts, GeoType, Mesh, Model, ShapeFlags, State
from newton.solvers import SolverMuJoCo, SolverRFT

from .mjwarp_solver import MJWarpSolver
from .rft_solver_cfg import RFTSolverCfg


def collision_surfaces(model: Model, pattern: str, segments: int) -> tuple[list[tuple[int, Mesh]], list[dict]]:
    """Return body-local surfaces and their source-shape inventory.

    Primitive dimensions and poses come from the finalized collision model.
    This does not modify collision geometry or inertial properties. Overlapping
    collider unions require separate validation; the manifest exposes them.
    """
    body = model.shape_body.numpy()
    kinds = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    poses = model.shape_transform.numpy()
    flags = model.shape_flags.numpy()
    surfaces, inventory = [], []
    cache = {}
    for shape, body_id in enumerate(body):
        if body_id < 0 or not re.fullmatch(pattern, model.body_label[body_id]):
            continue
        if not flags[shape] & int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES):
            continue
        kind = GeoType(int(kinds[shape]))
        dims = scales[shape]
        key = (kind, tuple(dims), tuple(poses[shape]), id(model.shape_source[shape]))
        if key not in cache:
            if kind == GeoType.BOX:
                mesh = Mesh.create_box(*map(float, dims), compute_inertia=False)
            elif kind in (GeoType.SPHERE, GeoType.ELLIPSOID):
                mesh = Mesh.create_sphere(
                    1.0, num_latitudes=segments // 2, num_longitudes=segments, compute_inertia=False
                )
                scale = np.full(3, dims[0]) if kind == GeoType.SPHERE else dims
                mesh = Mesh(mesh.vertices * scale, mesh.indices, compute_inertia=False)
            elif kind == GeoType.CAPSULE:
                mesh = Mesh.create_capsule(
                    float(dims[0]), float(dims[1]), up_axis=Axis.Z, segments=segments, compute_inertia=False
                )
            elif kind == GeoType.CYLINDER:
                mesh = Mesh.create_cylinder(
                    float(dims[0]), float(dims[1]), up_axis=Axis.Z, segments=segments, compute_inertia=False
                )
            elif kind in (GeoType.MESH, GeoType.CONVEX_MESH):
                source = model.shape_source[shape]
                if kind == GeoType.CONVEX_MESH:
                    source = source.compute_convex_hull()
                mesh = Mesh(source.vertices * dims, source.indices, compute_inertia=False)
            else:
                raise ValueError(f"Unsupported RFT collider {kind.name}: {model.shape_label[shape]}")
            pose = wp.transform(wp.vec3(*map(float, poses[shape, :3])), wp.quat(*map(float, poses[shape, 3:])))
            vertices = np.asarray([wp.transform_point(pose, wp.vec3(*map(float, v))) for v in mesh.vertices])
            cache[key] = Mesh(vertices, mesh.indices, compute_inertia=False)
        mesh = cache[key]
        surfaces.append((int(body_id), mesh))
        inventory.append(
            dict(
                shape=shape,
                body=int(body_id),
                body_label=model.body_label[body_id],
                label=model.shape_label[shape],
                kind=kind.name,
                scale=dims.tolist(),
                local_pose=poses[shape].tolist(),
                triangles=len(mesh.indices) // 3,
            )
        )
    if not surfaces:
        raise ValueError(f"No collision surfaces matched RFT body_pattern={pattern!r}")
    return surfaces, inventory


class RFTSolver(MJWarpSolver):
    """Experimental shared-model coupling; MJWarp advances every joint once."""

    solver_class = SolverRFT
    single_state = True
    deterministic = False  # Surface reduction uses floating-point atomics.
    contact_sensors = True  # Discrete rigid collisions; granular loads remain separate body impulses.

    @classmethod
    def coupling_error(cls, cfg: RFTSolverCfg, in_place: bool) -> str:
        return "uses shared-model RFT composition and cannot be a partitioned proxy/ADMM entry"

    @classmethod
    def solver_classes(cls, cfg: RFTSolverCfg) -> tuple[type, ...]:
        return (SolverMuJoCo, SolverRFT)

    @classmethod
    def build(
        cls, cfg: RFTSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverRFT:
        if mode != wp.DeterministicMode.NOT_GUARANTEED:
            raise ValueError("RFT surface reductions do not support deterministic mode")
        rigid = MJWarpSolver.build(cfg, model, mode, env_template)
        surfaces, inventory = collision_surfaces(model, cfg.body_pattern, cfg.segments)
        strength = (
            cfg.bulk_density
            * 9.81
            * (894 * cfg.internal_friction**3 - 386 * cfg.internal_friction**2 + 89 * cfg.internal_friction)
        )
        solver = SolverRFT(
            model,
            surfaces,
            rigid_solver=rigid,
            surface_height=cfg.surface_height,
            surface_bottom=cfg.surface_bottom,
            max_edge_length=cfg.max_edge_length,
            surface_shape_ids=[row["shape"] for row in inventory],
            config=SolverRFT.Config(
                density=cfg.bulk_density,
                internal_friction=cfg.internal_friction,
                surface_friction=cfg.surface_friction,
                strength=strength * cfg.strength_multiplier,
                velocity_regularization=cfg.velocity_regularization,
                bearing=cfg.bearing,
            ),
        )
        solver.surface_inventory = inventory
        return solver

    @classmethod
    def supports_capture(cls, cfg: RFTSolverCfg, solver: SolverRFT) -> bool:
        return MJWarpSolver.supports_capture(cfg, solver.rigid_solver)

    @classmethod
    def uses_collision_pipeline(cls, cfg: RFTSolverCfg, solver: SolverRFT) -> bool:
        return MJWarpSolver.uses_collision_pipeline(cfg, solver.rigid_solver)

    @classmethod
    def create_contacts(cls, cfg: RFTSolverCfg, solver: SolverRFT) -> Contacts:
        return MJWarpSolver.create_contacts(cfg, solver.rigid_solver)

    @classmethod
    def reset(cls, cfg: RFTSolverCfg, solver: SolverRFT, state: State, world_mask: wp.array) -> None:
        MJWarpSolver.reset(cfg, solver.rigid_solver, state, world_mask)

    @classmethod
    def check_step(cls, cfg: RFTSolverCfg, solver: SolverRFT, captured: bool, debug: bool) -> None:
        MJWarpSolver.check_step(cfg, solver.rigid_solver, captured, debug)

    @classmethod
    def fixed_tendon_control(cls, cfg: RFTSolverCfg, articulation, model: Model) -> None:
        raise NotImplementedError("RFT coupling currently supports articulations without fixed tendons")
