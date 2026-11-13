# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind all selected collision surfaces to an RFT–MJWarp shared-model solver."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import warp as wp
from newton import Axis, Contacts, GeoType, Mesh, Model, ShapeFlags, State
from newton.solvers import SolverMuJoCo, SolverRFT

from .mjwarp_solver import MJWarpSolver
from .rft_solver_cfg import RFTSolverCfg, RFTSurfaceResolutionCfg


def _octasphere(subdivisions: int) -> Mesh:
    """Construct a closed sphere with 8 * 4**subdivisions nondegenerate faces."""
    vertices = [np.array(p, dtype=float) for p in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(subdivisions):
        midpoints = {}
        refined = []
        for a, b, c in faces:
            indices = []
            for i, j in ((a, b), (b, c), (c, a)):
                key = tuple(sorted((i, j)))
                if key not in midpoints:
                    point = vertices[i] + vertices[j]
                    midpoints[key] = len(vertices)
                    vertices.append(point / np.linalg.norm(point))
                indices.append(midpoints[key])
            ab, bc, ca = indices
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        faces = refined
    return Mesh(np.array(vertices), np.array(faces).reshape(-1), compute_inertia=False)


def _cylinder_surface(radius: float, half_height: float, segments: int, axial_segments: int) -> Mesh:
    """Add axial quadrature panels while retaining the analytic cylinder dimensions."""
    theta = np.arange(segments) * (2 * np.pi / segments)
    vertices = [
        (radius * np.cos(t), radius * np.sin(t), z)
        for z in np.linspace(-half_height, half_height, axial_segments + 1)
        for t in theta
    ]
    faces = []
    for row in range(axial_segments):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = row * segments + i, row * segments + j
            c, d = a + segments, b + segments
            faces.extend(((a, b, c), (b, d, c)))
    bottom, top = len(vertices), len(vertices) + 1
    vertices.extend(((0, 0, -half_height), (0, 0, half_height)))
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(((bottom, j, i), (top, axial_segments * segments + i, axial_segments * segments + j)))
    return Mesh(np.array(vertices), np.array(faces).reshape(-1), compute_inertia=False)


def collision_surfaces(
    model: Model,
    pattern: str,
    segments: int,
    *,
    resolutions: Sequence[RFTSurfaceResolutionCfg | dict] = (),
) -> tuple[list[tuple[int, Mesh]], list[dict]]:
    """Return body-local surfaces and their source-shape inventory.

    Primitive dimensions and poses come from the finalized collision model.
    This does not modify collision geometry or inertial properties. Overlapping
    collider unions require separate validation; the manifest exposes them.
    Resolution rules accept config objects or their JSON dictionaries and match
    complete collider labels. Conflicting or unmatched rules raise an error.
    """
    rules = [RFTSurfaceResolutionCfg(**r) if isinstance(r, dict) else r for r in resolutions]
    for rule in rules:
        if rule.sphere_subdivisions is not None and (
            not isinstance(rule.sphere_subdivisions, int) or rule.sphere_subdivisions < 0
        ):
            raise ValueError("sphere_subdivisions must be a nonnegative integer")
        if rule.segments is not None and (not isinstance(rule.segments, int) or rule.segments < 3):
            raise ValueError("segments must be an integer of at least three")
        if not isinstance(rule.axial_segments, int) or rule.axial_segments < 1:
            raise ValueError("axial_segments must be a positive integer")
        if not isinstance(rule.primitive_quadrature_order, int) or rule.primitive_quadrature_order < 0:
            raise ValueError("primitive_quadrature_order must be a nonnegative integer")
    matched = [False] * len(rules)
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
        selection = [i for i, rule in enumerate(rules) if re.fullmatch(rule.shape_pattern, model.shape_label[shape])]
        if len(selection) > 1:
            raise ValueError(f"Multiple RFT resolution rules match {model.shape_label[shape]}")
        rule = rules[selection[0]] if selection else None
        if selection:
            matched[selection[0]] = True
        local_segments = segments if rule is None or rule.segments is None else rule.segments
        sphere_subdivisions = None if rule is None else rule.sphere_subdivisions
        axial_segments = 1 if rule is None else rule.axial_segments
        if sphere_subdivisions is not None and kind not in (GeoType.SPHERE, GeoType.ELLIPSOID):
            raise ValueError(f"Sphere subdivision requested for {kind.name}: {model.shape_label[shape]}")
        if axial_segments != 1 and kind != GeoType.CYLINDER:
            raise ValueError(f"Axial panels require a cylinder: {model.shape_label[shape]}")
        if rule is not None and rule.primitive_quadrature_order and kind not in (GeoType.SPHERE, GeoType.CYLINDER):
            raise ValueError(f"Analytic quadrature requires a sphere or cylinder: {model.shape_label[shape]}")
        key = (
            kind,
            tuple(dims),
            tuple(poses[shape]),
            id(model.shape_source[shape]),
            local_segments,
            sphere_subdivisions,
            axial_segments,
        )
        if key not in cache:
            if kind == GeoType.BOX:
                mesh = Mesh.create_box(*map(float, dims), compute_inertia=False)
            elif kind in (GeoType.SPHERE, GeoType.ELLIPSOID):
                mesh = (
                    _octasphere(sphere_subdivisions)
                    if sphere_subdivisions is not None
                    else Mesh.create_sphere(
                        1.0, num_latitudes=local_segments // 2, num_longitudes=local_segments, compute_inertia=False
                    )
                )
                scale = np.full(3, dims[0]) if kind == GeoType.SPHERE else dims
                mesh = Mesh(mesh.vertices * scale, mesh.indices, compute_inertia=False)
            elif kind == GeoType.CAPSULE:
                mesh = Mesh.create_capsule(
                    float(dims[0]), float(dims[1]), up_axis=Axis.Z, segments=local_segments, compute_inertia=False
                )
            elif kind == GeoType.CYLINDER:
                if axial_segments == 1:
                    mesh = Mesh.create_cylinder(
                        float(dims[0]), float(dims[1]), up_axis=Axis.Z, segments=local_segments, compute_inertia=False
                    )
                else:
                    mesh = _cylinder_surface(float(dims[0]), float(dims[1]), local_segments, axial_segments)
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
        if rule is not None:
            inventory[-1]["resolution"] = rule.to_dict()
    if not surfaces:
        raise ValueError(f"No collision surfaces matched RFT body_pattern={pattern!r}")
    unmatched = [rule.shape_pattern for rule, used in zip(rules, matched, strict=True) if not used]
    if unmatched:
        raise ValueError(f"RFT resolution rules matched no selected collider: {unmatched}")
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
        surfaces, inventory = collision_surfaces(
            model, cfg.body_pattern, cfg.segments, resolutions=cfg.surface_resolution
        )
        strength = (
            cfg.bulk_density
            * 9.81
            * (894 * cfg.internal_friction**3 - 386 * cfg.internal_friction**2 + 89 * cfg.internal_friction)
        )
        orders = [row.get("resolution", {}).get("primitive_quadrature_order", 0) for row in inventory]
        quadrature = {"surface_quadrature_orders": orders} if any(orders) else {}
        static_config = {}
        if cfg.static_contact_iterations:
            static_config = {
                "static_contact_iterations": cfg.static_contact_iterations,
                "static_contact_speed": cfg.static_contact_speed,
                "static_contact_strength": cfg.static_contact_strength,
                "static_contact_relaxation": cfg.static_contact_relaxation,
            }
        solver = SolverRFT(
            model,
            surfaces,
            rigid_solver=rigid,
            surface_height=cfg.surface_height,
            surface_bottom=cfg.surface_bottom,
            max_edge_length=cfg.max_edge_length,
            surface_shape_ids=[row["shape"] for row in inventory],
            **quadrature,
            config=SolverRFT.Config(
                density=cfg.bulk_density,
                internal_friction=cfg.internal_friction,
                surface_friction=cfg.surface_friction,
                strength=strength * cfg.strength_multiplier,
                velocity_regularization=cfg.velocity_regularization,
                bearing=cfg.bearing,
                **static_config,
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
        rigid = solver.rigid_solver
        # Native CPU MuJoCo owns only template world zero. Other selected
        # worlds must not rewind its history; RFT warm starts remain per-world.
        if rigid.use_mujoco_cpu and not world_mask.numpy()[0]:
            if cfg.static_contact_iterations:
                solver.reset_reactions(world_mask)
            return
        solver.reset(state, world_mask=world_mask, flags=0)
        MJWarpSolver._synchronize_reset_state(rigid, state, world_mask)

    @classmethod
    def check_step(cls, cfg: RFTSolverCfg, solver: SolverRFT, captured: bool, debug: bool) -> None:
        MJWarpSolver.check_step(cfg, solver.rigid_solver, captured, debug)

    @classmethod
    def fixed_tendon_control(cls, cfg: RFTSolverCfg, articulation, model: Model) -> None:
        raise NotImplementedError("RFT coupling currently supports articulations without fixed tendons")
