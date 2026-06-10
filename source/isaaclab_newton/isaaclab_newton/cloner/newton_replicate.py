# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdPhysics

from isaaclab_newton.physics import NewtonManager


def _prim_has_closed_kinematic_loops(prim: Usd.Prim) -> bool:
    """True if any body under *prim* is the child of more than one joint (cyclic graph)."""
    child_counts: dict[str, int] = {}
    for descendant in Usd.PrimRange(prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        try:
            joint = UsdPhysics.Joint(descendant)
            targets = joint.GetBody1Rel().GetTargets()
        except Exception:
            continue
        if not targets:
            continue
        child_path = targets[0].pathString
        if not child_path:
            continue
        child_counts[child_path] = child_counts.get(child_path, 0) + 1
        if child_counts[child_path] > 1:
            return True
    return False


def _get_collapse_fixed_joints() -> bool:
    """Check if collapse_fixed_joints is enabled via NewtonCfg or env var fallback."""
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    cfg = PhysicsManager._cfg
    if isinstance(cfg, NewtonCfg):
        return cfg.collapse_fixed_joints
    # Env var fallback for non-Newton configs (should not happen in practice)
    import os
    return os.environ.get("NEWTON_COLLAPSE_FIXED_JOINTS", "").lower() in ("1", "true", "yes")


def _get_disable_robot_self_collisions() -> bool:
    """Check if intra-robot self-collisions should be disabled via NewtonCfg."""
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    cfg = PhysicsManager._cfg
    if isinstance(cfg, NewtonCfg):
        return cfg.disable_robot_self_collisions
    return False


def _get_self_collision_shape_contraction() -> float:
    """Inward collision-hull contraction (meters) from NewtonCfg, 0.0 if unset."""
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    cfg = PhysicsManager._cfg
    if isinstance(cfg, NewtonCfg):
        return float(getattr(cfg, "self_collision_shape_contraction", 0.0))
    return 0.0


def _contract_collision_shapes_on_prototype(p, amount: float) -> int:
    """Shrink each collidable mesh hull inward toward its centroid by ``amount`` [m].

    Moves every vertex of each collidable mesh/convex-hull shape toward the hull's
    own centroid by ``amount`` meters (clamped so the hull cannot invert). This
    reduces the collision geometry so adjacent closed-loop links no longer overlap
    at the rest pose, WITHOUT modifying the source USD or recomputing body inertia
    (collision-only change). Returns the number of shapes contracted.
    """
    import numpy as np
    from newton import ShapeFlags

    n = 0
    for s in range(p.shape_count):
        if not (p.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES):
            continue
        src = p.shape_source[s]
        verts = getattr(src, "vertices", None)
        if verts is None:
            continue
        V = np.asarray(verts, dtype=np.float64)
        if V.ndim != 2 or V.shape[0] < 4:
            continue
        c = V.mean(axis=0)
        d = V - c
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        # shrink each vertex inward by `amount`, but never past the centroid
        shrink = np.clip(norm - amount, 0.0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(norm > 1e-12, d / norm, 0.0)
        Vnew = c + unit * shrink
        p.shape_source[s] = src.copy(vertices=Vnew, recompute_inertia=False)
        n += 1
    return n


def _get_jointed_self_collision_filter_hops() -> int:
    """Joint-graph hop distance within which self-collisions are filtered (0 = off)."""
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    cfg = PhysicsManager._cfg
    if isinstance(cfg, NewtonCfg):
        return int(getattr(cfg, "jointed_self_collision_filter_hops", 0))
    return 0


def _filter_jointed_self_collisions_on_prototype(p, hops: int) -> int:
    """Filter collisions between bodies within ``hops`` joints of each other.

    Treats the robot's joints (tree joints AND closed-loop / orphan joints, both of
    which populate ``joint_parent`` / ``joint_child``) as an undirected graph over
    bodies, then filters every collidable shape pair whose bodies are within
    ``hops`` joints (graph distance <= hops). ``hops=1`` filters only directly
    jointed neighbors; ``hops=2`` also filters grandparent-grandchild and sibling
    pairs that share an intermediate body, etc. Non-adjacent bodies (e.g. left leg
    vs right leg) still collide so the legs cannot pass through each other.

    DR Legs needs ``hops=2``: the rest-pose interpenetrations that inject the
    asymmetric yaw/pitch kick are between bodies TWO joints apart (e.g.
    pelvis -> hip_servos -> upperleg_link, and ankle_bracket_a -> ankle_bracket_b
    -> foot), which a direct-neighbor (hops=1) filter does not cover. Newton's
    articulation-keyed self-collision filter normally excludes directly-jointed
    pairs, but it does not fire for DR Legs (articulation_count=0) and would not
    cover the loop-closure joints anyway.

    Returns the number of filtered shape pairs.
    """
    from collections import deque

    from newton import ShapeFlags

    if hops <= 0:
        return 0

    # Build undirected adjacency over bodies from all joints (excluding world -1).
    adj: dict[int, set[int]] = {}
    for k in range(p.joint_count):
        a = p.joint_parent[k]
        b = p.joint_child[k]
        if a < 0 or b < 0 or a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # For each body, BFS out to `hops` to collect bodies within range.
    within: dict[int, set[int]] = {}
    for start in adj:
        seen = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            d = seen[cur]
            if d == hops:
                continue
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen[nb] = d + 1
                    q.append(nb)
        within[start] = {b for b in seen if b != start}

    # Collidable shapes grouped per body.
    shapes_of_body: dict[int, list[int]] = {}
    for s in range(p.shape_count):
        if not (p.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES):
            continue
        b = p.shape_body[s]
        shapes_of_body.setdefault(b, []).append(s)

    # Emit each unordered body pair once.
    n = 0
    for a, neighbors in within.items():
        for b in neighbors:
            if a >= b:
                continue
            for si in shapes_of_body.get(a, ()):
                for sj in shapes_of_body.get(b, ()):
                    p.add_shape_collision_filter_pair(si, sj)
                    n += 1
    return n


def _disable_self_collisions_on_prototype(p) -> None:
    """Filter every intra-robot shape pair in a prototype builder.

    Adds a collision-filter pair for all pairs of shapes belonging to different
    bodies within the prototype (and pairs sharing a body), so the robot's own
    bodies never collide with each other. Body-vs-ground and body-vs-other-robot
    collisions are unaffected (those involve shapes outside this prototype).

    Required for closed-loop / orphan-joint robots (DR Legs): convex-hull collider
    approximation gives every linkage body a collider, and the model has no
    articulation root, so Newton's articulation-keyed self-collision filter never
    fires. Mechanically-linked loop bodies would otherwise self-collide and inject
    spurious asymmetric impulses (observed: yaw spin-up -> topple -> reset).
    """
    from newton import ShapeFlags

    # Only consider shapes that actually collide.
    collidable = [
        s
        for s in range(p.shape_count)
        if p.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES
    ]
    for i in range(len(collidable)):
        for j in range(i + 1, len(collidable)):
            p.add_shape_collision_filter_pair(collidable[i], collidable[j])


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object, dict]:
    """Build a Newton model builder from clone mapping inputs.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder, stage metadata returned
        by ``add_usd``, and a site index map for
        :attr:`NewtonManager._cl_site_index_map`.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    collapse_fixed_joints = _get_collapse_fixed_joints()
    if collapse_fixed_joints:
        import logging
        logging.getLogger(__name__).info("collapse_fixed_joints=True — merging fixed-joint bodies")
    disable_robot_self_collisions = _get_disable_robot_self_collisions()
    if disable_robot_self_collisions:
        import logging
        logging.getLogger(__name__).info(
            "disable_robot_self_collisions=True — filtering all intra-robot shape pairs"
        )
    jointed_filter_hops = _get_jointed_self_collision_filter_hops()
    if jointed_filter_hops > 0:
        import logging
        logging.getLogger(__name__).info(
            f"jointed_self_collision_filter_hops={jointed_filter_hops} — filtering self-collisions "
            f"between bodies within {jointed_filter_hops} joint(s) (farther bodies still collide)"
        )
    self_collision_contraction = _get_self_collision_shape_contraction()
    if self_collision_contraction > 0.0:
        import logging
        logging.getLogger(__name__).info(
            f"self_collision_shape_contraction={self_collision_contraction} m — "
            f"shrinking robot collision hulls inward"
        )

    builder = NewtonManager.create_builder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs"] + sources,
        schema_resolvers=schema_resolvers,
        collapse_fixed_joints=collapse_fixed_joints,
    )

    # The prototype is built from env_0 in absolute world coordinates.
    # add_builder xforms are deltas from env_0 so positions don't get double-counted.
    env0_pos = positions[0]
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = NewtonManager.create_builder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
            collapse_fixed_joints=collapse_fixed_joints,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        if self_collision_contraction > 0.0:
            import logging
            _ncon = _contract_collision_shapes_on_prototype(p, self_collision_contraction)
            logging.getLogger(__name__).info(
                f"contracted {_ncon} collision hulls inward by {self_collision_contraction} m"
            )
        if jointed_filter_hops > 0:
            import logging
            _njf = _filter_jointed_self_collisions_on_prototype(p, jointed_filter_hops)
            logging.getLogger(__name__).info(
                f"filtered {_njf} self-collision shape pairs within {jointed_filter_hops} joint(s)"
            )
        if disable_robot_self_collisions:
            _disable_self_collisions_on_prototype(p)
        protos[src_path] = p

    # Inject registered sites into prototypes (and global sites into main builder)
    global_sites, proto_sites = NewtonManager._cl_inject_sites(builder, protos)

    # Global sites: (int, None)
    global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}

    # Local sites: per-world sublists, populated in the loop below
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    for col, _ in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        # add all active sources for this world
        delta_pos = (positions[col] - env0_pos).tolist()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            proto = protos[sources[row]]
            offset = builder.shape_count
            builder.add_builder(
                proto,
                xform=wp.transform(delta_pos, quaternions[col].tolist()),
            )
            # Compute final shape indices for sites in this proto
            for label, proto_shape_indices in proto_sites.get(id(proto), {}).items():
                if label not in local_site_map:
                    local_site_map[label] = [[] for _ in range(num_worlds)]
                for proto_shape_idx in proto_shape_indices:
                    local_site_map[label][col].append(offset + proto_shape_idx)
        # end the world context
        builder.end_world()

    site_index_map = {
        **global_site_map,
        **{label: (None, per_world) for label, per_world in local_site_map.items()},
    }

    return builder, stage_info, site_index_map


def _rename_builder_labels(
    builder: ModelBuilder, sources: list[str], destinations: list[str], env_ids: torch.Tensor, mapping: torch.Tensor
) -> None:
    """Rename builder labels/keys from source roots to destination roots.

    Args:
        builder: Newton model builder to update in-place.
        sources: Source prim root paths.
        destinations: Destination prim path templates.
        env_ids: Environment ids corresponding to mapping columns.
        mapping: Boolean source-to-environment mapping matrix.
    """
    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        src_prefix_len = len(src_path.rstrip("/"))
        swap = lambda name, new_root: new_root + name[src_prefix_len:]  # noqa: E731
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world IDs (sequential) to destination paths using env_ids
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}

        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key")
            worlds_arr = getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                world_id = int(w)
                if world_id in world_roots and labels[k].startswith(src_path):
                    labels[k] = swap(labels[k], world_roots[world_id])


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    builder, stage_info, site_index_map = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    NewtonManager._cl_site_index_map = site_index_map
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def newton_visualizer_prebuild(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate a clone plan into a finalized Newton model/state for visualization.

    Unlike :func:`newton_physics_replicate`, this path does not mutate ``NewtonManager`` and is intended
    for prebuilding visualizer-only artifacts that can be consumed by scene data providers.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of finalized Newton model and state.
    """
    builder, _, _site_index_map = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state
