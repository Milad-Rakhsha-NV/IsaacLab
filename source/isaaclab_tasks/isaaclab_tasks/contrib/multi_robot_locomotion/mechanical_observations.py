# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned Newton body/joint/DOF/shape descriptors for the Ant mechanics pilot.

This is an explicit rigid-articulation subset, not serialization of every Newton feature.
Labels/indices select and connect entities but are never numeric policy features. Descriptor
values use SI units and Newton's local frames (transform quaternions are xyzw). Inertia is the
full COM tensor in the body frame. Dynamic body quaternions use Isaac Lab's wxyz convention.
"""

import newton
import numpy as np
import torch
from isaaclab_newton.physics import NewtonManager
from tensordict import TensorDict

SCHEMA_VERSION = "newton-ant-mechanics-v1"
# Each typed token occupies named slots; zero means not applicable outside its type.
STATIC_FIELDS = {
    "kind": 4,  # body, joint, DOF, shape
    "body_mass": 1,
    "body_com": 3,
    "body_inertia": 9,
    "body_flags": 2,  # dynamic, kinematic
    "root_body": 1,
    "joint_type": 8,
    "joint_X_p": 7,
    "joint_X_c": 7,
    "joint_enabled": 1,
    "joint_dof_dim": 2,
    "joint_axis": 3,
    "angular_dof": 1,
    "joint_armature": 1,
    "joint_target_mode": 4,
    "joint_target_ke": 1,
    "joint_target_kd": 1,
    "joint_damping": 1,
    "joint_effort_limit": 1,
    "joint_velocity_limit": 1,
    "joint_friction": 1,
    "joint_limit_lower": 1,
    "joint_limit_upper": 1,
    "joint_limit_ke": 1,
    "joint_limit_kd": 1,
    "actuated": 1,
    "action_effort_gain": 1,
    "shape_type": 12,
    "shape_transform": 7,
    "shape_scale": 3,
    "shape_flags": 3,  # shape contact, particle contact, hydroelastic
    "shape_is_solid": 1,
    "shape_margin": 1,
    "shape_gap": 1,
    "shape_material_ke": 1,
    "shape_material_kd": 1,
    "shape_material_kf": 1,
    "shape_material_ka": 1,
    "shape_material_mu": 1,
    "shape_material_restitution": 1,
    "shape_material_mu_torsional": 1,
    "shape_material_mu_rolling": 1,
    "shape_material_kh": 1,
    "world_shape": 1,
}
STATIC_SLICES = {}
STATIC_WIDTH = 0
for _name, _width in STATIC_FIELDS.items():
    STATIC_SLICES[_name] = slice(STATIC_WIDTH, STATIC_WIDTH + _width)
    STATIC_WIDTH += _width

DYNAMIC_WIDTH = 23  # body state 13, q/dq/previous effort command 3, incoming wrench 6, wrench present 1
RELATION_COUNT = 10  # none, joint-parent/reverse, joint-child/reverse, DOF-joint/reverse, shape-body/reverse, contact


class NewtonMechanicalGraph:
    """Export a fixed prototype from the actual finalized Newton model.

    Includes unactuated free-base DOFs and ground contact properties. Supports free/fixed/
    revolute/prismatic joints and analytic shapes; rejects meshes, deformables and mimic
    constraints. Per-world physical randomization is deliberately unsupported by this cache.
    """

    def __init__(self, model, body_ids, joint_ids, action_joint_ids, action_gains, device):
        _validate_model_selection(model, body_ids, joint_ids, action_joint_ids)
        self.device = device
        self.num_bodies, self.num_joints = len(body_ids), len(action_joint_ids)
        cache = {}

        def values(name):
            if name not in cache:
                cache[name] = getattr(model, name).numpy()
            return cache[name]

        children = values("joint_child")[joint_ids].tolist()
        if len(set(children)) != len(children) or set(children) != set(body_ids):
            raise ValueError("Mechanical descriptor v1 requires a complete tree without loop-closing joints.")
        if np.count_nonzero(values("joint_parent")[joint_ids] == -1) != 1:
            raise ValueError("Mechanical descriptor v1 requires exactly one floating root.")
        body_nodes = {body: index for index, body in enumerate(body_ids)}
        joint_nodes = {joint: len(body_nodes) + index for index, joint in enumerate(joint_ids)}
        dofs = [d for j in joint_ids for d in range(values("joint_qd_start")[j], values("joint_qd_start")[j + 1])]
        dof_nodes = {dof: len(body_nodes) + len(joint_nodes) + index for index, dof in enumerate(dofs)}
        # The pilot uses an infinite shared plane. Arbitrary external objects need scene-relative descriptors.
        ground = np.flatnonzero(values("shape_body") == -1).tolist()
        if any(values("shape_type")[shape] != newton.GeoType.PLANE for shape in ground):
            raise ValueError("Mechanical descriptor v1 supports only plane-shaped external terrain.")
        shapes = ground + [s for body in body_ids for s in model.body_shapes[body]]
        shape_nodes = {s: len(body_nodes) + len(joint_nodes) + len(dof_nodes) + i for i, s in enumerate(shapes)}
        self.num_nodes = len(body_nodes) + len(joint_nodes) + len(dof_nodes) + len(shape_nodes)
        static = np.zeros((self.num_nodes, STATIC_WIDTH), dtype=np.float32)
        edges = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)

        def put(node, name, value):
            static[node, STATIC_SLICES[name]] = np.asarray(value).flatten()

        def connect(left, right, kind):
            edges[left, right], edges[right, left] = kind, kind + 1

        for body, node in body_nodes.items():
            put(node, "kind", [1, 0, 0, 0])
            for name in ("body_mass", "body_com", "body_inertia"):
                put(node, name, values(name)[body])
            put(node, "body_flags", [bool(values("body_flags")[body] & bit) for bit in (1, 2)])
        allowed_joints = (
            newton.JointType.FREE,
            newton.JointType.FIXED,
            newton.JointType.REVOLUTE,
            newton.JointType.PRISMATIC,
        )
        for joint, node in joint_nodes.items():
            kind = int(values("joint_type")[joint])
            if kind not in allowed_joints:
                raise ValueError(f"Joint type {kind} is outside mechanical descriptor v1.")
            parent, child = int(values("joint_parent")[joint]), int(values("joint_child")[joint])
            if child not in body_nodes or (parent != -1 and parent not in body_nodes):
                raise ValueError("Selected articulation omits a joint's body.")
            if parent == -1 and kind != newton.JointType.FREE:
                raise ValueError("This locomotion descriptor requires a floating base.")
            put(node, "kind", [0, 1, 0, 0])
            put(node, "joint_type", np.eye(8)[kind])
            for name in ("joint_X_p", "joint_X_c", "joint_enabled", "joint_dof_dim"):
                put(node, name, values(name)[joint])
            if parent == -1:
                # Free-base placement is episode state, never embodiment identity.
                put(node, "joint_X_p", [0, 0, 0, 0, 0, 0, 1])
                put(body_nodes[child], "root_body", 1)
            else:
                connect(node, body_nodes[parent], 1)
            connect(node, body_nodes[child], 3)
            start, end = values("joint_qd_start")[joint : joint + 2]
            for local, dof in enumerate(range(start, end)):
                dnode = dof_nodes[dof]
                connect(dnode, node, 5)
                put(dnode, "kind", [0, 0, 1, 0])
                put(dnode, "angular_dof", local >= values("joint_dof_dim")[joint, 0])
                mode = int(values("joint_target_mode")[dof])
                if mode not in range(4):
                    raise ValueError(f"Unsupported Newton drive mode: {mode}")
                put(dnode, "joint_target_mode", np.eye(4)[mode])
                for name in (
                    "joint_axis",
                    "joint_armature",
                    "joint_target_ke",
                    "joint_target_kd",
                    "joint_damping",
                    "joint_effort_limit",
                    "joint_velocity_limit",
                    "joint_friction",
                    "joint_limit_lower",
                    "joint_limit_upper",
                    "joint_limit_ke",
                    "joint_limit_kd",
                ):
                    put(dnode, name, values(name)[dof])
        self.action_nodes = []
        for joint, gain in zip(action_joint_ids, action_gains, strict=True):
            start, end = values("joint_qd_start")[joint : joint + 2]
            if end - start != 1:
                raise ValueError("Actions must map to single-DOF joints in descriptor v1.")
            node = dof_nodes[int(start)]
            self.action_nodes.append(node)
            put(node, "actuated", 1)
            put(node, "action_effort_gain", gain)
        allowed_shapes = (
            newton.GeoType.PLANE,
            newton.GeoType.SPHERE,
            newton.GeoType.CAPSULE,
            newton.GeoType.BOX,
            newton.GeoType.ELLIPSOID,
            newton.GeoType.CYLINDER,
            newton.GeoType.CONE,
        )
        for shape, node in shape_nodes.items():
            kind = int(values("shape_type")[shape])
            if kind not in allowed_shapes:
                raise ValueError(f"Shape type {kind} requires a geometry encoder outside descriptor v1.")
            put(node, "kind", [0, 0, 0, 1])
            put(node, "shape_type", np.eye(12)[kind])
            put(node, "shape_flags", [bool(values("shape_flags")[shape] & bit) for bit in (2, 4, 16)])
            for name in STATIC_FIELDS:
                if name.startswith("shape_") and name not in ("shape_type", "shape_flags"):
                    put(node, name, values(name)[shape])
            body = int(values("shape_body")[shape])
            if body != -1:
                connect(node, body_nodes[body], 7)
            else:
                put(node, "world_shape", 1)
        # Include permitted contact relations, not arbitrary collision-group identifiers.
        for left, right in values("shape_contact_pairs"):
            if left in shape_nodes and right in shape_nodes:
                i, j = shape_nodes[left], shape_nodes[right]
                edges[i, j] = edges[j, i] = 9
        if not np.isfinite(static).all():
            raise ValueError(
                "Newton mechanical descriptors must be finite (including finite unbounded-limit sentinels)."
            )
        self.static = torch.tensor(static, device=device)
        self.edges = torch.tensor(edges, device=device)
        self.action_nodes = torch.tensor(self.action_nodes, device=device, dtype=torch.long)
        self.total_mass = float(values("body_mass")[body_ids].sum())


class MechanicalAntGraph(NewtonMechanicalGraph):
    """Map the richer Ant observation into mechanics tokens, preserving actual action order."""

    def __init__(self, env, agent):
        robot = env.robots[agent]
        model = NewtonManager.get_model()
        prefix = f"/World/envs/env_0/{agent}/"
        bodies = {label: index for index, label in enumerate(model.body_label)}
        joints = {label: index for index, label in enumerate(model.joint_label)}
        body_ids = [bodies[label] for label in robot.root_view.link_labels]
        joint_ids = [index for label, index in joints.items() if label.startswith(prefix)]
        action_ids = [joints[label] for label in robot.root_view.joint_labels]
        gains = (env.joint_gears[agent] * env.cfg.robots[agent].task.action_scale).tolist()
        super().__init__(model, body_ids, joint_ids, action_ids, gains, env.device)
        self.num_feet = len(env.feet_indices[agent])
        self.base_width = 12 + 3 * self.num_joints + 6 * self.num_feet
        self.flat_width = self.base_width + 3 + 13 * self.num_bodies
        feet = [env.wrenches[agent].body_names[index] for index in env.feet_indices[agent]]
        self.foot_nodes = [robot.body_names.index(name) for name in feet]
        self.context = torch.tensor(
            [
                self.num_bodies,
                len(joint_ids),
                self.num_joints,
                self.total_mass,
                env.step_dt,
                env.cfg.sim.dt,
                *model.gravity.numpy()[0],
            ],
            device=env.device,
        )

    def encode(self, flat, node_width, action_width):
        """Encode either live or captured terminal state; never read simulator state here."""
        count, joints = flat.shape[0], self.num_joints
        if flat.shape[1] != self.flat_width or node_width < self.num_nodes or action_width < joints:
            raise ValueError("Mechanical graph observation dimensions do not match the selected prototype.")
        dynamic = torch.zeros(count, node_width, DYNAMIC_WIDTH, device=flat.device)
        dynamic[:, : self.num_bodies, :13] = flat[:, self.base_width + 3 :].reshape(count, self.num_bodies, 13)
        dynamic[:, self.action_nodes, 13] = flat[:, 12 : 12 + joints]
        dynamic[:, self.action_nodes, 14] = flat[:, 12 + joints : 12 + 2 * joints]
        dynamic[:, self.action_nodes, 15] = flat[:, self.base_width - joints : self.base_width]
        dynamic[:, self.foot_nodes, 16:22] = flat[:, 12 + 2 * joints : self.base_width - joints].reshape(count, -1, 6)
        dynamic[:, self.foot_nodes, 22] = 1
        static = torch.nn.functional.pad(self.static, (0, 0, 0, node_width - self.num_nodes))
        edges = torch.nn.functional.pad(self.edges, (0, node_width - self.num_nodes, 0, node_width - self.num_nodes))
        actions = torch.nn.functional.pad(self.action_nodes, (0, action_width - joints))
        return TensorDict(
            {
                "global": torch.cat(
                    (flat[:, :12], flat[:, self.base_width : self.base_width + 3], self.context.expand(count, -1)),
                    dim=-1,
                ),
                "static": static.expand(count, -1, -1),
                "dynamic": dynamic,
                "edges": edges.expand(count, -1, -1),
                "node_mask": (torch.arange(node_width, device=flat.device) < self.num_nodes).expand(count, -1),
                "action_nodes": actions.expand(count, -1),
                "action_mask": (torch.arange(action_width, device=flat.device) < joints).expand(count, -1),
            },
            batch_size=[count],
        )


def _validate_model_selection(model, body_ids, joint_ids, action_joint_ids):
    """Reject models whose mechanics or selected entities cannot be represented by this schema."""
    if any(getattr(model, name, 0) for name in ("constraint_mimic_count", "particle_count", "muscle_count")):
        raise ValueError("Mechanical descriptor v1 does not encode mimic constraints, deformables or muscles.")
    coupled = {
        name: count
        for name, count in model.custom_frequency_counts.items()
        if count and any(term in name for term in ("tendon", "equality_constraint", "actuator"))
    }
    if coupled:
        raise ValueError(f"Mechanical descriptor v1 needs an encoder for custom mechanics: {coupled}")
    if len(set(body_ids)) != len(body_ids) or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("Mechanical entities must be unique.")
    if len(set(action_joint_ids)) != len(action_joint_ids) or not set(action_joint_ids) <= set(joint_ids):
        raise ValueError("Every action must select a distinct joint in the articulation.")
