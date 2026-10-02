# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint graphs for the contributed Ant locomotion experiment.

Nodes follow the articulation's action order. USD connectivity and physical descriptors are data,
not learned joint identities. The existing 12 global locomotion features are retained for this
first comparison; this is not yet a general robot-frame/history/contact representation.
"""

import torch
from tensordict import TensorDict

from pxr import Gf, Usd, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage


class AntJointGraph:
    """Encode the current Ant flat-observation contract as a variable-size physical graph."""

    def __init__(self, env, agent: str):
        robot = env.robots[agent]
        self.num_joints = robot.num_joints
        self.num_feet = len(env.feet_indices[agent])
        self.device = env.device
        if self.num_joints != 2 * self.num_feet:
            raise ValueError("The initial graph observation adapter supports two-joint Ant legs only.")
        stage = get_current_stage()
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{agent}")
        joints = {
            prim.GetName(): UsdPhysics.RevoluteJoint(prim)
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.IsA(UsdPhysics.RevoluteJoint)
        }
        if set(joints) != set(robot.joint_names):
            raise ValueError("USD revolute joints must match the articulation's action coordinates exactly.")
        child_to_joint = {str(j.GetBody1Rel().GetTargets()[0]): name for name, j in joints.items()}
        static, parents, feet = [], [], []
        foot_names = [env.wrenches[agent].body_names[i] for i in env.feet_indices[agent]]
        actuator = env.cfg.robots[agent].task.scene.robot.actuators["body"]
        for index, name in enumerate(robot.joint_names):
            joint = joints[name]
            parent, child = joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]
            parent_name = child_to_joint.get(str(parent))
            parents.append(robot.joint_names.index(parent_name) if parent_name is not None else -1)
            child_name = child.name
            body_index = robot.body_names.index(child_name)
            feet.append(foot_names.index(child_name) if child_name in foot_names else -1)
            axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[joint.GetAxisAttr().Get()]
            axis = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(axis)
            inertia = robot.data.body_inertia.torch[0, body_index].reshape(3, 3).diagonal().tolist()
            static.append(
                [
                    *joint.GetLocalPos0Attr().Get(),
                    *joint.GetLocalPos1Attr().Get(),
                    *axis,
                    *robot.data.soft_joint_pos_limits.torch[0, index].tolist(),
                    float(robot.data.body_mass.torch[0, body_index]),
                    *inertia,
                    float(env.joint_gears[agent][index]) * env.cfg.robots[agent].task.action_scale,
                    actuator.stiffness,
                    actuator.damping,
                    actuator.armature,
                ]
            )
        self.static = torch.tensor(static, device=self.device)
        self.parents = torch.tensor(parents, device=self.device)
        self.foot_indices = torch.tensor(feet, device=self.device)
        self.adjacency = torch.eye(self.num_joints, device=self.device)
        for child, parent in enumerate(parents):
            if parent >= 0:
                self.adjacency[child, parent] = self.adjacency[parent, child] = 1

    def encode(self, flat: torch.Tensor, pad_to: int | None = None) -> TensorDict:
        """Preserve action order and mask padding; accept terminal observations as well as live ones."""
        count, joints = flat.shape[0], self.num_joints
        width = joints if pad_to is None else pad_to
        if width < joints or flat.shape[1] != 12 + 3 * joints + 6 * self.num_feet:
            raise ValueError("Observation size or graph padding is incompatible with this Ant.")
        q = flat[:, 12 : 12 + joints]
        velocity = flat[:, 12 + joints : 12 + 2 * joints]
        previous_action = flat[:, -joints:]
        wrench = flat[:, 12 + 2 * joints : -joints].reshape(count, self.num_feet, 6)
        foot_mask = self.foot_indices >= 0
        node_wrench = wrench[:, self.foot_indices.clamp_min(0)] * foot_mask[None, :, None]
        features = torch.cat(
            (
                self.static.expand(count, -1, -1),
                q.unsqueeze(-1),
                velocity.unsqueeze(-1),
                previous_action.unsqueeze(-1),
                node_wrench,
                foot_mask.expand(count, -1).unsqueeze(-1),
            ),
            dim=-1,
        )
        nodes = torch.nn.functional.pad(features, (0, 0, 0, width - joints))
        adjacency = torch.nn.functional.pad(self.adjacency, (0, width - joints, 0, width - joints))
        mask = (torch.arange(width, device=self.device) < joints).expand(count, -1)
        return TensorDict(
            {"global": flat[:, :12], "nodes": nodes, "adjacency": adjacency.expand(count, -1, -1), "mask": mask},
            batch_size=[count],
        )
