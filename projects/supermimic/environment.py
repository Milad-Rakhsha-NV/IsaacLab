# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit step/reset adapter over IsaacLab's Newton G1 scene."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from pxr import Usd, UsdPhysics

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import GroundPlaneCfg, SimulationCfg, SimulationContext, UsdFileCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

from .math_utils import JOINT_NAMES, QD, Q
from .task import TRACKED_BODIES, Kinematics, TrackingTask


def robot_cfg():
    # PD parameters follow the MIT-licensed BeyondMimic robot configuration.
    default = (
        [-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0] * 3 + [0.2, 0.2, 0, 0.6, 0, 0, 0] + [0.2, -0.2, 0, 0.6, 0, 0, 0]
    )
    armature = [0.010177520, 0.025101925, 0.010177520, 0.025101925, 2 * 0.003609725, 2 * 0.003609725] * 2
    armature += [0.010177520, 2 * 0.003609725, 2 * 0.003609725] + ([0.003609725] * 5 + [0.00425] * 2) * 2
    effort = [88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50] + ([25] * 5 + [5] * 2) * 2
    omega = 20 * math.pi
    stiffness = {name: arm * omega**2 for name, arm in zip(JOINT_NAMES, armature, strict=True)}
    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Robots/Unitree/G1/g1_29dof.usd",
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.8), joint_pos=dict(zip(JOINT_NAMES, default, strict=True)), joint_vel={".*": 0.0}
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=JOINT_NAMES,
                stiffness=stiffness,
                damping={name: 4 * arm * omega for name, arm in zip(JOINT_NAMES, armature, strict=True)},
                armature=dict(zip(JOINT_NAMES, armature, strict=True)),
                joint_effort_limit=dict(zip(JOINT_NAMES, effort, strict=True)),
            )
        },
    )
    scale = [0.25 * limit / stiffness[name] for name, limit in zip(JOINT_NAMES, effort, strict=True)]
    return cfg, default, scale


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        ),
    )
    robot = robot_cfg()[0]


def simulation_cfg(device):
    return SimulationCfg(
        device=device,
        dt=0.005,
        render_interval=4,
        use_newton_actuators=False,
        physics=NewtonCfg(
            num_substeps=1,
            solver_cfg=MJWarpSolverCfg(
                iterations=10,
                ls_iterations=50,
                njmax=500,
                nconmax=150,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000, kd=1100),
        ),
    )


class TrackingEnv:
    """Pre-reset transitions are returned explicitly; auto-reset cannot corrupt replay."""

    def __init__(self, sim_cfg, library, num_envs):
        self.library, self.device, self.num_envs = library, library.device, num_envs
        self.sim = SimulationContext(sim_cfg)
        # This shipped USD includes 14 finger DOFs. Weld them at zero BEFORE model
        # finalization, making the 29-DOF learned state physically complete.
        self.asset_directory = TemporaryDirectory(prefix="supermimic-g1-")
        scene_cfg = G1SceneCfg(num_envs=num_envs, env_spacing=4.0)
        source = Usd.Stage.Open(retrieve_file_path(scene_cfg.robot.spawn.usd_path))
        source.SetEditTarget(source.GetSessionLayer())
        finger_paths = [
            prim.GetPath()
            for prim in source.Traverse()
            if prim.IsA(UsdPhysics.RevoluteJoint) and "_hand_" in prim.GetName()
        ]
        if len(finger_paths) != 14:
            raise ValueError(f"Expected 14 finger joints in the source G1 asset, found {len(finger_paths)}")
        for path in finger_paths:
            UsdPhysics.FixedJoint.Define(source, path)
        asset_path = str(Path(self.asset_directory.name) / "g1_fixed_hands.usda")
        source.Flatten().Export(asset_path)
        scene_cfg.robot.spawn.usd_path = asset_path
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.scene.update(0.0)
        self.robot = self.scene["robot"]
        if set(self.robot.joint_names) != set(JOINT_NAMES):
            raise ValueError(f"G1 must expose exactly the 29 modeled DOFs: {self.robot.joint_names}")
        self.order = torch.tensor([self.robot.joint_names.index(n) for n in JOINT_NAMES], device=self.device)
        self.inverse_order = torch.argsort(self.order)
        self.body_ids = [self.robot.body_names.index(n) for n in TRACKED_BODIES]
        _, default, scale = robot_cfg()
        self.default = torch.tensor(default, device=self.device)
        self.scale = torch.tensor(scale, device=self.device)
        fk = Kinematics.from_model(self.sim.backend_manager.backend.model, self.device)
        limits = self.robot.data.soft_joint_pos_limits.torch[0, self.order].clone()
        self.task = TrackingTask(fk, self.default, limits)
        self.clip = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.frame = torch.zeros_like(self.clip)
        self.episode = torch.zeros_like(self.clip)
        self.steps = torch.zeros_like(self.clip)
        self.previous = torch.zeros(num_envs, 29, device=self.device)
        self.history = torch.zeros(num_envs, 10, 93, device=self.device)
        self.reset(torch.arange(num_envs, device=self.device))

    def read(self):
        pose = self.robot.data.root_link_pose_w.torch.clone()
        pose[:, :3] -= self.scene.env_origins
        return torch.cat(
            (
                pose,
                self.robot.data.root_link_vel_w.torch,
                self.robot.data.joint_pos.torch[:, self.order],
                self.robot.data.joint_vel.torch[:, self.order],
            ),
            -1,
        )

    @torch.no_grad()
    def reset(self, ids, clips=None, frames=None, noise=True):
        n = len(ids)
        if n == 0:
            return
        clips = torch.randint(len(self.library.names), (n,), device=self.device) if clips is None else clips
        frames = (
            (torch.rand(n, device=self.device) * (self.library.lengths[clips] - 33)).long()
            if frames is None
            else frames
        )
        state = self.library.reference(clips, frames).clone()
        if noise:
            state[:, Q] += (2 * torch.rand(n, 29, device=self.device) - 1) * 0.02
        pose = state[:, :7].clone()
        pose[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=state[:, 7:13], env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=state[:, Q][:, self.inverse_order], env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=state[:, QD][:, self.inverse_order], env_ids=ids)
        self.scene.reset(ids)
        self.robot.actuators.target_command.set_position_index(value=state[:, Q][:, self.inverse_order], env_ids=ids)
        self.clip[ids], self.frame[ids], self.steps[ids] = clips, frames, 0
        self.episode[ids] += 1
        self.previous[ids] = 0
        self.sim.forward()
        self.scene.update(0.0)
        clean = self.task.observation(self.read()[ids], self.previous[ids])
        self.history[ids] = clean[:, None].expand(-1, 10, -1)

    @torch.no_grad()
    def step(self, action, noise=True):
        state, previous = self.read(), self.previous.clone()
        clip, frame, episode = self.clip.clone(), self.frame.clone(), self.episode.clone()
        target = (self.default + action * self.scale)[:, self.inverse_order]
        self.robot.actuators.target_command.set_position_index(value=target)
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(0.005)
        successor = self.read()
        if not torch.isfinite(successor).all():
            raise FloatingPointError("Nonfinite simulator state; full training is unsafe")
        reference = self.library.reference(clip, frame + 1)
        reward = self.task.reward(successor, reference, action, previous)
        failure = self.task.termination(successor, reference)
        self.frame += 1
        self.steps += 1
        timeout = (self.frame >= self.library.lengths[clip] - 1) | (self.steps >= 500)
        self.previous.copy_(action)
        self.history = torch.cat((self.history[:, 1:], self.task.observation(successor, action, noise)[:, None]), 1)
        return {
            "state": state,
            "successor": successor,
            "action": action,
            "previous": previous,
            "episode": episode,
            "clip": clip,
            "frame": frame,
            "reward": reward,
            "failure": failure,
            "timeout": timeout,
        }

    def close(self):
        self.sim.stop()
        SimulationContext.clear_instance()
        self.asset_directory.cleanup()
