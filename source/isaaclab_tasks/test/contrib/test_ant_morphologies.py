# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate real Ant topology changes and their imported mechanics before policy training."""

import math

import pytest
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env import MultiRobotLocomotionEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env_cfg import (
    AntFiveSpecialistEnvCfg,
    AntThreeFourEnvCfg,
)

from isaaclab_assets.robots.ant_variants import build_ant_variant

pytestmark = pytest.mark.integration


def test_generated_ant_joint_frames_and_topology(tmp_path):
    """Added joints close in world space and removed joints leave no dangling body references."""
    for legs in (3, 5):
        stage = Usd.Stage.Open(build_ant_variant(legs, cache_dir=str(tmp_path)))
        bodies = {prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        joints = [UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
        assert len(bodies) == 1 + 2 * legs and len(joints) == 2 * legs
        transforms = UsdGeom.XformCache()
        children = set()
        for joint in joints:
            parent, child = joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]
            assert parent in bodies and child in bodies and child not in children
            children.add(child)
            world_frames = []
            for body, position, orientation in (
                (parent, joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                (child, joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            ):
                frame = Gf.Matrix4d().SetRotate(Gf.Quatd(orientation))
                frame.SetTranslateOnly(Gf.Vec3d(position))
                world_frames.append(frame * transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body)))
            for left, right in zip(world_frames[0], world_frames[1], strict=True):
                assert Gf.IsClose(left, right, 1e-5)
        # Equal hip radii and azimuth gaps are the physical radial-symmetry contract.
        hips = [j.GetLocalPos0Attr().Get() for j in joints if str(j.GetBody0Rel().GetTargets()[0]) == "/ant/torso"]
        angles = sorted(math.atan2(p[1], p[0]) % math.tau for p in hips)
        gaps = [(angles[(i + 1) % legs] - angles[i]) % math.tau for i in range(legs)]
        assert gaps == pytest.approx([math.tau / legs] * legs, abs=1e-6)
        radii = [math.hypot(p[0], p[1]) for p in hips]
        assert radii == pytest.approx([radii[0]] * legs, abs=1e-6)
        assert len(children) == len(bodies) - 1
        assert str(next(iter(bodies - children))) == "/ant/torso"
        for prim in stage.Traverse():
            for relationship in prim.GetRelationships():
                for target in relationship.GetTargets():
                    assert stage.GetObjectAtPath(target), (prim.GetPath(), target)


def test_ant_variants_import_mass_and_reset_dynamics():
    """All morphologies have finite positive inertia and valid independent short trajectories."""
    cfg = AntThreeFourEnvCfg()
    cfg.robots["ant_5"] = AntFiveSpecialistEnvCfg().robots["ant_5"]
    cfg.robots["ant_5"].offset = (0.0, 6.0, 0.0)
    cfg.scene.num_envs = 20
    cfg.scene.env_spacing = 12.0
    cfg.seed = 42
    cfg.sim.device = test_devices(DeviceScope.DEFAULT_CUDA)[0]
    cfg.sim.physics.debug_mode = True
    for spec in cfg.robots.values():
        spec.task.episode_length_s = 0.5
    with launch_simulation(cfg):
        env = MultiRobotLocomotionEnv(cfg)
        try:
            observations, _ = env.reset()
            masses = []
            for agent, legs in (("ant_3", 3), ("ant_4", 4), ("ant_5", 5)):
                robot = env.robots[agent]
                assert robot.num_joints == 2 * legs and robot.num_bodies == 1 + 2 * legs
                assert observations[agent].shape == (cfg.scene.num_envs, 12 + 12 * legs)
                mass = robot.data.body_mass.torch
                inertia = robot.data.body_inertia.torch.reshape(cfg.scene.num_envs, robot.num_bodies, 3, 3)
                assert torch.isfinite(mass).all() and (mass > 0).all()
                assert torch.isfinite(inertia).all() and (torch.linalg.eigvalsh(inertia) > 0).all()
                masses.append(float(mass[0].sum()))
                limits = robot.data.soft_joint_pos_limits.torch
                assert (robot.data.joint_pos.torch >= limits[..., 0]).all()
                assert (robot.data.joint_pos.torch <= limits[..., 1]).all()
            assert masses[0] < masses[1] < masses[2]
            print("Imported total masses for 3/4/5 legs:", masses)
            completed = {agent: 0 for agent in env.possible_agents}
            for step in range(120):
                actions = {
                    agent: torch.zeros(cfg.scene.num_envs, robot.num_joints, device=env.device)
                    if step < 60
                    else torch.rand(cfg.scene.num_envs, robot.num_joints, device=env.device) * 2 - 1
                    for agent, robot in env.robots.items()
                }
                observations, rewards, terminated, truncated, _ = env.step(actions)
                for agent, robot in env.robots.items():
                    assert torch.isfinite(observations[agent]).all() and torch.isfinite(rewards[agent]).all()
                    assert torch.isfinite(robot.data.body_vel_w.torch).all()
                    if step == 0:
                        assert robot.data.root_lin_vel_w.torch.norm(dim=-1).max() < 1.0
                    completed[agent] += int((terminated[agent] | truncated[agent]).sum())
            assert all(count >= 12 for count in completed.values())
            env.reset()  # explicit reset after stepped trajectories must remain valid
            # Paired airborne trials isolate every action coordinate from contact and resets.
            # Equal initial states with opposite effort must produce opposite own-joint responses.
            # This catches missing/misrouted action columns and unactuated added joints.
            pulse = {}
            for agent, robot in env.robots.items():
                pose = robot.data.root_pose_w.torch.clone()
                pose[:, 2] = 3.0
                robot.write_root_pose_to_sim_index(root_pose=pose)
                robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_vel_w.torch))
                midpoint = robot.data.soft_joint_pos_limits.torch.mean(dim=-1)
                robot.write_joint_state_to_sim_index(position=midpoint, velocity=torch.zeros_like(midpoint))
                pulse[agent] = torch.zeros_like(midpoint)
                for j in range(robot.num_joints):
                    pulse[agent][2 * j, j] = 0.3
                    pulse[agent][2 * j + 1, j] = -0.3
            for _ in range(8):
                env.step(pulse)
            for agent, robot in env.robots.items():
                velocity = robot.data.joint_vel.torch
                response = torch.stack([velocity[2 * j, j] - velocity[2 * j + 1, j] for j in range(robot.num_joints)])
                assert (response > 0.1).all(), (agent, robot.joint_names, response.tolist())
                print("Per-joint signed pulse response:", agent, response.tolist())
        finally:
            env.close()
