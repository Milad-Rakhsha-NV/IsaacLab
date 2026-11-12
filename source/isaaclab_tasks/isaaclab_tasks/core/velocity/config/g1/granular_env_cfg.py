# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matched direct G1 controls for rigid ground and reset-randomized RFT layers."""

from __future__ import annotations

import numpy as np
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.rft_solver_cfg import RFTSolverCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.utils import configclass, replace

from isaaclab_assets import G1_MINIMAL_CFG

from ..go2.rft_env_cfg import Go2GaitCfg, Go2RFTCurriculumEnvCfg, Go2RFTSceneCfg, Go2RigidEnvCfg


@sim_utils.clone
def spawn_g1_granular(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Add low-cost thigh and calf colliders to the minimal asset before cloning.

    The original minimal asset has torso and foot colliders only. Capsules span
    each knee/ankle parent joint anchor, using 45 mm thigh and 35 mm calf radii.
    The same collision geometry is used by rigid, RFT and MPM solvers. Authored
    body masses, inertia, joints and visible meshes are preserved.

    Args:
        prim_path: Destination robot prim path or clone expression.
        cfg: USD asset configuration.
        translation: Robot translation relative to its parent.
        orientation: Robot orientation in xyzw order.
        **kwargs: Additional arguments forwarded to the USD spawner.

    Returns:
        The spawned robot root prim.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    joints = {
        item.GetName(): UsdPhysics.Joint(item)
        for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if item.IsA(UsdPhysics.RevoluteJoint)
    }
    for side in ("left", "right"):
        for suffix, radius in (("knee_joint", 0.045), ("ankle_pitch_joint", 0.035)):
            joint = joints[f"{side}_{suffix}"]
            body = joint.GetBody0Rel().GetTargets()[0]
            endpoint = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=float)
            length = float(np.linalg.norm(endpoint))
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(endpoint / length))).GetQuat()
            imaginary = rotation.GetImaginary()
            shape = sim_utils.CapsuleCfg(
                radius=radius,
                height=length - 2 * radius,
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            )
            capsule = shape.func(
                str(body.AppendChild("GranularLegCollider")),
                shape,
                translation=tuple(endpoint / 2),
                orientation=(*imaginary, rotation.GetReal()),
            )
            UsdGeom.Imageable(capsule).MakeInvisible()
    return prim


@configclass
class G1GranularSpawnCfg(sim_utils.UsdFileCfg):
    """Minimal G1 geometry with additional shared leg collision capsules."""

    func: str = "isaaclab_tasks.core.velocity.config.g1.granular_env_cfg:spawn_g1_granular"


@configclass
class G1GranularGaitCfg(Go2GaitCfg):
    """Biped support timing and posture weights shared by both G1 policies."""

    hip_deviation_weight = 0.1
    joint_deviation_weight = 0.05
    stand_deviation_weight = 0.25
    joint_acceleration_weight = 1.0e-7
    vertical_velocity_weight = 0.2
    angular_velocity_weight = 0.05
    foot_sliding_weight = 1.0
    air_time_weight = 0.75
    air_time_target = 0.4
    biped_air_time = True
    stance_half_width = 0.08


@configclass
class G1GranularRigidEnvCfg(Go2RigidEnvCfg):
    """G1 minimal asset, explicit direct policy contract and a rigid support plane."""

    action_space = 37
    observation_space = 123
    state_space = 123
    actor_observes_linear_velocity = True
    target_height = 0.74
    minimum_base_height = 0.4
    effort_weight = 2.0e-6
    forward_command_range = (0.1, 1.0)
    foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_side_signs = (1.0, -1.0)
    hip_joint_pattern = ".*_hip_(yaw|roll)_joint"
    undesired_contact_pattern = ".*_(hip_yaw|knee)_link|torso_link"
    knee_to_sole_length = 0.35
    gait: G1GranularGaitCfg = G1GranularGaitCfg()
    sim = replace(
        Go2RigidEnvCfg().sim,
        physics=NewtonCfg(
            num_substeps=4,
            solver_cfg=MJWarpSolverCfg(njmax=128, nconmax=64, use_mujoco_contacts=True, integrator="implicitfast"),
        ),
    )
    scene: Go2RFTSceneCfg = replace(
        Go2RigidEnvCfg().scene,
        robot=replace(
            G1_MINIMAL_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1GranularSpawnCfg(
                **{
                    name: value
                    for name, value in vars(G1_MINIMAL_CFG.spawn).items()
                    if name not in ("func", "articulation_props")
                },
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
            ),
        ),
        contact_forces=ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True, update_period=0.005
        ),
    )


@configclass
class G1GranularRFTEnvCfg(G1GranularRigidEnvCfg):
    """Identical G1 mechanics with a material curriculum from thin to 118 mm layers."""

    randomize_material = True
    curriculum_enabled = True
    bed_depth = 0.118
    initial_bed_depth_range = (0.0, 0.01)
    bed_depth_range = (0.0, 0.118)
    sim = replace(
        G1GranularRigidEnvCfg().sim,
        physics=NewtonCfg(
            num_substeps=4,
            solver_cfg=RFTSolverCfg(
                njmax=128,
                nconmax=64,
                use_mujoco_contacts=True,
                integrator="implicitfast",
                bearing=0.87574,
                velocity_regularization=0.04,
                segments=12,
                surface_height=0.118,
                surface_bottom=0.0,
            ),
        ),
    )
    scene: Go2RFTSceneCfg = replace(
        G1GranularRigidEnvCfg().scene,
        bed=Go2RFTCurriculumEnvCfg().scene.bed,
    )
