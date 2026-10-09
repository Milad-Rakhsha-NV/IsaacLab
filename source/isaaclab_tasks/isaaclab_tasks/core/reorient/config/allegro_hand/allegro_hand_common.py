# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro Hand identity shared by the Direct and manager-based reorientation tasks.

Asset and marker configurations, joint/body name lists, backend physics
presets, and the sim mixin. No task tunables.
"""

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

ALLEGRO_HAND_ROBOT_CFG = replace(ALLEGRO_HAND_CFG, prim_path="{ENV_REGEX_NS}/Robot")

CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ],
        collision_props=[
            sim_utils.UsdPhysicsCollisionCfg(),
            sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
        ],
        scale=(1.2, 1.2, 1.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
)
"""In-hand cube for the Allegro reorientation task."""

GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.2, 1.2, 1.2),
        )
    },
)
"""Goal cube marker for the reorientation environments."""


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for the Allegro Hand reorientation environments."""

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
        ),
        num_substeps=2,
    )
    # Experimental DVI preset: preserve the task's assets, drives, and control rate.
    newton_dvi = NewtonCfg(
        collision_cfg=NewtonCollisionPipelineCfg(),
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_iterative_refinement_steps=1,
            contact_solver_type="sparse_jacobi",
            contact_max_iterations=20,
            # Dense hand/cube contacts need more conservative Jacobi relaxation.
            contact_omega=0.15,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            contact_friction_projection="tangential",
            coupling_iterations=2,
            post_stabilize_joints=True,
            cache_factorization=True,
            angular_damping=0.0,
        ),
        num_substeps=2,
        collapse_fixed_joints=True,
        use_cuda_graph=True,
    )
    # Experimental APGD alternative. RES4 measures the cone optimality condition;
    # normal-only complementarity can select a suboptimal APGD cone iterate.
    newton_dvi_apgd = NewtonCfg(
        collision_cfg=NewtonCollisionPipelineCfg(),
        solver_cfg=replace(
            newton_dvi.solver_cfg,
            post_stabilize_joints=True,
            contact_friction_projection="cone",
            contact_solver_type="sparse_apgd",
            contact_recovery_speed=5.0,
            contact_residual_mode="res4",
        ),
        num_substeps=2,
        collapse_fixed_joints=True,
        use_cuda_graph=True,
    )
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp
