# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import AnymalCRoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=120,
            nconmax=15,
            cone="elliptic",
            impratio=100,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_dvi = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_jacobi",
            contact_max_iterations=40,
            contact_omega=0.3,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            angular_damping=0.01,
            joint_limit_solver_type="sparse_jacobi",
        ),
        num_substeps=1,  # Match MuJoCo substeps
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    newton_dvi_apgd = None
    newton_dvi_pspg = None
    physx = default

    def __post_init__(self):
        # Keep APGD identical to the default DVI configuration except for the
        # contact numerical solver.  The default contact iteration cap (40)
        # is intentionally preserved for the solver comparison.
        self.newton_dvi_apgd = copy.deepcopy(self.newton_dvi)
        self.newton_dvi_apgd.solver_cfg.contact_solver_type = "sparse_apgd"
        self.newton_dvi_pspg = copy.deepcopy(self.newton_dvi_apgd)
        self.newton_dvi_pspg.solver_cfg.contact_solver_type = "sparse_pspg"


@configclass
class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AnymalCFlatEnvCfg_PLAY(AnymalCFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
