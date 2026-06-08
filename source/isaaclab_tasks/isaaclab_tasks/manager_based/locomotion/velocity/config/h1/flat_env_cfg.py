# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import H1RoughEnvCfg


DVI_SOLVER_CFG = DVISolverCfg(
    joint_solver_type="sparse_ldl",
    joint_max_iterations=50,
    joint_alpha=0.0,
    joint_recovery_speed=100000.0,
    joint_position_correction=False,
    contact_solver_type="sparse_jacobi",
    contact_max_iterations=40,
    contact_alpha=0.0,
    contact_recovery_speed=10000.0,
    contact_position_correction=False,
    angular_damping=0.0,
    actuator_integration="semi_implicit",
    joint_limit_ke_scale=0.1,
    joint_limit_solver_type="sparse_jacobi",
    joint_limit_recovery_speed=1.0,
    joint_iterative_refinement_steps=1,
)

DVI_NEWTON_CFG = NewtonCfg(
    solver_cfg=DVI_SOLVER_CFG,
    num_substeps=1,
    debug_mode=False,
    use_cuda_graph=True,
    collapse_fixed_joints=True,
    default_shape_cfg=NewtonShapeCfg(gap=0.005),
    collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=2**21),
)


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=15,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_dvi = DVI_NEWTON_CFG
    physx = default


@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Lower spawn height slightly (default 1.05 starts slightly airborne)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.98)

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6


class H1FlatEnvCfg_PLAY(H1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
