# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import ChronoSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=35,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_chrono = NewtonCfg(
        solver_cfg=ChronoSolverCfg(
            joint_solver_type="sparse_ldl",
            joint_max_iterations=50,
            contact_solver_type="sparse_jacobi",
            contact_max_iterations=50,
            contact_recovery_speed=1.0,
            angular_damping=0.05,
            use_implicit_pd=True,
            max_velocity=20.0,
        ),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.001, gap=0.01),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
    )
    physx = default


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = preset(default=0.25, newton_chrono=0.5)
        self.rewards.feet_air_time.params["threshold"] = preset(default=0.5, newton_chrono=0.3)

        # Chrono DVI has noisier accelerations and vertical bouncing;
        # reduce corresponding penalties so the optimizer focuses on locomotion.
        self.rewards.dof_acc_l2.weight = preset(default=-2.5e-7, newton_chrono=-5.0e-8)
        self.rewards.lin_vel_z_l2.weight = preset(default=-2.0, newton_chrono=-1.0)
        self.rewards.action_rate_l2.weight = preset(default=-0.01, newton_chrono=-0.005)
        self.rewards.track_lin_vel_xy_exp.weight = preset(default=1.5, newton_chrono=2.0)
        self.rewards.track_ang_vel_z_exp.weight = preset(default=0.75, newton_chrono=1.0)

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
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
