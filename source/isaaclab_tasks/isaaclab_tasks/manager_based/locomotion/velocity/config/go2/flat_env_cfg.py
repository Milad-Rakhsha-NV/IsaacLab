# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


def _dvi_solver_cfg(actuator_integration: str = "semi_implicit") -> DVISolverCfg:
    """Build DVISolverCfg with the given actuator integration mode."""
    return DVISolverCfg(
        joint_solver_type="sparse_ldl",
        joint_alpha=0.005,
        joint_recovery_speed=100000.0,
        joint_position_correction=False,
        contact_solver_type="sparse_jacobi",
        contact_max_iterations=20,
        contact_alpha=0.0,
        contact_recovery_speed=1.0,
        contact_position_correction=False,
        angular_damping=0.0,
        actuator_integration=actuator_integration,
        joint_limit_ke_scale=0.1,
        joint_limit_solver_type="sparse_jacobi",
        joint_iterative_refinement_steps=1,
    )


def _dvi_newton_cfg(actuator_integration: str = "semi_implicit") -> NewtonCfg:
    """Build NewtonCfg for DVI with the given actuator integration mode."""
    return NewtonCfg(
        solver_cfg=_dvi_solver_cfg(actuator_integration),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )


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
        collapse_fixed_joints=True,
    )
    newton_dvi = _dvi_newton_cfg("semi_implicit")
    newton_dvi_implicit = _dvi_newton_cfg("implicit")
    newton_dvi_semi_implicit = _dvi_newton_cfg("semi_implicit")
    physx = default


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override actuators: DVI presets use ImplicitActuator so ke/kd flow to solver
        self.scene.robot.actuators["base_legs"] = preset(
            default=self.scene.robot.actuators["base_legs"],
            newton_dvi=ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=25.0,
                damping=0.5,
            ),
            newton_dvi_implicit=ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=25.0,
                damping=0.5,
            ),
            newton_dvi_semi_implicit=ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=25.0,
                damping=0.5,
            ),
        )

        # override rewards — same weights for all solvers
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

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
