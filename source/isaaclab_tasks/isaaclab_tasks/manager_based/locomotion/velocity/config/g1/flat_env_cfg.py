# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import ChronoSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs.mdp import actions as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=95,
            nconmax=10,
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
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            joint_position_correction=False,
            contact_solver_type="sparse_jacobi",
            contact_max_iterations=40,
            contact_alpha=0.0,
            contact_recovery_speed=10000.0,
            contact_position_correction=False,
            angular_damping=0.01,
            use_implicit_pd=True,
            joint_limit_ke_scale=0.1,
            joint_limit_solver_type="sparse_jacobi",
            joint_iterative_refinement_steps=1,
        ),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Zero effort scale for finger joints — they add noise to the policy
        # without contributing to locomotion
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale={
                ".*_hip_.*": 0.5,
                ".*_knee_joint": 0.5,
                "torso_joint": 0.5,
                ".*_ankle_.*": 0.5,
                ".*_shoulder_.*": 0.5,
                ".*_elbow_.*": 0.5,
                ".*_zero_joint": 0.0,
                ".*_one_joint": 0.0,
                ".*_two_joint": 0.0,
                ".*_three_joint": 0.0,
                ".*_four_joint": 0.0,
                ".*_five_joint": 0.0,
                ".*_six_joint": 0.0,
            },
            use_default_offset=True,
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
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
