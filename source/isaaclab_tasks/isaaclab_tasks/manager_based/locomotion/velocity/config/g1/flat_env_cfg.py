# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs.mdp import actions as mdp
from isaaclab.envs.mdp import observations as obs_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, UniformNoiseCfg as Unoise

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
    newton_dvi = NewtonCfg(
        solver_cfg=DVISolverCfg(
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
            joint_iterative_refinement_steps=1,
            # Arm/hand armature only; finger joints keep URDF default (0.001).
            # Paradoxically, the resulting finger oscillations provide numerical
            # damping that prevents LDL solver NaN.  Finger joints are excluded
            # from the RL obs/action/reward loop so the oscillations don't
            # affect training.
            armature_override={
                "shoulder": 0.05, "elbow": 0.05, "wrist": 0.05, "hand": 0.05, "finger": 0.05,
            },
        ),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    newton_dvi_implicit = NewtonCfg(
        solver_cfg=DVISolverCfg(
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
            actuator_integration="implicit",
            joint_limit_ke_scale=0.1,
            joint_limit_solver_type="sparse_jacobi",
            joint_iterative_refinement_steps=1,
            armature_override={
                "shoulder": 0.05, "elbow": 0.05, "wrist": 0.05, "hand": 0.05, "finger": 0.05,
            },
        ),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    newton_dvi_semi_implicit = NewtonCfg(
        solver_cfg=DVISolverCfg(
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
            joint_iterative_refinement_steps=1,
            armature_override={
                "shoulder": 0.05, "elbow": 0.05, "wrist": 0.05, "hand": 0.05, "finger": 0.05,
            },
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

        # -- Locomotion joints only (exclude finger joints from RL loop) --
        # Finger joints (left/right_{zero..six}_joint) serve no purpose for
        # locomotion and cause massive obs noise + reward penalties in DVI.
        _LOCO_JOINTS = [
            ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*",
            "torso_joint",
            ".*_shoulder_.*", ".*_elbow_.*",
        ]
        _loco_asset = SceneEntityCfg("robot", joint_names=_LOCO_JOINTS)

        # Actions: only locomotion joints
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_LOCO_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        # Observations: restrict joint_pos, joint_vel, actions to locomotion joints
        self.observations.policy.joint_pos = ObsTerm(
            func=obs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": _loco_asset},
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=obs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": _loco_asset},
        )

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
        # Disable finger deviation penalty — fingers excluded from RL loop
        self.rewards.joint_deviation_fingers.weight = 0.0

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
