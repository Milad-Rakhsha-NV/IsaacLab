# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold-pose environment for the Disney DR Legs closed-loop biped (Kamino solver).

The robot must keep its pelvis upright at a target height.
"""

from isaaclab_newton.physics import (
    DVISolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.dr_legs.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.dr_legs import DR_LEGS_ACTUATED_JOINTS, DR_LEGS_IMPLICIT_PD_CFG

##
# Shared constants
##

_NUM_ENVS = 4096

_ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=DR_LEGS_ACTUATED_JOINTS, preserve_order=True)

_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
)


def _dvi_solver_cfg(actuator_integration: str = "semi_implicit") -> DVISolverCfg:
    """DVI solver preset for DR Legs.

    Uses the same parameters as the validated locomotion envs (Go2/Ant/Humanoid):
    sparse LDL joints, sparse Jacobi contacts, no position correction.
    """
    return DVISolverCfg(
        joint_solver_type="sparse_ldl",
        joint_max_iterations=50,
        joint_alpha=0.005,
        joint_recovery_speed=100000.0,
        joint_position_correction=False,
        # Closed-loop linkages produce redundant (rank-deficient) constraints; the
        # default joint_reg=1e-6 can hit a near-zero LDL pivot on a few envs and NaN.
        # diagonal_precondition is already on by default; bump reg to 1e-4 (validated
        # on the four-bar + standalone DR Legs tests) for a stable factorization.
        joint_reg=1e-4,
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
    """NewtonCfg for the DVI solver on DR Legs."""
    return NewtonCfg(
        solver_cfg=_dvi_solver_cfg(actuator_integration),
        # Closed-loop DR Legs needs a small solver dt to hold the 6 parallel loops.
        # sim dt is 0.004s; substeps set per Milad to match the Kamino baseline (2)
        # (see standalone four-bar / DR Legs closed-loop tests).
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=False,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )


@configclass
class DrLegsPhysicsCfg(PresetCfg):
    """Physics backend preset for DR Legs (Kamino + DVI)."""

    default: NewtonCfg = _dvi_newton_cfg("semi_implicit")
    newton_dvi: NewtonCfg = _dvi_newton_cfg("semi_implicit")
    newton_dvi_implicit: NewtonCfg = _dvi_newton_cfg("implicit")
    newton_dvi_semi_implicit: NewtonCfg = _dvi_newton_cfg("semi_implicit")


##
# Scene definition
##


@configclass
class HoldPoseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), physics_material=_PHYSICS_MATERIAL),
    )

    robot = DR_LEGS_IMPLICIT_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DR_LEGS_ACTUATED_JOINTS,
        preserve_order=True,
        scale=0.3,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ACTUATED_JOINT_CFG})
        setpoint_hist = ObsTerm(
            func=mdp.joint_position_setpoints,
            params={"action_term_name": "joint_pos"},
            history_length=2,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": _ACTUATED_JOINT_CFG,
            "operation": "scale",
            "distribution": "uniform",
            "armature_distribution_params": (0.8, 1.2),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": _ACTUATED_JOINT_CFG,
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=5.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.265})
    lin_vel = RewTerm(func=mdp.base_lin_vel_l2, weight=-1.0)
    ang_vel = RewTerm(func=mdp.base_ang_vel_l2, weight=-0.5)
    joint_torque = RewTerm(
        func=mdp.joint_pd_command_l2,
        weight=-1.0e-5,
        params={"asset_cfg": _ACTUATED_JOINT_CFG, "stiffness": 5.0, "damping": 0.2},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_rate2 = RewTerm(func=mdp.ActionRate2L2, weight=-0.01)
    joint_pos_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": _ACTUATED_JOINT_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.12})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


##
# Environment configuration
##


@configclass
class DrLegsHoldPoseEnvCfg(ManagerBasedRLEnvCfg):
    """DR Legs hold-pose environment (Newton/Kamino backend)."""

    scene: HoldPoseSceneCfg = HoldPoseSceneCfg(num_envs=_NUM_ENVS, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=0.004,
        render_interval=5,
        physics=DrLegsPhysicsCfg(),
        physics_material=_PHYSICS_MATERIAL,
    )

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10.0
        self.viewer.eye = (1.5, 0.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.265)
