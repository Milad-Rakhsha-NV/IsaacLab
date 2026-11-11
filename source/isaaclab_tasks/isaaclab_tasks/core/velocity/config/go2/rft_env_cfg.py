# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct Go2 locomotion in a particle-free, randomized granular field."""

from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.physics.rft_solver_cfg import RFTSolverCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def rigid_substrate_cfg(height: float) -> AssetBaseCfg:
    """Use identical substrate geometry in rigid controls and finite-layer experiments."""
    return AssetBaseCfg(
        prim_path="/World/Substrate",
        # The rendered mesh has finite size; the collision Plane is infinite.
        # A finite box misses outer clones once a 4096-world grid exceeds 100 m.
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, height)),
    )


@configclass
class Go2RFTSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    bed = AssetBaseCfg(
        prim_path="/World/BedVisual",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.3),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.68, 0.46, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.15)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    substrate: AssetBaseCfg | None = None
    contact_forces: ContactSensorCfg | None = None


@configclass
class Go2GaitCfg:
    """Rigid-control posture and contact criteria, independent of material parameters."""

    hip_deviation_weight = 1.0
    joint_deviation_weight = 0.05
    stand_deviation_weight = 0.5
    joint_acceleration_weight = 2.5e-7
    vertical_velocity_weight = 2.0
    angular_velocity_weight = 0.05
    soft_joint_limit_weight = 1.0
    foot_sliding_weight = 0.1
    air_time_weight = 0.25
    air_time_target = 0.3
    undesired_contact_weight = 1.0
    stance_weight = 20.0
    stance_half_width = 0.10
    min_signed_foot_y = 0.01


@configclass
class Go2RFTEnvCfg(DirectRLEnvCfg):
    """Experimental RFT with a labeled near-rest normal-bearing extension.

    The default half-space has no rigid support plane or MPM asset.
    ``bed_depth`` explicitly enables a finite layer with a rigid substrate.
    Material randomization changes persistent GPU world rows on reset.
    """

    decimation = 4
    episode_length_s = 10.0
    action_space = 12
    observation_space = 45
    state_space = 48
    action_scale = 0.25
    target_height = 0.32
    randomize_material = True
    randomize_commands = True
    observation_noise = True
    command_speed = 0.35
    density_range = (1400.0, 1800.0)
    friction_range = (0.3, 0.5)
    roughness_range = (0.6, 1.0)
    strength_multiplier_range = (0.8, 1.2)
    bed_height_range = (-0.02, 0.02)
    bed_depth: float | None = None
    """Optional finite-layer depth [m]; creates a labeled rigid substrate. None keeps a half-space."""
    material_difficulty = 1.0
    gait: Go2GaitCfg | None = None
    surface_relative_height = False
    """Use the sampled surface for spawning, height reward and termination in new experiments."""
    curriculum_enabled = False
    curriculum_initial_difficulty = 0.0
    curriculum_step = 0.1
    curriculum_min_episodes = 4096
    curriculum_patience = 2
    curriculum_success_threshold = 0.95
    curriculum_error_threshold = 0.1
    curriculum_retreat_threshold = 0.7
    curriculum_anchor_fraction = 0.1
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(
            num_substeps=4,
            solver_cfg=RFTSolverCfg(
                bearing=0.87574,
                velocity_regularization=0.04,
                segments=12,
                njmax=128,
                nconmax=64,
                use_mujoco_contacts=True,
            ),
        ),
        visualizer_cfgs=VisualizerCfg(eye=(1.8, -2.3, 1.0), lookat=(0.0, 0.0, 0.25)),
    )
    scene: Go2RFTSceneCfg = Go2RFTSceneCfg(num_envs=4096, env_spacing=3.0)


@configclass
class Go2RFTCurriculumEnvCfg(Go2RFTEnvCfg):
    """Success-driven material randomization with full commands from the first reset."""

    curriculum_enabled = True
    surface_relative_height = True


@configclass
class Go2RigidEnvCfg(Go2RFTEnvCfg):
    """Rigid-flat control with the same policy contract and rewards as the granular task."""

    randomize_material = False
    surface_relative_height = True
    gait: Go2GaitCfg = Go2GaitCfg()
    sim: SimulationCfg = replace(
        Go2RFTEnvCfg().sim,
        physics=NewtonCfg(
            num_substeps=4,
            solver_cfg=MJWarpSolverCfg(njmax=128, nconmax=64, use_mujoco_contacts=True),
        ),
    )
    scene: Go2RFTSceneCfg = replace(
        Go2RFTEnvCfg().scene,
        bed=None,
        robot=replace(
            Go2RFTEnvCfg().scene.robot,
            spawn=replace(
                Go2RFTEnvCfg().scene.robot.spawn,
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
            ),
        ),
        substrate=rigid_substrate_cfg(0.0),
        contact_forces=ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True, update_period=0.005
        ),
    )
