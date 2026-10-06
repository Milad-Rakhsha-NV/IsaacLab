# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import preset

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 65
        newton_mjwarp.solver_cfg.nconmax = 35
        self.sim.physics.default = newton_mjwarp
        # Preserve the paper's DVI PD drives and collapsed-foot contact selection.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = preset(default=".*_foot", newton_dvi=".*_calf")
        # Use rigid joint constraints for the validated paper-budget DVI run.
        self.sim.physics.newton_dvi.solver_cfg.joint_alpha = 0.0
        self.sim.physics.newton_dvi.solver_cfg.contact_recovery_speed = 1.0
        self.sim.physics.newton_dvi.solver_cfg.coupling_iterations = 2
        self.sim.physics.newton_dvi.solver_cfg.post_stabilize_joints = False
        self.scene.robot.actuators["base_legs"] = preset(
            default=self.scene.robot.actuators["base_legs"],
            newton_dvi=ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                joint_effort_limit=23.5,
                joint_velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
            ),
        )
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # curriculum
        self.curriculum.terrain_levels = None
