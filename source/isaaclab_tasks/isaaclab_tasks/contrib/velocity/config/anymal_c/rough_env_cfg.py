# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = replace(ANYMAL_C_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        # Retain the MJWarp armature used by the paper's learned actuator.
        legs = self.scene.robot.actuators["legs"]
        legs.armature = preset(default=legs.armature, newton_mjwarp=0.01)
        # The paper's DVI runs used solver-side PD, rather than the stock learned actuator.
        self.scene.robot.actuators["legs"] = preset(
            default=self.scene.robot.actuators["legs"],
            newton_dvi=ImplicitActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                joint_effort_limit=80.0,
                joint_velocity_limit=7.5,
                stiffness=80.0,
                damping=5.0,
                armature=0.06,
            ),
        )
        # Fixed FOOT links collapse into SHANK in the DVI prototype.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = preset(default=".*FOOT", newton_dvi=".*SHANK")
