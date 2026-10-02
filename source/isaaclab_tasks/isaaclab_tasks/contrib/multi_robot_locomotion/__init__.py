# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent locomotion policies for multiple robots in each world."""

import gymnasium as gym

for task, config in (
    ("IsaacContrib-Multi-Ant-Direct", "MultiAntEnvCfg"),
    ("IsaacContrib-Ant-Humanoid-Direct", "AntHumanoidEnvCfg"),
    ("IsaacContrib-Ant-Three-Four-Direct", "AntThreeFourEnvCfg"),
    ("IsaacContrib-Ant-Five-Specialist-Direct", "AntFiveSpecialistEnvCfg"),
):
    gym.register(
        id=task,
        entry_point=f"{__name__}.multi_robot_env:MultiRobotLocomotionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.multi_robot_env_cfg:{config}",
            "default_agent": "skrl",
            "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ippo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{__name__}.agents:skrl_ippo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:IndependentPPORunnerCfg",
        },
    )
