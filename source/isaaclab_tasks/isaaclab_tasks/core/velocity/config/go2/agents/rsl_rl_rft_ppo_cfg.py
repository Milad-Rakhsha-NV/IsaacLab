# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO using the existing runner; MPM is inference-only."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg


@configclass
class Go2RFTPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    experiment_name = "go2_rft"
    max_iterations = 500
    save_interval = 50
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_rft"
        self.max_iterations = 500


@configclass
class Go2RigidPPORunnerCfg(Go2RFTPPORunnerCfg):
    """Same PPO and policy interface for the rigid-ground control."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_rigid"
        self.max_iterations = 1000
        self.init_at_random_ep_len = False


@configclass
class Go2RFTCurriculumPPORunnerCfg(Go2RFTPPORunnerCfg):
    """Reset curriculum with complete initial episodes for its progression metric."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_rft_curriculum"
        self.max_iterations = 1500
        self.init_at_random_ep_len = False
