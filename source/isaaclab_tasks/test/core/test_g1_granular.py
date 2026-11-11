# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid action layout and full-leg granular coupling absent from the Go2 fixture."""

import numpy as np
import pytest
import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.g1.granular_env_cfg import G1GranularRFTEnvCfg
from isaaclab_tasks.core.velocity.granular_env import GranularVelocityEnv

pytestmark = pytest.mark.kitless


def test_g1_full_leg_coupling_and_clone_local_reset():
    """Exercise the 37-joint humanoid and its added thigh/calf collision proxies."""
    cfg = G1GranularRFTEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.visualizer_cfgs = []
    cfg.curriculum_enabled = False
    with launch_simulation(cfg):
        env = GranularVelocityEnv(cfg)
        try:
            observations, _ = env.reset(seed=123)
            assert observations["policy"].shape == (2, 123)
            assert observations["critic"].shape == (2, 123)
            assert env.robot.num_joints == 37
            assert len(env.foot_ids) == 2
            assert env.rft.model.particle_count == 0
            bodies = {row["body_label"].split("/")[-1] for row in env.rft.surface_inventory}
            for side in ("left", "right"):
                assert {f"{side}_hip_yaw_link", f"{side}_knee_link", f"{side}_ankle_roll_link"} <= bodies
            material = env.material.clone()
            surface = env.rft.surface_height.numpy().copy()
            env._reset_idx(torch.tensor([1], device=env.device))
            torch.testing.assert_close(env.material[env.world_ids[0]], material[env.world_ids[0]], rtol=0, atol=0)
            np.testing.assert_array_equal(env.rft.surface_height.numpy()[env.world_ids[0]], surface[env.world_ids[0]])
            peak = 0.0
            for _ in range(30):
                observations, reward, _, _, _ = env.step(torch.zeros(2, 37, device=env.device))
                assert torch.isfinite(observations["policy"]).all()
                assert torch.isfinite(reward).all()
                assert not np.any(env.rft.rigid_solver.mjw_data.overflow.numpy())
                peak = max(peak, float(np.linalg.norm(env.rft.body_f.numpy())))
            assert peak > 1.0
        finally:
            env.close()
