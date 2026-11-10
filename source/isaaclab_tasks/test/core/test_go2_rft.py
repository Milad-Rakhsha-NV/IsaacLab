# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Go2 task contracts absent from Newton's local-traction unit tests."""

import numpy as np
import pytest
import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.go2.rft_env import Go2RFTEnv
from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTEnvCfg

pytestmark = pytest.mark.kitless


def test_go2_rft_articulation_and_masked_material_reset():
    """Full-limb binding, finite articulated response, and clone-local reset.

    A feet-only selection, global material rewrite, or wrong actor layout must
    fail here; plate-only tests cannot detect these Isaac Lab integration errors.
    """
    cfg = Go2RFTEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.visualizer_cfgs = []
    with launch_simulation(cfg):
        env = Go2RFTEnv(cfg)
        try:
            obs, _ = env.reset(seed=123)
            assert obs["policy"].shape == (4, 45)
            assert obs["critic"].shape == (4, 48)
            assert env.rft.model.particle_count == 0
            names = {row["body_label"].split("/")[-1] for row in env.rft.surface_inventory}
            for leg in ("FL", "FR", "RL", "RR"):
                assert {f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf", f"{leg}_foot"} <= names
            material = env.material.clone()
            pose = env.robot.data.root_pos_w.torch.clone()
            env._reset_idx(torch.tensor([1], device=env.device))
            keep = env.world_ids[torch.tensor([0, 2, 3], device=env.device)]
            torch.testing.assert_close(env.material[keep], material[keep], rtol=0, atol=0)
            assert not torch.equal(env.material[env.world_ids[1]], material[env.world_ids[1]])
            env.scene.write_data_to_sim()
            env.sim.forward()
            torch.testing.assert_close(env.robot.data.root_pos_w.torch[[0, 2, 3]], pose[[0, 2, 3]])
            peak = 0.0
            for _ in range(30):
                obs, reward, _, _, _ = env.step(torch.zeros(4, 12, device=env.device))
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(reward).all()
                peak = max(peak, float(np.linalg.norm(env.rft.body_f.numpy())))
            assert peak > 1.0
        finally:
            env.close()
