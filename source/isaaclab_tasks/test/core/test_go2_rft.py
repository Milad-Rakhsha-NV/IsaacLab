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
from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTEnvCfg, Go2RigidEnvCfg

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

            # Curriculum decisions change only the next reset's field. Failed
            # episodes must not promote, and old-stage episodes cannot qualify.
            cfg.curriculum_enabled = True
            cfg.curriculum_min_episodes = 2
            cfg.curriculum_patience = 1
            cfg.curriculum_anchor_fraction = 0.0
            env._reset_idx(torch.arange(4, device=env.device))
            ids = torch.tensor([0, 1], device=env.device)
            env.steps_sum[ids] = 20
            env.error_sum[ids] = 0.01
            env.reset_terminated[ids] = True
            env.reset_time_outs[ids] = False
            env._reset_idx(ids)
            assert float(env.curriculum_difficulty) == 0
            untouched = env.material[env.world_ids[2:]].clone()
            env.steps_sum[ids] = env.max_episode_length
            env.error_sum[ids] = env.max_episode_length * 0.05
            env.reset_terminated[ids] = False
            env.reset_time_outs[ids] = True
            env._reset_idx(ids)
            assert float(env.curriculum_difficulty) == pytest.approx(0.1)
            torch.testing.assert_close(env.material[env.world_ids[2:]], untouched, rtol=0, atol=0)

            cfg.surface_relative_height = True
            env._reset_idx(ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            torch.testing.assert_close(env.relative_base_height()[ids], env.robot.data.default_root_pose.torch[ids, 2])

            # The material curriculum must leave the full command distribution
            # unchanged, including lateral and yaw commands at nominal material.
            all_ids = torch.arange(4, device=env.device)
            env.curriculum_difficulty.zero_()
            torch.manual_seed(71)
            env._reset_idx(all_ids)
            initial_commands = env.commands.clone()
            assert initial_commands[:, 1].abs().max() > 0.01
            assert initial_commands[:, 2].abs().max() > 0.03
            env.curriculum_difficulty.fill_(0.8)
            torch.manual_seed(71)
            env._reset_idx(all_ids)
            torch.testing.assert_close(env.commands, initial_commands, rtol=0, atol=0)

            env.curriculum_difficulty.fill_(1)
            env._reset_idx(torch.arange(4, device=env.device))
            assert not env.episode_anchor.any()
            env.steps_sum[ids] = 20
            env.reset_terminated[ids] = True
            env.reset_time_outs[ids] = False
            env._reset_idx(ids)
            assert float(env.curriculum_difficulty) == 1  # Final distribution stays fixed.

            # In free space, internal actuator/joint forces cannot change total
            # linear momentum. Every applied granular impulse must reach MJWarp,
            # including loads on fixed foot bodies, exactly once.
            cfg.curriculum_enabled = False
            cfg.randomize_material = False
            cfg.surface_relative_height = False
            env.rft.surface_height.fill_(1.0)
            env.material[:, 3] = 100.0
            env._reset_idx(torch.arange(4, device=env.device))
            env.scene.write_data_to_sim()
            env.sim.forward()
            model = env.rft.model
            state = env.sim.backend_manager.backend.state_0
            mass = model.body_mass.numpy().astype(np.float64)
            momentum = (mass[:, None] * state.body_qd.numpy()[:, :3]).sum(0)
            before = env.rft.body_impulse.numpy().copy()
            env.step(torch.zeros(4, 12, device=env.device))
            change = (mass[:, None] * state.body_qd.numpy()[:, :3]).sum(0) - momentum
            impulse = (env.rft.body_impulse.numpy() - before)[:, :3].sum(0)
            assert np.linalg.norm(impulse) > 0.01
            gravity = model.gravity.numpy()[model.body_world.numpy()]
            expected = impulse + (mass[:, None] * gravity).sum(0) * env.step_dt
            np.testing.assert_allclose(change, expected, rtol=0, atol=1e-3)
        finally:
            env.close()


def test_zero_layer_recovers_rigid_articulation():
    """With prescribed identical actions, an empty layer adds no force to rigid-flat dynamics."""
    trajectories = []
    actions = np.random.default_rng(17).normal(0, 0.1, size=(100, 2, 12)).astype(np.float32)
    for cfg in (Go2RigidEnvCfg(), Go2RFTEnvCfg()):
        cfg.seed = 17
        cfg.scene.num_envs = 2
        cfg.scene.env_spacing = 240.0  # Exercise clones beyond the original 100 m visual bed.
        cfg.sim.visualizer_cfgs = []
        cfg.observation_noise = False
        cfg.randomize_commands = False
        cfg.randomize_material = False
        cfg.surface_relative_height = True
        if not isinstance(cfg, Go2RigidEnvCfg):
            cfg.bed_depth = 0.0
            cfg.scene.robot = Go2RigidEnvCfg().scene.robot  # Match the control's self-collision configuration.
        with launch_simulation(cfg):
            env = Go2RFTEnv(cfg)
            try:
                env.reset(seed=17)
                history = []
                for action in actions:
                    _, reward, failed, _, _ = env.step(torch.as_tensor(action, device=env.device))
                    assert torch.isfinite(reward).all()
                    assert not failed.any()
                    history.append(
                        torch.cat((env.robot.data.root_pos_w.torch, env.robot.data.joint_pos.torch), -1).cpu().numpy()
                    )
                assert float(env.relative_base_height().min()) > 0.25
                if env.material is not None:
                    np.testing.assert_array_equal(env.rft.body_impulse.numpy(), 0)
                trajectories.append(np.array(history))
                if isinstance(cfg, Go2RigidEnvCfg):
                    # A deliberately crossed pose at a safe torso height must
                    # terminate even with no fall, tilt, or nonfinite state.
                    q = env.robot.data.default_joint_pos.torch.clone()
                    q[:, env.robot.joint_names.index("RL_hip_joint")] = -0.8
                    q[:, env.robot.joint_names.index("RR_hip_joint")] = 0.8
                    pose = env.robot.data.default_root_pose.torch.clone()
                    pose[:, :3] += env.scene.env_origins
                    pose[:, 2] = 0.6
                    ids = torch.arange(2, device=env.device)
                    env.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                    env.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    feet = env.robot.data.body_link_pos_w.torch[:, env.foot_ids]
                    assert (feet[:, 2, 1] < feet[:, 3, 1]).all()
                    _, _, crossed, _, _ = env.step(torch.zeros((2, 12), device=env.device))
                    assert crossed.all()
            finally:
                env.close()
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=5e-4)
