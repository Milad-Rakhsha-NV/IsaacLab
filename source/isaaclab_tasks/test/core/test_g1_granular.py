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
    cfg.bearing_range = (0.2, 0.6)
    with launch_simulation(cfg):
        env = GranularVelocityEnv(cfg)
        try:
            observations, _ = env.reset(seed=123)
            assert observations["policy"].shape == (2, 123)
            assert observations["critic"].shape == (2, 123)
            assert env.robot.num_joints == 37
            assert len(env.foot_ids) == 2
            assert env.rft.model.particle_count == 0
            assert torch.all((env.material[env.world_ids, 6] >= 0.2) & (env.material[env.world_ids, 6] <= 0.6))
            bodies = {row["body_label"].split("/")[-1] for row in env.rft.surface_inventory}
            for side in ("left", "right"):
                assert {f"{side}_hip_yaw_link", f"{side}_knee_link", f"{side}_ankle_roll_link"} <= bodies
            material = env.material.clone()
            surface = env.rft.surface_height.numpy().copy()
            env._reset_idx(torch.tensor([1], device=env.device))
            torch.testing.assert_close(env.material[env.world_ids[0]], material[env.world_ids[0]], rtol=0, atol=0)
            np.testing.assert_array_equal(env.rft.surface_height.numpy()[env.world_ids[0]], surface[env.world_ids[0]])
            cfg.material_difficulty = 0.0
            env._reset_idx(torch.tensor([1], device=env.device))
            torch.testing.assert_close(
                env.material[env.world_ids[1], 6],
                torch.tensor(cfg.sim.physics.solver_cfg.bearing, device=env.device),
            )
            cfg.material_difficulty = 1.0
            env._reset_idx(torch.tensor([1], device=env.device))
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


def test_g1_implicit_drive_parity_before_material_contact():
    """Preserve G1 drive behavior through both compositions before any sand contact.

    The remote particle block activates the real MPM partition/control path.
    A missing target, gain or joint-order update must differ from the native
    rigid trajectory even though every granular feedback wrench is zero.
    """
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    from isaaclab_tasks.core.velocity.config.g1.granular_env_cfg import G1GranularRigidEnvCfg

    reference = None
    for field in ("rigid", "rft", "mpm"):
        cfg = G1GranularRFTEnvCfg() if field == "rft" else G1GranularRigidEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.robot.init_state.pos = (0.0, 0.0, 2.0)
        cfg.sim.visualizer_cfgs = []
        cfg.gait = None
        cfg.scene.contact_forces = None
        cfg.surface_relative_height = False
        cfg.curriculum_enabled = False
        cfg.randomize_material = False
        cfg.randomize_commands = False
        cfg.observation_noise = False
        if field == "mpm":
            cfg.scene.sand = MPMObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sand",
                spawn=MPMGridCfg(
                    lower=(3.0, 3.0, 0.2),
                    upper=(3.1, 3.1, 0.3),
                    voxel_size=0.05,
                    particles_per_cell=2,
                    jitter=0,
                    material=MPMParticleMaterialCfg(density=1600, young_modulus=1.0e6),
                ),
            )
            cfg.sim.physics = NewtonCfg(
                num_substeps=4,
                solver_cfg=CouplerProxyCfg(
                    entries=[
                        CouplerEntryCfg(
                            name="rigid",
                            solver_cfg=cfg.sim.physics.solver_cfg,
                            bodies=[r"/World/envs/env_.*/Robot"],
                            include_static_shapes=True,
                        ),
                        CouplerEntryCfg(
                            name="mpm",
                            all_particles=True,
                            include_static_shapes=False,
                            include_child_joints=False,
                            in_place=True,
                            solver_cfg=MPMSolverCfg(
                                voxel_size=0.05,
                                max_active_cell_count=4096,
                                max_leaf_node_count=1024,
                                max_lower_node_count=64,
                                max_upper_node_count=16,
                                max_iterations=10,
                            ),
                        ),
                    ],
                    proxies=[
                        CouplerProxyMappingCfg(
                            source="rigid",
                            destination="mpm",
                            bodies=[r"/World/envs/env_.*/Robot"],
                            mode="staggered",
                            proxy_relaxation=0.1,
                        )
                    ],
                ),
            )
        trajectory = []
        with launch_simulation(cfg):
            env = GranularVelocityEnv(cfg)
            try:
                env.reset(seed=123)
                torch.testing.assert_close(env.robot.data.joint_pos.torch, env.robot.data.default_joint_pos.torch)
                for step in range(20):
                    action = 0.1 * torch.sin(torch.arange(37, device=env.device) + 0.2 * step)[None]
                    _, _, failed, timed_out, _ = env.step(action)
                    assert not failed.any() and not timed_out.any()
                    data = env.robot.data
                    assert float(data.joint_vel.torch.abs().max()) < 5.0, field
                    trajectory.append(
                        torch.cat(
                            (data.root_pos_w.torch, data.root_quat_w.torch, data.joint_pos.torch, data.joint_vel.torch),
                            dim=-1,
                        )
                        .cpu()
                        .numpy()
                    )
                if field == "rft":
                    assert not np.any(env.rft.body_impulse.numpy())
                elif field == "mpm":
                    assert not np.any(env.rft.get_proxy_body_wrenches("rigid", "mpm").numpy())
            finally:
                env.close()
        assert np.isfinite(trajectory).all()
        if reference is None:
            reference = np.array(trajectory)
        else:
            np.testing.assert_allclose(trajectory, reference, rtol=0, atol=1e-4, err_msg=field)
