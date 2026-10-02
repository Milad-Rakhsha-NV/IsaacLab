# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise independent trajectories and policies at the real simulator/runner boundary."""

from __future__ import annotations

import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env import MultiRobotLocomotionEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env_cfg import AntHumanoidEnvCfg

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env():
    cfg = AntHumanoidEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = test_devices(DeviceScope.DEFAULT_CUDA)[0]
    cfg.seed = 42
    with launch_simulation(cfg):
        environment = MultiRobotLocomotionEnv(cfg)
        yield environment
        environment.close()


def test_independent_episodes_and_heterogeneous_spaces(env, monkeypatch):
    """A timeout resets one robot/world only and preserves pre-reset observations and reward."""
    obs, _ = env.reset()
    assert obs["ant"].shape == (2, 60)
    assert obs["humanoid"].shape == (2, 87)
    actions = {agent: torch.zeros(2, robot.num_joints, device=env.device) for agent, robot in env.robots.items()}
    for _ in range(3):
        env.step(actions)
    # Make the terminal state distinguishable from the reset pose, without causing a fall.
    ant = env.robots["ant"]
    pose = ant.data.root_pose_w.torch[0:1].clone()
    pose[:, 0] += 0.5
    ant.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=env.device))
    env.agent_episode_length_buf["ant"][0] = env.robot_max_episode_length["ant"] - 1
    partner_before = env.robots["humanoid"].data.root_pos_w.torch.clone()
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert truncated["ant"].tolist() == [True, False]
    assert not terminated["ant"].any()
    assert not (terminated["humanoid"] | truncated["humanoid"]).any()
    assert env.agent_episode_length_buf["ant"].tolist() == [0, 4]
    assert env.agent_episode_length_buf["humanoid"].tolist() == [4, 4]
    assert info["ant"]["final_obs_mask"].tolist() == [True, False]
    assert not info["humanoid"]["final_obs_mask"].any()
    assert not torch.equal(info["ant"]["final_obs"][0], obs["ant"][0])
    assert (env.robots["humanoid"].data.root_pos_w.torch - partner_before).abs().max() < 0.1
    assert all(torch.isfinite(value).all() for value in (*obs.values(), *rewards.values()))
    env.step(actions)
    assert "final_obs" not in env.extras["ant"]

    # A real fall produces termination, not a timeout, and resets only the fallen slot.
    pose = ant.data.root_pose_w.torch[1:2].clone()
    pose[:, 2] = -2.0
    ant.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([1], device=env.device))
    _, _, terminated, truncated, _ = env.step(actions)
    assert terminated["ant"].tolist() == [False, True]
    assert not truncated["ant"].any()
    assert env.agent_episode_length_buf["humanoid"].tolist() == [6, 6]

    # An explicit UI reset ends every trajectory and still publishes terminal observations.
    with monkeypatch.context() as patch:
        patch.setattr(env.sim, "consume_reset_request", lambda: True)
        _, _, terminated, _, info = env.step(actions)
    for agent in env.possible_agents:
        assert terminated[agent].all()
        assert info[agent]["final_obs_mask"].all()
        assert not env.agent_episode_length_buf[agent].any()


def test_ippo_owns_separate_models_and_updates(env, tmp_path):
    """Real IPPO updates both unequal policies without sharing parameters or rollout memory."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper, import_skrl_runner

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry("IsaacContrib-Ant-Humanoid-Direct", "skrl_cfg_entry_point")
    cfg["agent"].update(rollouts=4, learning_epochs=1, mini_batches=1)
    cfg["agent"]["experiment"].update(directory=str(tmp_path), write_interval=0, checkpoint_interval=0)
    cfg["trainer"].update(timesteps=8, close_environment_at_exit=False, disable_progressbar=True)
    wrapped = SkrlVecEnvWrapper(env)
    runner = import_skrl_runner("torch", independent_agents=True)(wrapped, cfg)
    policies = [runner.agent.models[agent]["policy"] for agent in env.possible_agents]
    parameter_ids = [{id(p) for p in policy.parameters()} for policy in policies]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    all_parameter_ids = [
        {id(p) for model in runner.agent.models[agent].values() for p in model.parameters()}
        for agent in env.possible_agents
    ]
    assert all_parameter_ids[0].isdisjoint(all_parameter_ids[1])
    assert runner.agent.memories["ant"] is not runner.agent.memories["humanoid"]
    assert runner.agent.optimizers["ant"] is not runner.agent.optimizers["humanoid"]
    before = [[p.detach().clone() for p in policy.parameters()] for policy in policies]
    runner.run()
    for policy, original in zip(policies, before, strict=True):
        assert any(not torch.equal(p, initial) for p, initial in zip(policy.parameters(), original, strict=True))
        assert all(torch.isfinite(p).all() for p in policy.parameters())
    checkpoint = str(tmp_path / "independent.pt")
    runner.agent.save(checkpoint)
    saved = [[p.detach().clone() for p in policy.parameters()] for policy in policies]
    with torch.no_grad():
        for policy in policies:
            for parameter in policy.parameters():
                parameter.zero_()
    runner.agent.load(checkpoint)
    for policy, original in zip(policies, saved, strict=True):
        for parameter, expected in zip(policy.parameters(), original, strict=True):
            torch.testing.assert_close(parameter, expected)

    # At a time limit, IPPO must value the terminal state rather than the newly reset robot.
    agent = runner.agent
    observations, _ = wrapped.reset()
    states = wrapped.state()
    with torch.no_grad():
        actions, _ = agent.act(observations, states, timestep=8, timesteps=12)
        env.agent_episode_length_buf["ant"][0] = env.robot_max_episode_length["ant"] - 1
        next_obs, rewards, terminated, truncated, infos = wrapped.step(actions)
        assert truncated["ant"][0] and not terminated["ant"][0]
        terminal_values, _ = agent.values["ant"].act(
            {"observations": agent._observation_preprocessor["ant"](infos["ant"]["final_obs"]), "states": None},
            role="value",
        )
        terminal_values = agent._value_preprocessor["ant"](terminal_values, inverse=True)
        expected_reward = rewards["ant"][0].clone() + agent.cfg.discount_factor["ant"] * terminal_values[0]
        slot = agent.memories["ant"].memory_index
        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            next_states=wrapped.state(),
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=8,
            timesteps=12,
        )
        torch.testing.assert_close(agent.memories["ant"].get_tensor_by_name("rewards")[slot, 0], expected_reward)


def test_overlapping_robots_do_not_change_each_others_motion(env):
    """Overlapping the two robots preserves their separate-world motion, including ground contact."""
    env.reset()
    actions = {agent: torch.zeros(2, robot.num_joints, device=env.device) for agent, robot in env.robots.items()}
    shifts = {}
    thresholds = {agent: spec.task.termination_height for agent, spec in env.cfg.robots.items()}
    ids = torch.tensor([1], device=env.device)
    try:
        for agent, robot in env.robots.items():
            env.cfg.robots[agent].task.termination_height = -100.0
            shift = env.scene.env_origins[1] - env.scene.env_origins[0]
            if agent == "humanoid":
                shift = shift + torch.tensor([0.0, -4.0, 0.0], device=env.device)
            shifts[agent] = shift
            pose = robot.data.root_pose_w.torch[0:1].clone()
            pose[:, :3] += shift
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            robot.write_root_velocity_to_sim_index(root_velocity=robot.data.root_vel_w.torch[0:1].clone(), env_ids=ids)
            robot.write_joint_state_to_sim_index(
                position=robot.data.joint_pos.torch[0:1].clone(),
                velocity=robot.data.joint_vel.torch[0:1].clone(),
                env_ids=ids,
            )
        for _ in range(90):
            env.step(actions)
        for agent, robot in env.robots.items():
            torch.testing.assert_close(
                robot.data.root_pos_w.torch[1] - shifts[agent],
                robot.data.root_pos_w.torch[0],
                atol=0.01,
                rtol=0.0,
            )
            assert robot.data.root_pos_w.torch[:, 2].min() > -0.1
    finally:
        for agent, threshold in thresholds.items():
            env.cfg.robots[agent].task.termination_height = threshold


def test_rsl_rl_collects_once_and_restores_independent_policies(env, tmp_path):
    """RSL-RL trains unequal policies with one physics step and correct per-robot timeouts."""
    import copy

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.rsl_rl.independent import RslRlIndependentVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry("IsaacContrib-Ant-Humanoid-Direct", "rsl_rl_cfg_entry_point")
    cfg = handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())
    cfg.num_steps_per_env = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    cfg.device = env.device
    wrapped = RslRlIndependentVecEnvWrapper(env)
    runner = create_rsl_rl_runner(wrapped, cfg)
    ant, humanoid = [runner.runners[key].alg for key in ("ant", "humanoid")]
    parameters = [{id(p) for model in (alg.actor, alg.critic) for p in model.parameters()} for alg in (ant, humanoid)]
    assert parameters[0].isdisjoint(parameters[1])
    assert ant.storage is not humanoid.storage
    assert ant.optimizer is not humanoid.optimizer
    before = [copy.deepcopy(alg.actor.state_dict()) for alg in (ant, humanoid)]
    start = env.common_step_counter
    runner.learn(2)
    assert env.common_step_counter - start == 8  # not 8 times the number of policies
    for alg, original in zip((ant, humanoid), before, strict=True):
        assert any(not torch.equal(value, original[key]) for key, value in alg.actor.state_dict().items())
        assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
        assert alg.optimizer.state

    checkpoint = str(tmp_path / "rsl_independent.pt")
    runner.save(checkpoint)
    saved = [copy.deepcopy(alg.save()) for alg in (ant, humanoid)]
    policy = runner.get_inference_policy(env.device)
    with torch.inference_mode():
        expected_actions = policy(wrapped.get_observations())
        for alg in (ant, humanoid):
            alg.optimizer.state.clear()
            for model in (alg.actor, alg.critic):
                for parameter in model.parameters():
                    parameter.zero_()
                for buffer in model.buffers():
                    buffer.zero_()
    runner.current_learning_iteration = 0
    runner.load(checkpoint)
    assert runner.current_learning_iteration == 2
    with torch.inference_mode():
        actual_actions = runner.get_inference_policy(env.device)(wrapped.get_observations())
    for agent in env.possible_agents:
        torch.testing.assert_close(actual_actions[agent], expected_actions[agent])
    for alg, original in zip((ant, humanoid), saved, strict=True):
        # Includes normalization buffers and optimizer state, not just network weights.
        torch.testing.assert_close(alg.save(), original, check_device=False)
    runner.learn(1)
    assert runner.current_learning_iteration == 3
    assert env.common_step_counter - start == 12

    observations, _ = wrapped.reset()
    with torch.inference_mode():
        actions = {agent: item.alg.act(observations[agent]) for agent, item in runner.runners.items()}
        env.agent_episode_length_buf["ant"][:] = env.robot_max_episode_length["ant"] - 1
        # The second Ant falls on its timeout step: termination must take precedence.
        pose = env.robots["ant"].data.root_pose_w.torch[1:2].clone()
        pose[:, 2] = -2.0
        env.robots["ant"].write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([1], device=env.device))
        observations, rewards, dones, infos = wrapped.step(actions)
        assert dones["ant"].all() and not dones["humanoid"].any()
        assert infos["ant"]["time_outs"].tolist() == [True, False]
        value = ant.critic(wrapped.observation(infos["ant"]["final_obs"])).squeeze(-1)
        expected = rewards["ant"].clone()
        expected[0] += ant.gamma * value[0]
        runner.record_transition(observations, rewards, dones, infos)
        torch.testing.assert_close(ant.storage.rewards[0, :, 0], expected)
        torch.testing.assert_close(humanoid.storage.rewards[0, :, 0], rewards["humanoid"])
