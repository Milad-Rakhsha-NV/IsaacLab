# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph policy contracts: enumeration/padding invariance and real shared PPO collection."""

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.multi_robot_locomotion.agents.rsl_rl_ppo_cfg import IndependentPPORunnerCfg
from isaaclab_tasks.contrib.multi_robot_locomotion.graph_policy import GraphModel
from isaaclab_tasks.contrib.multi_robot_locomotion.graph_training import GraphVecEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env import MultiRobotLocomotionEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env_cfg import AntThreeFourEnvCfg


def test_graph_policy_permutation_padding_and_variable_joint_count():
    """Joint renumbering only reorders actions; padding changes neither policy/value nor PPO density."""
    torch.manual_seed(42)
    obs = TensorDict(
        {
            "global": torch.randn(2, 12),
            "nodes": torch.randn(2, 6, 29),
            "mask": torch.ones(2, 6, dtype=torch.bool),
            "adjacency": torch.eye(6).expand(2, -1, -1).clone(),
        },
        batch_size=[2],
    )
    obs["adjacency"][:, 0, 1] = obs["adjacency"][:, 1, 0] = 1
    actor = GraphModel(obs, {}, "actor", 6)
    critic = GraphModel(obs, {}, "critic", 1)
    actor.eval()
    critic.eval()
    action, value = actor(obs), critic(obs)
    order = torch.tensor([3, 1, 5, 0, 2, 4])
    permuted = obs.clone()
    permuted["nodes"] = obs["nodes"][:, order]
    permuted["adjacency"] = obs["adjacency"][:, order][:, :, order]
    torch.testing.assert_close(actor(permuted), action[:, order])
    torch.testing.assert_close(critic(permuted), value)
    padded = TensorDict(
        {
            "global": obs["global"],
            "nodes": torch.randn(2, 10, 29) * 10000,
            "mask": torch.zeros(2, 10, dtype=torch.bool),
            "adjacency": torch.ones(2, 10, 10),
        },
        batch_size=[2],
    )
    padded["nodes"][:, :6] = obs["nodes"]
    padded["mask"][:, :6] = True
    padded["adjacency"][:, :6, :6] = obs["adjacency"]
    torch.testing.assert_close(actor(padded)[:, :6], action)
    torch.testing.assert_close(actor(padded)[:, 6:], torch.zeros(2, 4))
    torch.testing.assert_close(critic(padded), value)
    actor(obs, stochastic_output=True)
    log_prob, entropy = actor.get_output_log_prob(action), actor.output_entropy
    actor(padded, stochastic_output=True)
    padded_actions = torch.cat((action, torch.full((2, 4), 1000.0)), dim=-1)
    torch.testing.assert_close(actor.get_output_log_prob(padded_actions), log_prob)
    torch.testing.assert_close(actor.output_entropy, entropy)
    shapes = {k: v.shape for k, v in actor.state_dict().items()}
    for count in (8, 10):
        graph = padded.clone()
        graph["mask"][:] = torch.arange(10) < count
        assert actor(graph).shape == (2, 10)
        assert {k: v.shape for k, v in actor.state_dict().items()} == shapes
    # Every morphology reaches the same decision-core parameters.
    for graph in (obs, padded):
        actor.zero_grad()
        actor(graph).square().sum().backward()
        assert actor.decider[0].weight.grad.abs().sum() > 0


def test_shared_graph_real_rollout_timeout_update_and_reload(tmp_path):
    """Real unequal Ant graphs share one PPO, one physical step, correct terminal bootstrap and saved weights."""
    cfg = AntThreeFourEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = test_devices(DeviceScope.DEFAULT_CUDA)[0]
    cfg.seed = 42
    with launch_simulation(cfg):
        raw = MultiRobotLocomotionEnv(cfg)
        try:
            env = GraphVecEnv(raw)
            obs = env.get_observations()
            assert obs["mask"].sum(-1).tolist() == [6] * 4 + [8] * 4
            # Pooled graph observations must retain all named physical joints in action order.
            for name, graph in env.graphs.items():
                assert graph.static.shape == (raw.robots[name].num_joints, 19)
                assert (graph.parents < 0).sum() == graph.num_feet
            train = to_dict(IndependentPPORunnerCfg())
            prefix = "isaaclab_tasks.contrib.multi_robot_locomotion"
            train["actor"] = {"class_name": f"{prefix}.graph_policy:GraphModel"}
            train["critic"] = {"class_name": f"{prefix}.graph_policy:GraphModel"}
            train["algorithm"]["class_name"] = f"{prefix}.graph_training:GraphPPO"
            train["obs_groups"] = {k: list(obs.keys()) for k in ("actor", "critic")}
            train["num_steps_per_env"] = 8
            runner = OnPolicyRunner(env, train, log_dir=str(tmp_path), device=env.device)
            raw.agent_episode_length_buf["ant_3"][0] = raw.robot_max_episode_length["ant_3"] - 1
            raw.agent_episode_length_buf["ant_4"][0] = 0
            before_step = raw._sim_step_counter
            with torch.inference_mode():
                actions = runner.alg.act(obs)
                next_obs, reward, done, extra = env.step(actions)
                assert raw._sim_step_counter - before_step == cfg.decimation
                assert done[0] and not done[4]
                assert extra["terminal_timeout"][0]
                expected = (
                    reward
                    + runner.alg.gamma
                    * runner.alg.critic(extra["terminal_graph"]).squeeze(-1)
                    * extra["terminal_timeout"]
                )
                runner.alg.process_env_step(next_obs, reward, done, extra)
                torch.testing.assert_close(runner.alg.storage.rewards[0, :, 0], expected)
                runner.alg.storage.clear()
            before = runner.alg.actor.decider[0].weight.detach().clone()
            runner.learn(2)
            assert not torch.equal(before, runner.alg.actor.decider[0].weight)
            path = str(tmp_path / "roundtrip.pt")
            runner.save(path)
            runner.alg.eval_mode()
            with torch.inference_mode():
                expected = runner.alg.actor(env.get_observations()).clone()
                runner.alg.actor.decider[0].weight.add_(1)
            runner.load(path)
            with torch.inference_mode():
                torch.testing.assert_close(runner.alg.actor(env.get_observations()), expected)
            # Target adapters start at the source behavior and alone change under real PPO.
            # Reuse this scene: freezing is independent of topology, already exercised above.
            train["actor"]["adapter_dim"] = 8
            adapted = OnPolicyRunner(env, train, log_dir=str(tmp_path / "adapt"), device=env.device)
            source_state = runner.alg.actor.state_dict()
            missing, unexpected = adapted.alg.actor.load_state_dict(source_state, strict=False)
            assert missing and all(k.startswith("adapters.") for k in missing) and not unexpected
            adapted.alg.eval_mode()
            with torch.inference_mode():
                torch.testing.assert_close(adapted.alg.actor(env.get_observations()), expected, rtol=0, atol=0)
            frozen = {k: v.clone() for k, v in source_state.items()}
            adapter_before = {k: v.clone() for k, v in adapted.alg.actor.adapters.state_dict().items()}
            critic_before = adapted.alg.critic.decider[0].weight.clone()
            adapted.learn(2)
            for key, value in frozen.items():
                torch.testing.assert_close(adapted.alg.actor.state_dict()[key], value, rtol=0, atol=0)
            assert any(
                not torch.equal(v, adapted.alg.actor.adapters.state_dict()[k]) for k, v in adapter_before.items()
            )
            assert not torch.equal(critic_before, adapted.alg.critic.decider[0].weight)
            adapted.alg.eval_mode()
            with torch.inference_mode():
                expected = adapted.alg.actor(env.get_observations()).clone()
            adapted.save(str(tmp_path / "adapters.pt"))
            with torch.no_grad():
                adapted.alg.actor.adapters["actions"][-1].bias.add_(1)
            adapted.load(str(tmp_path / "adapters.pt"))
            with torch.inference_mode():
                torch.testing.assert_close(adapted.alg.actor(env.get_observations()), expected, rtol=0, atol=0)
        finally:
            raw.close()
