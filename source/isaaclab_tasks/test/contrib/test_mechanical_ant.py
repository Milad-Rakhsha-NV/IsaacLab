# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical descriptor completeness, invariant routing, and real mechanics-conditioned PPO."""

import math

import newton
import torch
import torch.nn.functional as F
import warp as wp
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import MechanicalAntEnv, mechanical_ant_cfg
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import STATIC_SLICES, NewtonMechanicalGraph
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_policy import MechanicalActor
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv, mechanical_train_cfg


def test_newton_graph_contains_root_mass_full_inertia_and_action_order():
    """An offset pair of spheres has analytically known COM/inertia, including a nonzero product term."""
    builder = newton.ModelBuilder()
    root, limb, other_limb = builder.add_link(), builder.add_link(), builder.add_link()
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.7)
    for sign in (-1, 1):
        builder.add_shape_sphere(
            root, radius=0.1, xform=wp.transform((sign * 0.2, sign * 0.2, 0.0), wp.quat_identity()), cfg=shape_cfg
        )
    builder.add_shape_sphere(limb, radius=0.05, cfg=shape_cfg)
    builder.add_shape_sphere(other_limb, radius=0.05, cfg=shape_cfg)
    free = builder.add_joint_free(root)
    hinge = builder.add_joint_revolute(
        root, limb, axis=(0.0, 1.0, 0.0), limit_lower=-0.4, limit_upper=0.8, armature=0.03
    )
    other_hinge = builder.add_joint_revolute(root, other_limb, axis=(0.0, 0.0, 1.0))
    builder.add_articulation([free, hinge, other_hinge])
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    graph = NewtonMechanicalGraph(
        model, [root, limb, other_limb], [free, hinge, other_hinge], [other_hinge, hinge], [12.0, 7.0], "cpu"
    )
    mass = 2 * (4 / 3) * math.pi * 0.1**3 * 100
    torch.testing.assert_close(graph.static[0, STATIC_SLICES["body_mass"]], torch.tensor([mass]))
    torch.testing.assert_close(graph.static[0, STATIC_SLICES["body_com"]], torch.zeros(3))
    expected = mass * torch.tensor([[0.044, -0.04, 0], [-0.04, 0.044, 0], [0, 0, 0.084]])
    torch.testing.assert_close(graph.static[0, STATIC_SLICES["body_inertia"]].reshape(3, 3), expected)
    assert graph.static[0, STATIC_SLICES["root_body"]].item() == 1
    assert graph.num_joints == 2
    other_action, action = graph.action_nodes.tolist()
    torch.testing.assert_close(graph.static[action, STATIC_SLICES["joint_axis"]], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(graph.static[other_action, STATIC_SLICES["joint_axis"]], torch.tensor([0.0, 0.0, 1.0]))
    assert graph.static[action, STATIC_SLICES["action_effort_gain"]].item() == 7
    assert graph.static[other_action, STATIC_SLICES["action_effort_gain"]].item() == 12
    # Passive free-base coordinates exist, but cannot accidentally become policy outputs.
    assert graph.static[:, STATIC_SLICES["kind"]][:, 2].sum().item() == 8
    assert graph.static[:, STATIC_SLICES["actuated"]].sum().item() == 2


def test_mechanical_family_routing_terminal_state_and_ppo(tmp_path, monkeypatch):
    """One real scene owns scaling, body capture, padding/permutation, and trainable routing paths."""
    cfg = mechanical_ant_cfg("diverse", worlds=2)
    cfg.sim.device = test_devices(DeviceScope.DEFAULT_CUDA)[0]
    with launch_simulation(cfg):
        raw = MechanicalAntEnv(cfg)
        try:
            env = MechanicalVecEnv(raw)
            obs = env.get_observations()
            graphs = env.graphs
            stock, small, payload = (graphs[name] for name in ("ant_3", "ant_3_small", "ant_3_payload"))
            for field, exponent in (("body_mass", 3), ("body_inertia", 5)):
                torch.testing.assert_close(
                    small.static[:7, STATIC_SLICES[field]], stock.static[:7, STATIC_SLICES[field]] * 0.9**exponent
                )
                torch.testing.assert_close(
                    payload.static[0, STATIC_SLICES[field]], stock.static[0, STATIC_SLICES[field]] * 1.5
                )
                torch.testing.assert_close(
                    payload.static[1:7, STATIC_SLICES[field]], stock.static[1:7, STATIC_SLICES[field]]
                )
            for name, graph in graphs.items():
                assert graph.num_joints == raw.robots[name].num_joints
                assert graph.num_bodies == raw.robots[name].num_bodies
                assert torch.isfinite(graph.static).all()
            obs = obs[:4].clone()
            for routing in ("attention", "pooled", "message_passing"):
                if routing == "message_passing":

                    def reject_attention(*args, **kwargs):
                        raise AssertionError("The message-passing comparator must run without attention.")

                    monkeypatch.setattr(F, "scaled_dot_product_attention", reject_attention)
                actor = MechanicalActor(obs, {}, "actor", env.num_actions, routing=routing).to(env.device).eval()
                with torch.no_grad():
                    expected = actor(obs)
                    if routing == "message_passing":
                        disconnected = obs.clone()
                        disconnected["edges"].zero_()
                        assert not torch.allclose(actor(disconnected), expected, atol=1e-6, rtol=1e-5)
                    order = torch.randperm(env.node_width, device=env.device)
                    inverse = order.argsort()
                    permuted = obs.clone()
                    for key in ("static", "dynamic", "node_mask"):
                        permuted[key] = obs[key][:, order]
                    permuted["edges"] = obs["edges"][:, order][:, :, order]
                    permuted["action_nodes"] = inverse[obs["action_nodes"]]
                    torch.testing.assert_close(actor(permuted), expected, atol=2e-6, rtol=1e-5)
                    padded = obs.clone()
                    for key in ("static", "dynamic"):
                        padded[key] = F.pad(obs[key], (0, 0, 0, 5), value=10000)
                    padded["node_mask"] = F.pad(obs["node_mask"], (0, 5))
                    padded["edges"] = F.pad(obs["edges"], (0, 5, 0, 5), value=9)
                    padded["action_nodes"] = F.pad(obs["action_nodes"], (0, 2))
                    padded["action_mask"] = F.pad(obs["action_mask"], (0, 2))
                    torch.testing.assert_close(actor(padded)[:, : env.num_actions], expected, atol=2e-6, rtol=1e-5)
                    assert (actor(padded)[:, env.num_actions :] == 0).all()
                    actor(obs, stochastic_output=True)
                    log_prob, entropy = actor.get_output_log_prob(expected), actor.output_entropy
                    actor(padded, stochastic_output=True)
                    torch.testing.assert_close(
                        actor.get_output_log_prob(F.pad(expected, (0, 2), value=10000)), log_prob
                    )
                    torch.testing.assert_close(actor.output_entropy, entropy)
                train = mechanical_train_cfg(env, routing)
                train["num_steps_per_env"] = 4
                runner = OnPolicyRunner(env, train, log_dir=str(tmp_path / routing), device=env.device)
                if routing == "attention":
                    robot = raw.robots["ant_3"]
                    pose = robot.data.root_link_pose_w.torch.clone()
                    angle = torch.full((raw.num_envs,), 0.3, device=env.device)
                    pose[:, 3:] = quat_from_euler_xyz(torch.zeros_like(angle), angle, torch.zeros_like(angle))
                    robot.write_root_pose_to_sim(pose)
                    raw.agent_episode_length_buf["ant_3"][0] = raw.robot_max_episode_length["ant_3"] - 1
                    with torch.inference_mode():
                        action = runner.alg.act(env.get_observations())
                        live, reward, done, extra = env.step(action)
                        assert done[0] and extra["terminal_timeout"][0]
                        # Full pitch-sensitive gravity belongs to the pre-reset terminal state.
                        assert extra["terminal_graph"]["global"][0, 12].abs() > 0.2
                        assert live["global"][0, 12].abs() < 1e-5
                        expected_reward = (
                            reward
                            + runner.alg.gamma
                            * runner.alg.critic(extra["terminal_graph"]).squeeze(-1)
                            * extra["terminal_timeout"]
                        )
                        runner.alg.process_env_step(live, reward, done, extra)
                        torch.testing.assert_close(runner.alg.storage.rewards[0, :, 0], expected_reward)
                        runner.alg.storage.clear()
                before = {key: value.clone() for key, value in runner.alg.actor.named_parameters()}
                runner.learn(2)
                assert any(not torch.equal(before[key], value) for key, value in runner.alg.actor.named_parameters())
                assert all(torch.isfinite(value).all() for value in runner.alg.actor.parameters())
                runner.alg.eval_mode()
                with torch.inference_mode():
                    expected = runner.alg.actor(env.get_observations()).clone()
                path = str(tmp_path / routing / "roundtrip.pt")
                runner.save(path)
                with torch.no_grad():
                    runner.alg.actor.decoder[-1].bias.add_(1)
                runner.load(path)
                with torch.inference_mode():
                    torch.testing.assert_close(runner.alg.actor(env.get_observations()), expected)
        finally:
            raw.close()
