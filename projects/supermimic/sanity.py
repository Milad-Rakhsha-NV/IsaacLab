# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent physical checks required before model-based training."""

import argparse
import json

import torch

from .motions import MotionLibrary


@torch.no_grad()
def check_environment(env):
    ids = torch.arange(env.num_envs, device=env.device)
    clips = ids % len(env.library.names)
    frames = torch.minimum(torch.full_like(ids, 10), env.library.lengths[clips] - 2)
    env.reset(ids, clips, frames, noise=False)
    expected = env.library.reference(clips, frames)
    state = env.read()
    errors = {"reset_max_error": float((state - expected).abs().max())}
    predicted = env.task.kinematics(state)
    observed = (
        env.robot.data.body_link_pos_w.torch[:, env.body_ids] - env.scene.env_origins[:, None],
        env.robot.data.body_link_quat_w.torch[:, env.body_ids],
        env.robot.data.body_link_lin_vel_w.torch[:, env.body_ids],
        env.robot.data.body_link_ang_vel_w.torch[:, env.body_ids],
    )
    from .math_utils import conjugate, log_rotation, multiply

    errors["fk_position_max_m"] = float((predicted[0] - observed[0]).abs().max())
    errors["fk_orientation_max_rad"] = float(log_rotation(multiply(predicted[1], conjugate(observed[1]))).abs().max())
    errors["fk_linear_velocity_max_mps"] = float((predicted[2] - observed[2]).abs().max())
    errors["fk_angular_velocity_max_radps"] = float((predicted[3] - observed[3]).abs().max())
    for key, value in errors.items():
        if not value < 0.002:
            raise AssertionError(f"Physical contract failed: {key}={value}")
    neutral = state.clone()
    neutral[:, 13:42] = env.default
    reward = env.task.reward(neutral, neutral, env.previous, env.previous)
    if not torch.allclose(reward, torch.full_like(reward, 0.1), atol=1e-5):
        raise AssertionError(f"Perfect tracking should yield 0.1 reward: {reward}")
    # Exercise reset/step ordering, including returned transitions at a known failure.
    rewards = []
    for _ in range(100):
        action = ((env.library.reference(env.clip, env.frame + 1)[:, 13:42] - env.default) / env.scale).clamp(-10, 10)
        transition = env.step(action, noise=False)
        if not torch.isfinite(transition["reward"]).all():
            raise AssertionError("Nonfinite simulator rewards")
        rewards.append(transition["reward"].mean())
        env.reset((transition["failure"] | transition["timeout"]).nonzero().flatten())
    errors["reference_pd_mean_reward"] = float(torch.stack(rewards).mean())
    errors["num_modeled_joints"] = len(env.robot.joint_names)
    # The final transition belongs to the old episode and must survive a later reset.
    env.reset(ids, clips, env.library.lengths[clips] - 2, noise=False)
    transition = env.step(torch.zeros_like(env.previous), noise=False)
    if not transition["timeout"].all() or not (env.frame == env.library.lengths[clips] - 1).all():
        raise AssertionError("Motion-end transitions must be exposed before explicit reset")
    saved_successor = transition["successor"].clone()
    env.reset(ids, clips, torch.zeros_like(ids), noise=False)
    torch.testing.assert_close(transition["successor"], saved_successor, rtol=0, atol=0)
    errors["terminal_successor_survives_reset"] = True
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from isaaclab.app import launch_simulation

    from .environment import TrackingEnv, simulation_cfg

    library = MotionLibrary(args.motions, args.device)
    cfg = simulation_cfg(args.device)
    with launch_simulation(cfg, {"headless": True, "visualizer": "none"}):
        env = TrackingEnv(cfg, library, args.num_envs)
        try:
            print("SANITY_RESULT", json.dumps(check_environment(env)), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
