# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure five-legged Ant motor motion, work and actual ground load; optionally disable one leg.

The contact sensor is diagnostic only and is not added to policy observations. The steady-window
statistics use steps 120 through 899; returns/survival use completed episodes over 1200 steps.
Disabling two action coordinates is a closed-loop intervention, not proof of isolated propulsive work.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.sensors import ContactSensorCfg

from isaaclab_rl.entrypoints.common import set_hydra_args
from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.rsl_rl.independent import RslRlIndependentVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--disable",
    type=int,
    choices=range(-1, 5),
    default=-1,
    help="Radial branch to set to zero effort; -1 preserves the policy.",
)
p.add_argument("--seed", type=int, default=123)
a = p.parse_args()
set_hydra_args([])
cfg, acfg = resolve_task_config("IsaacContrib-Ant-Five-Specialist-Direct", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = 64
cfg.seed = a.seed
cfg.scene.foot_contacts = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/ant_5/.*_foot", track_friction_forces=True, history_length=2
)
with launch_simulation(cfg):
    raw = gym.make("IsaacContrib-Ant-Five-Specialist-Direct", cfg=cfg)
    try:
        env = RslRlIndependentVecEnvWrapper(raw, clip_actions=acfg.clip_actions)
        runner = create_rsl_rl_runner(env, handle_deprecated_rsl_rl_cfg(acfg, check_rsl_rl_version()))
        runner.load(a.checkpoint)
        policy = runner.get_inference_policy(env.device)
        obs, _ = env.reset()
        robot = raw.unwrapped.robots["ant_5"]
        sensor = raw.unwrapped.scene["foot_contacts"]
        trace = {k: [] for k in ["q", "v", "act", "torque", "normal", "friction"]}
        totals = dict(episodes=0, return_=0.0, duration=0.0, progress=0.0, survival=0.0)
        with torch.inference_mode():
            for step in range(1200):
                act = policy(obs)
                if a.disable >= 0:
                    for suffix in ["leg", "foot"]:
                        act["ant_5"][:, robot.joint_names.index(f"radial_{a.disable}_{suffix}")] = 0
                obs, _, done, info = env.step(act)
                policy.reset(done)
                count = int(done["ant_5"].sum())
                if count:
                    totals["episodes"] += count
                    for dst, src in [
                        ("return_", "return"),
                        ("duration", "length_s"),
                        ("progress", "progress_m"),
                        ("survival", "survival"),
                    ]:
                        totals[dst] += count * float(info["ant_5"]["log"][f"Episode/{src}"])
                if 120 <= step < 900:
                    for k, tensor in [
                        ("q", robot.data.joint_pos.torch),
                        ("v", robot.data.joint_vel.torch),
                        ("act", act["ant_5"]),
                        ("torque", robot.data.applied_torque.torch),
                        ("normal", sensor.data.net_normal_forces_w.torch),
                        ("friction", sensor.data.net_friction_forces_w.torch),
                    ]:
                        trace[k].append(tensor.clone())
        trace = {k: torch.stack(v).cpu() for k, v in trace.items()}
        for k in totals:
            if k != "episodes":
                totals[k] /= totals["episodes"]
        totals["speed"] = totals["progress"] / totals["duration"]
        result = dict(
            control_dt=raw.unwrapped.step_dt,
            trace_start_step=120,
            trace_stop_step=900,
            contact_threshold_N=0.1,
            checkpoint=a.checkpoint,
            disable=a.disable,
            seed=a.seed,
            worlds=64,
            joint_names=robot.joint_names,
            body_names=sensor.body_names,
            metrics=totals,
            joints={},
            feet={},
        )
        for j, name in enumerate(robot.joint_names):
            result["joints"][name] = {
                "world0_range": float(trace["q"][:, 0, j].max() - trace["q"][:, 0, j].min()),
                "mean_range": float((trace["q"][:, :, j].max(0).values - trace["q"][:, :, j].min(0).values).mean()),
                "mean_abs_power": float((trace["v"][:, :, j] * trace["torque"][:, :, j]).abs().mean()),
                "mean_abs_torque": float(trace["torque"][:, :, j].abs().mean()),
            }
        for j, name in enumerate(sensor.body_names):
            force = trace["normal"][:, :, j, :]
            friction = trace["friction"][:, :, j, :]
            result["feet"][name] = {
                "duty_gt_0_1N": float((force.norm(dim=-1) > 0.1).float().mean()),
                "mean_normal_z": float(force[:, :, 2].mean()),
                "mean_friction_x": float(friction[:, :, 0].mean()),
                "world0_duty": float((force[:, 0].norm(dim=-1) > 0.1).float().mean()),
                "world0_mean_normal_z": float(force[:, 0, 2].mean()),
                "normal_load_share": float(force[:, :, 2].sum() / trace["normal"][:, :, :, 2].sum()),
            }
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
        torch.save(trace, str(Path(a.output).with_suffix(".pt")))
        print(json.dumps(result, indent=2))
    finally:
        raw.close()
