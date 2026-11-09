# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate frozen mechanical actors on Ant, Unitree Go2 and A1 with explicit effort control."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from isaaclab_newton.physics import NewtonManager

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_env import (
    MechanicalAntEnv,
    MechanicalAntEnvCfg,
    mechanical_ant_cfg,
)
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_observations import SCHEMA_VERSION
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_policy import MechanicalActor
from isaaclab_tasks.contrib.multi_robot_locomotion.mechanical_training import MechanicalVecEnv
from isaaclab_tasks.contrib.multi_robot_locomotion.multi_robot_env_cfg import LocomotionRobotCfg
from isaaclab_tasks.core.locomotion.ant.ant_direct_env_cfg import AntEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO2_CFG


class StandingMechanicalEnv(MechanicalAntEnv):
    """Check physical setup using nominal-posture PD feedback at each physics step."""

    def _apply_action(self):
        for name, robot in self.robots.items():
            effort = 25.0 * (robot.data.default_joint_pos.torch - robot.data.joint_pos.torch)
            effort -= 0.5 * robot.data.joint_vel.torch
            gain = self.joint_gears[name] * self.cfg.robots[name].task.action_scale
            self.actions[name].copy_(effort / gain)
            robot.actuators.target_command.set_effort_index(value=self.actions[name].clamp(-1, 1) * gain)


def transfer_cfg(robot: str, worlds: int, seed: int) -> MechanicalAntEnvCfg:
    """Preserve source observation units and define ideal-effort versions of the Unitree assets.

    Unitree USD geometry, masses, joints and nominal posture are retained. The stock DC motor
    controller is replaced by effort actuation with the source Ant's 0.1 N m s/rad damping.
    Descriptor v1 cannot encode Newton's pure-effort drive enum; the supported velocity-drive
    mode describes this damping explicitly. There is no position drive or torque-speed curve.
    This is not the stock hardware drive model.
    """
    if robot in ("ant_3", "ant_4", "dev_5"):
        cfg = mechanical_ant_cfg("development" if robot == "dev_5" else "canonical", worlds, seed)
        cfg.robots = {robot: cfg.robots[robot]}
        cfg.mechanics = {robot: cfg.mechanics[robot]}
        cfg.robots[robot].offset = (0.0, 0.0, 0.0)
        return cfg
    asset = copy.deepcopy({"go2": UNITREE_GO2_CFG, "a1": UNITREE_A1_CFG}[robot])
    asset.prim_path = "{ENV_REGEX_NS}/Robot"
    motor = asset.actuators["base_legs"]
    effort = motor.actuator_effort_limit
    velocity = motor.actuator_velocity_limit
    asset.actuators = {
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.1,
            joint_effort_limit=effort,
            joint_velocity_limit=velocity,
            friction=0.0,
        )
    }
    task = AntEnvCfg()
    task.scene.robot = asset
    task.action_space = 12
    task.feet_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    task.observation_space = 12 + 3 * task.action_space + 6 * len(task.feet_body_names)
    task.action_scale = 1.0
    task.joint_gears = dict(effort) if isinstance(effort, dict) else {".*": effort}
    task.termination_height = 0.5 * asset.init_state.pos[2]
    cfg = MechanicalAntEnvCfg()
    cfg.scene.num_envs, cfg.seed = worlds, seed
    cfg.robots = {robot: LocomotionRobotCfg(task=task)}
    cfg.mechanics = {}
    # Capacity only: larger assets can contact the floor with many shapes after a fall.
    cfg.sim.physics.solver_cfg.njmax = 2048
    cfg.sim.physics.solver_cfg.nconmax = 256
    # Keep the source 60 Hz control rate, with finer integration for Unitree dynamics.
    cfg.sim.dt = 1 / 480
    cfg.decimation = 8
    cfg.sim.render_interval = cfg.decimation
    return cfg


def rollout(env: MechanicalVecEnv, actor: MechanicalActor | None, baseline: str | None, steps: int) -> dict:
    """Measure completed episodes and total progress including partial episodes, with reset accounting."""
    name = env.agents[0]
    raw, robot = env.env, env.env.robots[name]
    obs = env.get_observations()
    totals = {"episodes": 0, "return": 0.0, "length_s": 0.0, "progress_m": 0.0, "survival": 0.0}
    saturated, count_actions, height_sum = 0.0, 0, 0.0
    trace = []
    with torch.inference_mode():
        for step in range(steps):
            if actor is not None:
                action = actor(obs)
            else:
                action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            if not torch.isfinite(action).all():
                raise RuntimeError("Non-finite transfer action.")
            saturated += float((action.abs() >= 1).sum())
            count_actions += action.numel()
            obs, _, dones, _ = env.step(action)
            if not torch.isfinite(obs["dynamic"]).all():
                raise RuntimeError(f"Non-finite transfer observation at step {step + 1}.")
            height_sum += float(obs["global"][:, 0].sum())
            count = int(dones.sum())
            if count:
                totals["episodes"] += count
                for key in ("return", "length_s", "progress_m", "survival"):
                    totals[key] += count * float(env.last_infos[name]["log"][f"Episode/{key}"])
            if step % 30 == 0:
                trace.append(
                    {
                        "step": step + 1,
                        "root_position_world0": robot.data.root_pos_w.torch[0].tolist(),
                        "reset_world0": bool(dones[0]),
                    }
                )
    unfinished_progress = float(raw.episode_progress[name].sum())
    all_horizon_speed = (totals["progress_m"] + unfinished_progress) / (steps * raw.step_dt * raw.num_envs)
    completed_speed = totals["progress_m"] / totals["length_s"] if totals["length_s"] else None
    episodes = totals["episodes"]
    for key in ("return", "length_s", "progress_m", "survival"):
        totals[key] = totals[key] / episodes if episodes else None
    totals.update(
        progress_speed_m_s=completed_speed,
        all_horizon_progress_speed_m_s=all_horizon_speed,
        saturation_fraction=saturated / count_actions if baseline != "stand" else None,
        mean_post_reset_root_height_m=height_sum / (steps * raw.num_envs),
        world0_trace=trace,
    )
    return totals


def main() -> None:
    """Run a new evaluation report; weights, normalizers and source checkpoints stay frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("ant_3", "ant_4", "dev_5", "go2", "a1"), required=True)
    control = parser.add_mutually_exclusive_group(required=True)
    control.add_argument("--checkpoint", type=Path)
    control.add_argument("--baseline", choices=("zero", "stand"))
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.steps) < 1 or args.report.exists():
        parser.error("Positive counts and a new report path are required.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "newton", "mujoco-warp", "warp-lang", "rsl-rl-lib")
    }
    if saved and (saved["infos"]["schema"] != SCHEMA_VERSION or saved["infos"]["versions"] != versions):
        parser.error("Checkpoint schema or dependency versions differ; establish a new reference first.")
    cfg = transfer_cfg(args.robot, args.worlds, args.seed)
    task = cfg.robots[args.robot].task
    metadata = {
        "robot": args.robot,
        "seed": args.seed,
        "worlds": args.worlds,
        "steps": args.steps,
        "schema": SCHEMA_VERSION,
        "versions": versions,
        "target_training": False,
        "checkpoint": str(args.checkpoint.resolve()) if saved else None,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if saved else None,
        "training_agents": saved["infos"]["training_agents"] if saved else [],
        "imitation_steps": saved["infos"].get("imitation_steps") if saved else None,
        "routing": saved["infos"]["routing"] if saved else args.baseline,
        "standing_control": {"kp": 25.0, "kd": 0.5, "feedback_dt": cfg.sim.dt} if args.baseline == "stand" else None,
        "actuation": "ideal effort plus 0.1 N m s/rad damping; no position drive or motor torque-speed curve"
        if args.robot in ("a1", "go2")
        else "source Ant actuation",
        "asset": task.scene.robot.spawn.usd_path,
        "action_scale": task.action_scale,
        "joint_gears": task.joint_gears,
        "termination_height": task.termination_height,
        "reset_position_noise_rad": task.initial_joint_pos_range,
        "reset_velocity_noise_rad_s": task.initial_joint_vel_range,
        "step_dt": cfg.sim.dt * cfg.decimation,
        "physics_config": to_dict(cfg.sim.physics),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_config": to_dict(task),
    }
    torch.manual_seed(args.seed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg):
        raw = None
        try:
            raw = (StandingMechanicalEnv if args.baseline == "stand" else MechanicalAntEnv)(cfg)
            model = NewtonManager.get_model()
            metadata["shape_types"] = sorted(set(model.shape_type.numpy().tolist()))
            metadata["body_names"] = raw.robots[args.robot].body_names
            metadata["joint_names"] = raw.robots[args.robot].joint_names
            env = MechanicalVecEnv(raw)
            graph = env.graphs[args.robot]
            descriptors = {key: getattr(graph, key).cpu() for key in ("static", "edges", "action_nodes", "context")}
            fingerprint = hashlib.sha256()
            for value in descriptors.values():
                fingerprint.update(value.numpy().tobytes())
            metadata.update(descriptor_sha256=fingerprint.hexdigest(), tokens=graph.num_nodes, actions=env.num_actions)
            torch.save(descriptors, args.report.with_suffix(".descriptors.pt"))
            actor = None
            if saved:
                actor = MechanicalActor(
                    env.get_observations(), {}, "actor", env.num_actions, routing=saved["infos"]["routing"]
                ).to(env.device)
                actor.load_state_dict(saved["actor_state_dict"], strict=True)
                actor.eval()
                original = {key: value.clone() for key, value in actor.state_dict().items()}
            metadata["metrics"] = {args.robot: rollout(env, actor, args.baseline, args.steps)}
            if actor is not None:
                if not all(torch.equal(value, original[key]) for key, value in actor.state_dict().items()):
                    raise RuntimeError("Evaluation changed the actor or normalization buffers.")
                metadata["actor_and_normalization_unchanged"] = True
            metadata["status"] = "complete"
        except Exception as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            args.report.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
            if raw is not None:
                raw.close()
    print(json.dumps({key: value for key, value in metadata.items() if key != "task_config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
