# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant mechanics pilot with body state and a fixed, documented source family."""

from dataclasses import dataclass

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

from .multi_robot_env import MultiRobotLocomotionEnv
from .multi_robot_env_cfg import LocomotionRobotCfg, MultiAntEnvCfg, ant_morphology_recipe


@dataclass(frozen=True)
class AntMechanics:
    """Multipliers relative to stock mechanics; mass changes preserve inertia consistency."""

    legs: int
    size: float = 1.0
    torso_density: float = 1.0
    limb_density: float = 1.0
    motor: float = 1.0
    damping: float = 1.0


# A small factorial pilot, not a claim of broad robot diversity. Five legs remain development-only.
ANT_FAMILY = {
    "ant_3": AntMechanics(3),
    "ant_4": AntMechanics(4),
    "ant_3_small": AntMechanics(3, size=0.9, motor=0.8),
    "ant_4_large": AntMechanics(4, size=1.1, motor=1.2),
    "ant_3_payload": AntMechanics(3, torso_density=1.5),
    "ant_4_payload": AntMechanics(4, torso_density=1.5),
    "ant_3_heavy_limbs": AntMechanics(3, limb_density=1.5, damping=1.5),
    "ant_4_light_limbs": AntMechanics(4, limb_density=0.75, damping=0.75),
    "dev_5": AntMechanics(5),
    "test_5_small_payload": AntMechanics(5, size=0.95, torso_density=1.25, motor=0.9),
    "test_5_large_limbs": AntMechanics(5, size=1.05, limb_density=1.25, damping=1.25),
}
ANT_SPLITS = {
    "canonical": ("ant_3", "ant_4"),
    "diverse": tuple(name for name in ANT_FAMILY if name.startswith("ant_")),
    "development": ("dev_5",),
    "test": ("test_5_small_payload", "test_5_large_limbs"),
}


@configclass
class MechanicalAntEnvCfg(MultiAntEnvCfg):
    """Opt-in richer observations and mechanics; the original graph task is unchanged."""

    mechanics: dict[str, AntMechanics] = {}


def mechanical_ant_cfg(split: str = "canonical", worlds: int = 512, seed: int = 42) -> MechanicalAntEnvCfg:
    """Create fixed prototypes, replicated across worlds without reset-time physical randomization."""
    cfg = MechanicalAntEnvCfg()
    cfg.scene.num_envs, cfg.seed = worlds, seed
    cfg.robots = {}
    cfg.mechanics = {name: ANT_FAMILY[name] for name in ANT_SPLITS[split]}
    for index, (name, mechanics) in enumerate(cfg.mechanics.items()):
        task = ant_morphology_recipe(mechanics.legs)
        task.scene.robot.spawn.scale = (mechanics.size,) * 3
        task.scene.robot.init_state.pos = (0.0, 0.0, 0.6 * mechanics.size)
        task.termination_height *= mechanics.size
        task.joint_gears = {key: value * mechanics.motor for key, value in task.joint_gears.items()}
        actuator = task.scene.robot.actuators["body"]
        actuator.joint_effort_limit *= mechanics.motor
        actuator.damping *= mechanics.damping
        cfg.robots[name] = LocomotionRobotCfg(task=task, offset=(0.0, 3.0 * index, 0.0))
    cfg.scene.env_spacing = max(8.0, 3.0 * len(cfg.robots))
    return cfg


class MechanicalAntEnv(MultiRobotLocomotionEnv):
    """Capture body poses/velocities before reset, including the terminal observation.

    Appended data are projected gravity (3), then link position (3), quaternion wxyz (4),
    COM linear velocity (3), and angular velocity (3) for every body, all in the root link
    frame. Relative positions exclude world translation. Velocities are absolute velocities
    expressed in that frame, not derivatives of relative positions. SI units are retained.
    """

    def _configure_env_spaces(self):
        for agent in self.cfg.possible_agents:
            self.cfg.observation_spaces[agent] += 3 + 13 * self.scene[agent].num_bodies
        super()._configure_env_spaces()

    def _exclude_robot_contacts(self, builder, world, position, orientation):
        super()._exclude_robot_contacts(builder, world, position, orientation)
        for body in range(builder.body_count - 1, -1, -1):
            if builder.body_world[body] != world:
                break
            label = builder.body_label[body]
            for agent, mechanics in self.cfg.mechanics.items():
                if f"/{agent}/" in label:
                    factor = mechanics.torso_density if label.endswith("/torso") else mechanics.limb_density
                    builder.body_mass[body] *= factor
                    builder.body_inertia[body] *= factor
                    break

    def _get_observations(self):
        flat = super()._get_observations()
        for agent, robot in self.robots.items():
            data = robot.data
            root = data.root_link_quat_w.torch[:, None].expand(-1, robot.num_bodies, -1)
            position = quat_apply_inverse(root, data.body_link_pos_w.torch - data.root_link_pos_w.torch[:, None])
            rotation = quat_mul(quat_conjugate(root), data.body_link_quat_w.torch)
            # Choose one quaternion hemisphere to remove the q/-q representation ambiguity.
            rotation = torch.where(rotation[..., :1] < 0, -rotation, rotation)
            linear = quat_apply_inverse(root, data.body_com_lin_vel_w.torch)
            angular = quat_apply_inverse(root, data.body_link_ang_vel_w.torch)
            bodies = torch.cat((position, rotation, linear, angular), dim=-1)
            flat[agent] = torch.cat((flat[agent], data.projected_gravity_b.torch, bodies.flatten(1)), dim=-1)
        return flat
