# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot manifests for independent, concurrent locomotion tasks."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.locomotion.ant.ant_common import TERRAIN_CFG
from isaaclab_tasks.core.locomotion.ant.ant_direct_env_cfg import AntEnvCfg
from isaaclab_tasks.core.locomotion.humanoid.humanoid_direct_env_cfg import HumanoidEnvCfg

from isaaclab_assets.robots.ant_variants import ant_variant_cfg


@configclass
class LocomotionRobotCfg:
    """One independent policy slot, repeated in every world.

    The source task supplies its articulation, observation/action contract, reward scales and
    reset distribution. Its own simulation and scene placement are not instantiated.
    """

    task: AntEnvCfg | HumanoidEnvCfg = AntEnvCfg()
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation relative to the source task's initial pose [m]."""

    def __post_init__(self):
        # Source tasks are recipes, not independently selectable physics simulations. Removing their
        # backend presets prevents the task resolver/browser from advertising unsupported backends.
        self.task.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())


@configclass
class MultiRobotSceneCfg(InteractiveSceneCfg):
    """Shared ground and lighting; the robot manifest supplies articulations and sensors."""

    terrain = TERRAIN_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class MultiAntEnvCfg(DirectMARLEnvCfg):
    """Two independently controlled Ants per world, using Newton MJWarp."""

    decimation = 2
    episode_length_s = 16.0
    independent_resets = True
    compute_final_obs = True
    state_space = 0
    # Resolved from the manifest before constructing the environment.
    possible_agents = []
    observation_spaces = {}
    action_spaces = {}
    sim = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=200, nconmax=100, cone="pyramidal", integrator="implicitfast"),
            num_substeps=1,
        ),
    )
    scene = MultiRobotSceneCfg(num_envs=1024, env_spacing=8.0)
    robots: dict[str, LocomotionRobotCfg] = {
        "ant_a": LocomotionRobotCfg(offset=(0.0, -2.0, 0.0)),
        "ant_b": LocomotionRobotCfg(offset=(0.0, 2.0, 0.0)),
    }


@configclass
class AntHumanoidEnvCfg(MultiAntEnvCfg):
    """Heterogeneous 8- and 21-action policies in the same physical world."""

    robots = {
        "ant": LocomotionRobotCfg(offset=(0.0, -2.0, 0.0)),
        "humanoid": LocomotionRobotCfg(task=HumanoidEnvCfg(), offset=(0.0, 2.0, 0.0)),
    }


def ant_morphology_recipe(num_legs: int) -> AntEnvCfg:
    """Use the stock locomotion objective with morphology-specific asset and sensor dimensions."""
    task = AntEnvCfg()
    # All morphology baselines start clear of the plane throughout the stock ankle reset range.
    task.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    if num_legs == 4:
        return task
    task.scene.robot = ant_variant_cfg(num_legs)
    task.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    task.feet_body_names = [f"radial_{index}_foot" for index in range(num_legs)]
    task.action_space = 2 * num_legs
    task.observation_space = 12 + 3 * task.action_space + 6 * num_legs
    return task


@configclass
class AntThreeFourEnvCfg(MultiAntEnvCfg):
    """Three-/four-legged source robots, with independent specialist policies."""

    robots = {
        "ant_3": LocomotionRobotCfg(task=ant_morphology_recipe(3), offset=(0.0, -2.0, 0.0)),
        "ant_4": LocomotionRobotCfg(task=ant_morphology_recipe(4), offset=(0.0, 2.0, 0.0)),
    }


@configclass
class AntFiveSpecialistEnvCfg(MultiAntEnvCfg):
    """Held-out five-legged specialist baseline; never included in source pretraining."""

    robots = {"ant_5": LocomotionRobotCfg(task=ant_morphology_recipe(5))}
