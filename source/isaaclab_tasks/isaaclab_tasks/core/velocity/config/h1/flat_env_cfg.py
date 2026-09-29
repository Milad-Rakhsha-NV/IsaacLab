# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab_newton.physics import DVISolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import H1RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    """Physics presets for the flat H1 velocity task.

    The DVI presets mirror the configuration used for the technical-report H1 runs
    (`Isaac-Velocity-Flat-H1`, 1000 iterations, contact iterations 10). Three solver
    fields from that era were removed upstream and are intentionally not carried over:
    ``joint_position_correction`` / ``contact_position_correction`` (position correction is
    no longer a solver-level toggle) and ``joint_limit_ke_scale`` (joint-limit stiffness now
    comes from the joint-limit solver configuration).
    """

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=15,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_dvi = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_jacobi",
            # The technical-report H1 runs were swept at contact iterations 10; the paper
            # video used that setting, so it is persisted here rather than passed on the CLI.
            contact_max_iterations=10,
            contact_alpha=0.0,
            contact_recovery_speed=10000.0,
            angular_damping=0.0,
            actuator_integration="semi_implicit",
            joint_limit_solver_type="sparse_jacobi",
            joint_limit_recovery_speed=1.0,
            joint_iterative_refinement_steps=1,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        collapse_fixed_joints=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=2**21),
    )
    newton_dvi_apgd = None
    newton_dvi_pspg = None
    physx = default

    def __post_init__(self):
        _apgd = copy.deepcopy(self.newton_dvi)
        _apgd.solver_cfg.contact_solver_type = "sparse_apgd"
        _apgd.solver_cfg.contact_max_iterations = 20
        _apgd.solver_cfg.contact_tolerance = 1e-4
        self.newton_dvi_apgd = _apgd
        _pspg = copy.deepcopy(self.newton_dvi)
        _pspg.solver_cfg.contact_solver_type = "sparse_pspg"
        self.newton_dvi_pspg = _pspg


@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        super().__post_init__()

        # physics
        self.sim.physics.default = self.sim.physics.newton_mjwarp
        # Pin the H1 asset to the Isaac 6.0 release used for the paper results. The default
        # asset root now points at 6.1, whose h1_minimal.usd differs from the trained-era one.
        self.scene.robot.spawn.usd_path = (
            "https://omniverse-content-staging.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/H1/h1_minimal.usd"
        )
        # Lower spawn height slightly (default 1.05 starts slightly airborne)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.98)
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        # curriculum
        self.curriculum.terrain_levels = None
