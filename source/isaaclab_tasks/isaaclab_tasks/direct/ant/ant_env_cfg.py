# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import DVISolverCfg, KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.ant import ANT_CFG


@configclass
class AntPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=45,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            use_collision_detector=False,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.05,
            padmm_eta=1e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_dvi: NewtonCfg = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_jacobi",
            contact_max_iterations=20,
            contact_omega=0.3,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            angular_damping=0.01,
            joint_limit_solver_type="sparse_jacobi",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    # APGD contacts variant: same as newton_dvi but Nesterov-accelerated projected
    # gradient for contacts. Proper backtracking line search: LARGE backtrack cap
    # (20) + graph-safe early exit (NEED_BT latch) so the search runs only as long
    # as the descent condition is violated, terminating early once satisfied.
    newton_dvi_apgd: NewtonCfg = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_apgd",
            # Fewer contact iterations than Jacobi: APGD is Nesterov-accelerated
            # with backtracking line search and converges far faster per iter on
            # the resting-contact LCPs seen during Ant training (see box-stack
            # convergence sweep: APGD hits ~1e-3 by ~20 iters vs Jacobi still at
            # ~1e-1). 10 iters is plenty here.
            contact_max_iterations=10,
            contact_tolerance=1e-4,  # SWEEP KNOB — vary per run
            # omega is INERT for APGD (step size comes from Nesterov + backtracking
            # line search, not omega). Set 0.0 so L0 is the Rayleigh-quotient seed
            # (Chrono ChSolverAPGDREF), never an omega override.
            contact_omega=0.0,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            angular_damping=0.01,
            joint_limit_solver_type="sparse_jacobi",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    # ASPG contacts variant: identical to newton_dvi_apgd except the contact solver
    # is the Barzilai-Borwein spectral accelerated projected gradient (sparse_aspg).
    # All other solver params held identical so FPS/results differences are
    # attributable solely to the contact solver.
    newton_dvi_aspg: NewtonCfg = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_aspg",
            contact_max_iterations=20,
            contact_tolerance=1e-4,
            # REVERTED to fixed omega=0.3 seed. Tested omega=0.0 (Rayleigh-quotient
            # L0, same formula/kernels as APGD -- verified identical code, not a
            # reimplementation) both WITH momentum (peak 915) and WITHOUT momentum
            # (aspg_no_momentum=True, peak 330) -- both far worse than omega=0.3's
            # peak 10,868. So the Rayleigh L0 value itself is a poor operator-norm
            # estimate for this contact system; the issue is NOT Nesterov momentum
            # vs BB-secant-pair inconsistency. APGD tolerates a bad L0 via its
            # per-iteration L*=0.9 backtracking decay; ASPG has no equivalent
            # correction once seeded, so a bad L0 here is simply unrecoverable
            # regardless of momentum. Fixed omega=0.3 remains the only config that
            # actually trains (peak 10,868, final 8,372); its known late-run drift
            # is the real remaining problem, not L0 init.
            # SWEEP: omega sweep (baseline settings, NO alpha_max_rel cap i.e.
            # default 2.0) to isolate why the fixed L0 seed value itself matters so
            # much. Known points so far (all baseline, uncapped):
            #   omega=0.3 -> peak 10,868 / final 8,372  (works, but late-drifts)
            #   omega=0.5 -> peak ~11,190 / final ~11,169 (works, STABLE, no drift!)
            #   omega=0.0 (Rayleigh) -> peak 622 / final 62  (collapses)
            # Testing omega=1.0 next to see whether reward keeps improving/staying
            # stable as omega increases further, or whether it eventually degrades.
            # TEST (b): seed alpha_0 = alpha_max per Tasora Algorithm 4 Init line,
            # bypassing the omega/L0 graft entirely. Diagnosis: omega=0 seeds
            # alpha_0 = 1/lambda_max(N) ~ alpha_min (timid) -> collapse; omega>0
            # seeds alpha_0 = 1/omega (aggressive) -> trains. Algorithm 4 starts
            # at alpha_max and lets GLL+descent-guard pull it down. omega value is
            # now irrelevant (seed bypassed) but left at a benign nonzero to skip
            # the Rayleigh L0 launch path.
            contact_omega=0.5,
            contact_aspg_seed_alpha_max=True,
            contact_aspg_no_momentum=False,
            contact_aspg_alpha_max_rel=2.0,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            angular_damping=0.01,
            joint_limit_solver_type="sparse_jacobi",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )
    # P-SPG-FB contacts variant: faithful Tasora Algorithm 4 preconditioned
    # spectral projected gradient (Fischer-Burmeister), sparse_pspg. Diagonal
    # preconditioner + GLL nonmonotone line search + descent guard make it the
    # fastest-converging contact solver in the box-stack sweep (near-solution in
    # ~1 preconditioned step on resting-contact LCPs), so it runs with the
    # FEWEST contact iterations. All other solver params held identical to the
    # APGD/Jacobi variants so FPS/results differences are attributable solely to
    # the contact solver.
    newton_dvi_pspg: NewtonCfg = NewtonCfg(
        solver_cfg=DVISolverCfg(
            joint_solver_type="sparse_ldl",
            joint_alpha=0.0,
            joint_recovery_speed=100000.0,
            contact_solver_type="sparse_pspg",
            contact_max_iterations=8,
            contact_tolerance=1e-4,
            # omega is INERT for PSPG (alpha0 = alpha_max seed, spectral BB step,
            # no omega override). Left benign nonzero for parity with siblings.
            contact_omega=0.5,
            contact_alpha=0.0,
            contact_recovery_speed=1.0,
            angular_damping=0.01,
            joint_limit_solver_type="sparse_jacobi",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=665536),
    )


@configclass
class AntEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 36
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=AntPhysicsCfg())
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.31

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1
