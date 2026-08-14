# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton DVI solver."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class DVISolverCfg(NewtonSolverCfg):
    """Configuration for the DVI (Differential Variational Inequality) solver.

    This solver uses a maximal coordinate formulation with cone complementarity
    for frictional contact dynamics. Joint constraints and contacts are solved
    using iterative methods (Jacobi, APGD, Gauss-Seidel, or sparse LDL).

    References:
        - A. Tasora, M. Anitescu. "A matrix-free cone complementarity approach for
          solving large-scale, nonsmooth, rigid body dynamics." CMAME, 2011.
    """

    class_type: type[NewtonManager] | str = "{DIR}.dvi_manager:NewtonDVIManager"
    """Manager class for the DVI solver."""

    solver_type: str = "dvi"
    """Solver type metadata."""

    # NOTE: DVI solver parameter blocks are ordered to match the solve order:
    # joint limits (unilateral) -> joints (bilateral) -> contacts.

    # -- Joint limit constraint solver config (optional) --
    # When joint_limit_solver_type is set, joint limits switch from penalty
    # (spring-damper in actuation) to constraint-based enforcement
    # (unilateral λ≥0 solver before bilateral joints).

    joint_limit_solver_type: str | None = "sparse_jacobi"
    """Numerical solver type for joint limit constraints.
    When set (e.g. ``"sparse_jacobi"``), enables constraint-based joint limits.
    When None, joint limits use penalty-based spring-damper forces.
    """

    joint_limit_max_iterations: int = 10
    """Maximum iterations for joint limit constraint solver."""

    joint_limit_omega: float = 0.3
    """Relaxation parameter for joint limit solver."""

    joint_limit_relax: float = 0.9
    """SOR relaxation for joint limit solver."""

    joint_limit_reg: float = 1e-8
    """Regularization for joint limit solver."""

    joint_limit_alpha: float = 0.0
    """Baumgarte damping for joint limit constraints."""

    joint_limit_recovery_speed: float = 10.0
    """Max Baumgarte recovery speed for joint limits (rad/s)."""

    # -- Joint solver config --
    joint_solver_type: str = "sparse_ldl"
    """Numerical solver type for joint constraints. One of:
    sparse_ldl, sparse_jacobi, sparse_apgd, sparse_block_gs, dense_direct,
    dense_jacobi, dense_gauss_seidel, dense_apgd.

    Note: ``sparse_ldl`` is a direct factorization; iterative controls such as
    ``joint_omega``/``joint_relax`` are unused for it (use
    ``joint_iterative_refinement_steps`` to control LDL refinement).
    """

    joint_omega: float = 0.3
    """Relaxation parameter for joint solver."""

    joint_relax: float = 0.8
    """SOR relaxation for joint solver."""

    joint_reg: float = 1e-6
    """Regularization for joint solver."""

    joint_alpha: float = 0.0
    """Baumgarte damping for joint constraints.  correction = phi / (dt + alpha).
    0 = full correction each step, large value (1e6) = effectively disabled."""

    joint_recovery_speed: float = 100000.0
    """Max Baumgarte recovery speed for joints (rad/s or m/s).
    Very large = effectively unlimited."""

    use_meca: bool = True
    """Use MECA fill-reducing pivot ordering for sparse LDL joint solves."""

    use_rcm: bool = False
    """Use block-level RCM ordering for sparse LDL solves.

    RCM takes precedence over MECA when both are enabled.
    """

    # -- Contact solver config --
    contact_solver_type: str = "sparse_jacobi"
    """Numerical solver type for contact forces."""

    contact_max_iterations: int = 20
    """Maximum iterations for contact solver."""

    coupling_iterations: int = 1
    """Number of block Gauss-Seidel sweeps coupling joint limits, bilateral
    joints, and contacts per physics step."""

    cache_factorization: bool = True
    """Reuse the joint direct-solver factorization across coupling sweeps."""

    post_stabilize_joints: bool = False
    """After the configured coupling sweeps, run one additional bilateral-joint
    solve only using the latest joint-limit and contact impulses. This avoids
    another contact solve while correcting the bilateral-joint RHS."""

    contact_omega: float = 0.3
    """Relaxation parameter for contact solver."""

    contact_relax: float = 0.9
    """SOR relaxation for contact solver."""

    contact_reg: float = 1e-4
    """Regularization for contact solver."""

    contact_alpha: float = 0.0
    """Baumgarte damping for contact constraints.  correction = phi / (dt + alpha).
    0 = full correction each step."""

    contact_compliance: float = 0.0
    """Physical contact compliance passed to NumericalSolverConfig.
    0.0 gives rigid contacts; positive values add E = c/(dt*(dt+alpha))
    to the contact Delassus operator diagonal."""

    contact_recovery_speed: float = 1.0
    """Max Baumgarte recovery speed for contacts (m/s)."""


    contact_tolerance: float | None = None
    """Early-exit convergence tolerance for the contact numerical solver.

    For iterative solvers (APGD/ASPG/Jacobi/GS) the solve latches converged and
    stops early once the projected-gradient KKT residual ``||gamma - P_K(gamma - g)||``
    (rho=1) drops below this value. Looser (larger) => exits sooner (faster, but
    risks returning an under-solved impulse that can drift/blow up under
    recovery+momentum). Tighter (smaller) => more iterations, better contacts.
    ``None`` keeps the Newton solver default (1e-8). Only affects APGD/iterative
    contacts; ignored by direct solvers.
    """

    contact_aspg_seed_alpha_max: bool = False
    """Seed the initial spectral step at alpha_max (Tasora 2013 Algorithm 4:
    alpha_0 = alpha_max) instead of alpha_0 = 1/L_0 from the omega/Rayleigh seed.
    Only affects ``contact_solver_type="sparse_aspg"``; ignored otherwise.

    The omega=0 (Rayleigh L_0) path seeds alpha_0 = 1/lambda_max(N) ~ alpha_min on
    stiff Delassus operators, so P-SPG-FB starts with a near-minimal step and
    (with few iterations) never recovers -> reward collapse. Algorithm 4 instead
    starts aggressive (alpha_max) and lets the GLL line search + descent guard
    pull the step down. When True, the omega/L_0 machinery is bypassed for the
    alpha_0 seed only.
    """

    contact_aspg_no_momentum: bool = False
    """Disable Nesterov momentum in SPARSE_ASPG contacts (pure Spectral Projected
    Gradient: y = gamma_new each iteration, beta=0). Only affects
    ``contact_solver_type="sparse_aspg"``; ignored otherwise.

    ASPG's BB step size is only meaningful when evaluated on a *consistent*
    secant pair (s = x_k - x_{k-1}, dg = g(x_k) - g(x_{k-1}) at the SAME
    iterates). With Nesterov momentum on, the gradient is evaluated at the
    extrapolated look-ahead point while s is measured between projected
    iterates, so the pair is inconsistent and the BB step becomes unreliable
    -> compounding divergence on stiff/indeterminate contact problems, since
    (unlike APGD) ASPG has no per-iteration backtracking safety net to catch
    a bad step. Setting this True trades acceleration for stability.
    """

    deterministic: bool = True
    """Enable deterministic fixed-point accumulation for iterative DVI solvers.

    This is forwarded explicitly to Newton's ``NumericalSolverConfig`` for
    joint-limit, joint, and contact numerical solvers.  Keeping it explicit at
    the Isaac Lab boundary prevents a future low-level default change from
    silently making RL experiments non-reproducible.
    """

    contact_aspg_alpha_max_rel: float = 2.0
    """Operator-relative upper cap for the BB spectral step in SPARSE_ASPG:
    ``alpha <= aspg_alpha_max_rel / L``, where L is the Rayleigh Lipschitz
    estimate (1/L is APGD's stable step size). Only affects
    ``contact_solver_type="sparse_aspg"``; ignored otherwise.

    Without a line search / backtracking safety net (unlike APGD), an
    unclamped BB step can run several multiples of the stable step and
    overshoot on well-conditioned contact operators. Default 2.0 allows up to
    2x the stable step. Lowering toward 1.0 caps ASPG at (approximately) the
    same stable step APGD uses, trading BB acceleration for a bound closer to
    APGD's implicit step cap -- a direct attempt to curb mid/late-training
    reward drift/collapse. Set 0 to disable the cap entirely (unclamped BB).
    """

    # -- General solver params --
    angular_damping: float = 0.01
    """Angular velocity damping coefficient."""

    actuator_integration: str = "semi_implicit"
    """Actuator integration mode for PD control. One of:

    - ``"explicit"``: Forces evaluated at current state, no mass augmentation.
    - ``"semi_implicit"`` (default): Mass augmentation via augmented inertia.
    - ``"implicit"``: Same augmentation as semi_implicit, plus an additional
      velocity correction term for exact implicit treatment.
    """

    enable_actuation: bool = True
    """Whether to enable PD control actuation."""

    diagonal_precondition: bool = True
    """Enable diagonal (Jacobi) preconditioning for the LDL joint solver.

    Scales the Schur complement N by S = diag(1/sqrt(N_ii)) before LDL
    factorization. Improves numerical stability for systems with high mass
    ratios across fixed joints. Only effective with sparse_ldl joint solver.
    """

    precond_reg: float = 1e-4
    """Post-preconditioning regularization for the scaled LDL system.

    After diagonal scaling brings all N diagonal entries to ~1.0, this adds
    fresh regularization. Prevents near-zero pivots during LDL factorization.
    Only used when diagonal_precondition=True.
    """

    joint_iterative_refinement_steps: int = 0
    """Number of iterative refinement steps after joint LDL factorization.

    Each step computes residual r = b - N@x matrix-free, re-solves using
    existing L/D factors, and updates x += dx. Recovers ~3-4 digits of
    precision per step in float32. Only used by SparseLDL solver.
    Typical values: 0 (disabled), 1-2 (recommended for high mass-ratio systems).
    """

    armature_override: dict[str, float] | None = None
    """Optional per-joint armature overrides applied before solver construction.

    Maps joint name substrings to armature values.  When set, any joint whose
    label contains the key string will have its armature replaced with the
    corresponding value.  Example::

        armature_override={"shoulder": 0.05, "elbow": 0.05, "wrist": 0.05,
                           "hand": 0.05, "finger": 0.05}

    Useful for maximal-coordinate solvers that need higher armature on
    lightweight distal links (e.g. humanoid hands) than what the MJCF/URDF
    provides.
    """

    contact_block_precondition: bool = False
    """Use block-3x3 inverse preconditioner for contact Jacobi/GS solvers.

    When ``True`` (default), computes the full 3x3 diagonal block of
    J M^{-1} J^T per contact and inverts it, capturing cross-coupling
    between normal and tangential directions.

    When ``False``, uses a scalar trace-based approximation:
    D_eff = trace(diag_block) / 3. Cheaper but less stable for some envs.
    """

    contact_friction_projection: str = "tangential"
    """Friction cone projection mode for contact solver.

    - ``"cone"`` (default): Anitescu-Tasora minimum-norm cone projection.
      Classical CCP formulation. Inflates normal impulse by
      (1 + 2μ²) / (1 + μ²) when friction is saturated.
    - ``"tangential"``: Tangential-only clamp. Preserves normal component,
      only clamps tangential magnitude. Gives exact normal forces and
      correct Coulomb sliding velocity.
    """


