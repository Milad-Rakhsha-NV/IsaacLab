# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Chrono solver."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class ChronoSolverCfg(NewtonSolverCfg):
    """Configuration for the Chrono DVI (Differential Variational Inequality) solver.

    This solver uses a maximal coordinate formulation with cone complementarity
    for frictional contact dynamics. Joint constraints and contacts are solved
    using iterative methods (Jacobi, APGD, Gauss-Seidel, or sparse LDL).

    References:
        - A. Tasora, M. Anitescu. "A matrix-free cone complementarity approach for
          solving large-scale, nonsmooth, rigid body dynamics." CMAME, 2011.
    """

    class_type: type[NewtonManager] | str = "{DIR}.chrono_manager:NewtonChronoManager"
    """Manager class for the Chrono solver."""

    solver_type: str = "chrono"
    """Solver type metadata."""

    # -- Joint solver config --
    joint_solver_type: str = "sparse_ldl"
    """Numerical solver type for joint constraints. One of:
    sparse_ldl, sparse_jacobi, sparse_apgd, sparse_block_gs, dense_direct,
    dense_jacobi, dense_gauss_seidel, dense_apgd.
    """

    joint_max_iterations: int = 50
    """Maximum iterations for joint constraint solver."""

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

    joint_position_correction: bool = False
    """Enable position-level correction for joint drift."""

    # -- Contact solver config --
    contact_solver_type: str = "sparse_jacobi"
    """Numerical solver type for contact forces."""

    contact_max_iterations: int = 50
    """Maximum iterations for contact solver."""

    contact_omega: float = 0.3
    """Relaxation parameter for contact solver."""

    contact_relax: float = 0.9
    """SOR relaxation for contact solver."""

    contact_reg: float = 1e-4
    """Regularization for contact solver."""

    contact_alpha: float = 0.0
    """Baumgarte damping for contact constraints.  correction = phi / (dt + alpha).
    0 = full correction each step."""

    contact_recovery_speed: float = 1.0
    """Max Baumgarte recovery speed for contacts (m/s)."""

    contact_position_correction: bool = False
    """Enable position-level correction for contact penetration."""

    # -- General solver params --
    angular_damping: float = 0.01
    """Angular velocity damping coefficient."""

    use_implicit_pd: bool = True
    """Whether to use implicit PD (mass-matrix augmentation) for joint stiffness/damping.
    Allows larger timesteps for stiff springs.
    """

    joint_limit_ke_scale: float = 1.0
    """Scale factor for joint limit stiffness (ke) and damping (kd).
    USD/MJCF importers produce limit stiffness tuned for implicit constraint solvers
    (e.g. 10,000).  For penalty-based enforcement, lower values (0.01-0.1) keep limit
    forces proportional to actuator effort.  Default 1.0 = no scaling.
    Only used in penalty mode (when no ``joint_limit_*`` solver fields are set).
    """

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

    enable_gyroscopic: bool = True
    """Whether to include gyroscopic torque."""

    enable_contacts: bool = True
    """Whether to enable contact force solving."""

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

    contact_backtrack_iterations: int = 5
    """Number of backtracking line-search iterations per APGD outer iteration.

    Only used when ``contact_solver_type`` is ``"sparse_apgd"``.
    Each backtracking step performs a full Schur product, so this is the
    main cost multiplier for APGD.  Set to 1 to disable backtracking
    (pure Nesterov APGD with ``L *= 0.9`` decay).  Default 5.
    """

    joint_iterative_refinement_steps: int = 0
    """Number of iterative refinement steps after joint LDL factorization.

    Each step computes residual r = b - N@x matrix-free, re-solves using
    existing L/D factors, and updates x += dx. Recovers ~3-4 digits of
    precision per step in float32. Only used by SparseLDL solver.
    Typical values: 0 (disabled), 1-2 (recommended for high mass-ratio systems).
    """

    contact_block_precondition: bool = True
    """Use block-3x3 inverse preconditioner for contact Jacobi/GS solvers.

    When ``True`` (default), computes the full 3x3 diagonal block of
    J M^{-1} J^T per contact and inverts it, capturing cross-coupling
    between normal and tangential directions.

    When ``False``, uses a scalar trace-based approximation:
    D_eff = trace(diag_block) / 3. Cheaper but less stable for some envs.
    """

    contact_friction_projection: str = "cone"
    """Friction cone projection mode for contact solver.

    - ``"cone"`` (default): Anitescu-Tasora minimum-norm cone projection.
      Classical CCP formulation. Inflates normal impulse by
      (1 + 2μ²) / (1 + μ²) when friction is saturated.
    - ``"tangential"``: Tangential-only clamp. Preserves normal component,
      only clamps tangential magnitude. Gives exact normal forces and
      correct Coulomb sliding velocity.
    """


