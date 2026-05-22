# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Chrono solver."""

from __future__ import annotations

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

    contact_recovery_speed: float = 1.0
    """Contact recovery speed (Baumgarte stabilization)."""

    # -- General solver params --
    angular_damping: float = 0.01
    """Angular velocity damping coefficient."""

    use_implicit_pd: bool = True
    """Whether to use implicit PD (mass-matrix augmentation) for joint stiffness/damping.
    Allows larger timesteps for stiff springs.
    """

    enable_gyroscopic: bool = True
    """Whether to include gyroscopic torque."""

    enable_contacts: bool = True
    """Whether to enable contact force solving."""

    enable_actuation: bool = True
    """Whether to enable PD control actuation."""

    max_velocity: float = 20.0
    """Maximum allowed body/joint velocity component. Values exceeding this are clamped.
    Prevents catastrophic divergence from deep penetration events.
    Set to 0 or negative to disable clamping.
    """
