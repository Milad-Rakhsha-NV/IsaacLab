# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental surface RFT composed with MJWarp articulation dynamics."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .mjwarp_solver_cfg import MJWarpSolverCfg


@configclass
class RFTSurfaceResolutionCfg:
    """Construction-time resolution for matching RFT collision surfaces.

    Patterns match complete finalized shape labels. These settings change the
    RFT integration surface while preserving the source collision geometry.
    """

    shape_pattern: str = MISSING
    """Regular expression matching collider labels, including the body path."""

    sphere_subdivisions: int | None = None
    """Octahedral sphere subdivision level; two gives 128 uniformly distributed faces.

    None retains UV tessellation. Each subdivision projects new vertices onto
    the unit sphere before applying the authored sphere or ellipsoid scale.
    """

    segments: int | None = None
    """Circumferential segments for curved primitives; None retains the solver default."""

    axial_segments: int = 1
    """Axial panels on cylinder walls. Planar caps retain a minimal triangle fan."""

    primitive_quadrature_order: int = 0
    """Gauss order on analytic sphere/cylinder patches; zero retains planar triangles.

    Positive orders use order squared integration points per triangle. These
    points improve curved normals and area without increasing triangle count.
    Wet-region point rejection needs a separate quadrature convergence check.
    """


@configclass
class RFTSolverCfg(MJWarpSolverCfg):
    """MJWarp settings plus a dry granular field; no granular particles are created.

    ``bearing`` enables an experimental near-rest normal-pressure closure, not
    the published flowing RFT law. It provides no exact tangential sticking.
    """

    class_type: type | str = "{DIR}.rft_solver:RFTSolver"
    body_pattern: str = r".*/Robot/.*"
    bulk_density: float = 1600.0
    """Granular bulk density [kg/m³]; distinct from MuJoCo's ambient fluid density."""
    internal_friction: float = 0.4
    surface_friction: float = 0.4
    strength_multiplier: float = 1.0
    velocity_regularization: float = 0.02
    bearing: float = 0.0
    surface_height: float = 0.0
    surface_bottom: float | None = None
    segments: int = 12
    max_edge_length: float | None = None
    surface_resolution: list[RFTSurfaceResolutionCfg | dict] = []
    """Optional per-collider resolution overrides. Empty preserves the legacy surfaces."""
