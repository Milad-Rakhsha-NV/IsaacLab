# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental surface RFT composed with MJWarp articulation dynamics."""

from isaaclab.utils import configclass

from .mjwarp_solver_cfg import MJWarpSolverCfg


@configclass
class RFTSolverCfg(MJWarpSolverCfg):
    """MJWarp settings plus a dry granular field; no granular particles are created.

    ``bearing`` enables an experimental near-rest normal-pressure closure, not
    the published flowing RFT law. It provides no exact tangential sticking.
    """

    class_type: type | str = "{DIR}.rft_solver:RFTSolver"
    body_pattern: str = r".*/Robot/.*"
    density: float = 1600.0
    internal_friction: float = 0.4
    surface_friction: float = 0.4
    strength_multiplier: float = 1.0
    velocity_regularization: float = 0.02
    bearing: float = 0.0
    surface_height: float = 0.0
    segments: int = 12
    max_edge_length: float | None = None
