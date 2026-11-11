# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Direct Go2 locomotion using the shared granular task contract."""

from ...granular_env import GranularVelocityEnv


class Go2RFTEnv(GranularVelocityEnv):
    """Go2 locomotion on rigid ground or in a granular field."""
