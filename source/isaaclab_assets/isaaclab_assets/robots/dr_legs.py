# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney DR Legs closed-loop biped.

DR Legs is a parallel-linkage bipedal lower body with 36 revolute joints: 12
actuated joints and 24 passive closed-loop linkage degrees of freedom. The
robot has cyclic kinematic loops and therefore no USD ``ArticulationRootAPI``;
it is simulated with the Kamino solver, which handles closed kinematic chains in
maximal coordinates. See :class:`~isaaclab_newton.assets.articulation.closed_loop_view.ClosedLoopView`.

The following configuration is available:

* :data:`DR_LEGS_IMPLICIT_PD_CFG`: DR Legs with implicit (solver-side) PD on the
  12 actuated joints and zero-PD on the 24 passive linkage joints.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# TODO: switch ``usd_path`` to ``ISAACLAB_NUCLEUS_DIR`` once the DR Legs USD is hosted on Nucleus.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dr_legs")

DR_LEGS_JOINT_ORDER: list[str] = [
    "j1_l_i",
    "j2_l_i",
    "j3_l_i",
    "j4_l_i",
    "j5_l_i",
    "j6_l_i",
    "j7_l_i",
    "j8_l_i",
    "j9_l_i",
    "j1_l_o",
    "j2_l_o",
    "j3_l_o",
    "j4_l_o",
    "j5_l_o",
    "j6_l_o",
    "j7_l_o",
    "j8_l_o",
    "j9_l_o",
    "j1_r_i",
    "j2_r_i",
    "j3_r_i",
    "j4_r_i",
    "j5_r_i",
    "j6_r_i",
    "j7_r_i",
    "j8_r_i",
    "j9_r_i",
    "j1_r_o",
    "j2_r_o",
    "j3_r_o",
    "j4_r_o",
    "j5_r_o",
    "j6_r_o",
    "j7_r_o",
    "j8_r_o",
    "j9_r_o",
]
"""Canonical ordering of all 36 DR Legs joints."""

DR_LEGS_ACTUATED_JOINTS: list[str] = [
    "j1_l_i",
    "j2_l_i",
    "j6_l_i",
    "j7_l_i",
    "j2_l_o",
    "j7_l_o",
    "j1_r_i",
    "j2_r_i",
    "j6_r_i",
    "j7_r_i",
    "j2_r_o",
    "j7_r_o",
]
"""The 12 servo-driven joints (6 per leg)."""

DR_LEGS_PASSIVE_JOINTS: list[str] = [j for j in DR_LEGS_JOINT_ORDER if j not in DR_LEGS_ACTUATED_JOINTS]
"""The 24 closed-loop linkage DOFs that are not driven by an actuator on real hardware."""


_DR_LEGS_SPAWN = sim_utils.UsdFileCfg(
    usd_path=os.path.join(_DATA_DIR, "dr_legs.usda"),
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=10.0,
        enable_gyroscopic_forces=True,
    ),
    copy_from_source=False,
)


_DR_LEGS_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.28),
    # Closed-loop FK is only valid at the assembled reference (all joint coords zero).
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


DR_LEGS_IMPLICIT_PD_CFG = ArticulationCfg(
    spawn=_DR_LEGS_SPAWN,
    init_state=_DR_LEGS_INIT_STATE,
    actuators={
        "driven_joints": ImplicitActuatorCfg(
            joint_names_expr=DR_LEGS_ACTUATED_JOINTS,
            stiffness=5.0,
            damping=0.2,
            effort_limit_sim=3.1,
        ),
        # Linkage DOFs are undriven: explicit zeros so the solver ignores USD drive defaults.
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=DR_LEGS_PASSIVE_JOINTS,
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            friction=0.0,
            effort_limit_sim=400.0,
        ),
    },
)
"""DR Legs with implicit PD on the 12 actuated joints (Kamino solver, closed-loop)."""
