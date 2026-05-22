# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chrono Newton manager — maximal-coordinate DVI solver for Isaac Lab."""

from __future__ import annotations

import logging

import torch
import warp as wp
from newton import Contacts, Control, Model, State, eval_fk, eval_ik
from newton.solvers import SolverChrono, SolverType, NumericalSolverConfig

from isaaclab.physics import PhysicsManager

from .chrono_manager_cfg import ChronoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

# Map string names to SolverType enum values
_SOLVER_TYPE_MAP = {
    "sparse_jacobi": SolverType.SPARSE_JACOBI,
    "sparse_ldl": SolverType.SPARSE_LDL,
    "sparse_apgd": SolverType.SPARSE_APGD,
    "sparse_block_gs": SolverType.SPARSE_GS,
}


def _make_numerical_config(
    solver_type_str: str,
    max_iterations: int,
    omega: float,
    relax: float,
    reg: float,
    recovery_speed: float = -1.0,
) -> NumericalSolverConfig:
    """Build a NumericalSolverConfig from string parameters."""
    solver_type = _SOLVER_TYPE_MAP.get(solver_type_str)
    if solver_type is None:
        raise ValueError(
            f"Unknown solver type '{solver_type_str}'. "
            f"Available: {list(_SOLVER_TYPE_MAP.keys())}"
        )
    return NumericalSolverConfig(
        solver_type=solver_type,
        max_iterations=max_iterations,
        omega=omega,
        relax=relax,
        reg=reg,
        recovery_speed=recovery_speed,
        position_correction=None,
    )


class NewtonChronoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Chrono DVI solver.

    The Chrono solver operates in maximal coordinates (body_q/body_qd) and
    requires:
      - Newton's CollisionPipeline for contact detection (per substep)
      - eval_ik after stepping to sync joint_q/joint_qd from body state
      - CUDA graphs disabled (collision involves variable contact counts)
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ChronoSolverCfg) -> None:
        """Construct :class:`SolverChrono` and set base-class slots."""

        joint_config = _make_numerical_config(
            solver_type_str=solver_cfg.joint_solver_type,
            max_iterations=solver_cfg.joint_max_iterations,
            omega=solver_cfg.joint_omega,
            relax=solver_cfg.joint_relax,
            reg=solver_cfg.joint_reg,
        )

        contact_config = _make_numerical_config(
            solver_type_str=solver_cfg.contact_solver_type,
            max_iterations=solver_cfg.contact_max_iterations,
            omega=solver_cfg.contact_omega,
            relax=solver_cfg.contact_relax,
            reg=solver_cfg.contact_reg,
            recovery_speed=solver_cfg.contact_recovery_speed,
        )

        NewtonManager._solver = SolverChrono(
            model,
            joint_solver=joint_config,
            contact_solver=contact_config,
            angular_damping=solver_cfg.angular_damping,
            enable_actuation=solver_cfg.enable_actuation,
            enable_contacts=solver_cfg.enable_contacts,
            enable_gyroscopic=solver_cfg.enable_gyroscopic,
            use_implicit_pd=solver_cfg.use_implicit_pd,
            enable_timers=False,
        )
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        # Store max velocity for clamping
        NewtonManager._chrono_max_velocity = solver_cfg.max_velocity

        logger.info(
            f"Chrono solver: joint={solver_cfg.joint_solver_type} "
            f"contact={solver_cfg.contact_solver_type} "
            f"implicit_pd={solver_cfg.use_implicit_pd} "
            f"angular_damping={solver_cfg.angular_damping}"
        )

    @classmethod
    def _step_solver(cls, state_0: State, state_1: State, control: Control, substep_dt: float) -> None:
        """Run one Chrono substep: collide → step → eval_ik.

        Overrides the base to:
          1. Run collision detection per substep (Chrono needs fresh contacts each step)
          2. Step the solver
          3. Run eval_ik to sync joint_q/joint_qd from body_q/body_qd
        """
        # Collision detection per substep
        cls._collision_pipeline.collide(state_0, cls._contacts)

        # Step the solver with contacts
        cls._solver.step(state_0, state_1, control, cls._contacts, substep_dt)

        # Clamp body velocities to prevent catastrophic divergence
        # The DVI solver can produce extreme velocities from deep penetration.
        # Clamp BEFORE eval_ik to avoid propagating bad velocities to joint space.
        max_vel = cls._chrono_max_velocity
        if max_vel > 0:
            body_qd_torch = wp.to_torch(state_1.body_qd)
            torch.clamp_(body_qd_torch, min=-max_vel, max=max_vel)

        # Chrono works in maximal coordinates — sync joint coords from body state
        eval_ik(cls._model, state_1, state_1.joint_q, state_1.joint_qd)

        # Also clamp joint velocities (eval_ik derives them from body_qd)
        if max_vel > 0:
            joint_qd_torch = wp.to_torch(state_1.joint_qd)
            torch.clamp_(joint_qd_torch, min=-max_vel, max=max_vel)

        # Sanitize: replace any NaN/Inf values with zero
        # This is a safety net for edge cases the clamping misses
        body_q_torch = wp.to_torch(state_1.body_q)
        body_qd_torch2 = wp.to_torch(state_1.body_qd)
        joint_q_torch = wp.to_torch(state_1.joint_q)
        joint_qd_torch2 = wp.to_torch(state_1.joint_qd)
        torch.nan_to_num_(body_q_torch, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(body_qd_torch2, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(joint_q_torch, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(joint_qd_torch2, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Run one physics step — Chrono-specific override.

        Key differences from the base:
          - Collision detection happens inside _step_solver (per substep), not once before
          - CUDA graphs are not used (variable contact counts)
          - eval_ik is called after each substep via _step_solver
        """
        # Note: collision is handled inside _step_solver per substep,
        # so we do NOT call cls._collision_pipeline.collide() here.

        cfg = PhysicsManager._cfg
        need_copy_on_last_substep = (cfg is not None and getattr(cfg, "use_cuda_graph", False)) and cls._num_substeps % 2 == 1

        for i in range(cls._num_substeps):
            cls._step_solver(cls._state_0, cls._state_1, cls._control, cls._solver_dt)
            if need_copy_on_last_substep and i == cls._num_substeps - 1:
                cls._state_0.assign(cls._state_1)
            else:
                NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
            cls._state_0.clear_forces()

        # Update frame transform sensors
        if cls._newton_frame_transform_sensors:
            for sensor in cls._newton_frame_transform_sensors:
                sensor.update(cls._state_0)

        # Update IMU sensors
        if cls._newton_imu_sensors:
            for sensor in cls._newton_imu_sensors:
                sensor.update(cls._state_0)

        # Populate contacts for contact sensors
        if cls._report_contacts:
            eval_contacts = cls._contacts
            # SolverChrono.update_contacts populates rigid_contact_force but NOT
            # contacts.force (spatial_vector).  SensorContact reads contacts.force,
            # so we need to bridge the gap.
            cls._safe_update_contacts(eval_contacts)
            cls._populate_spatial_forces(eval_contacts)
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, eval_contacts)

    # ------------------------------------------------------------------
    # Safe update_contacts wrapper
    # ------------------------------------------------------------------

    @classmethod
    def _safe_update_contacts(cls, contacts: Contacts) -> None:
        """Call solver.update_contacts with bounds protection.

        The Chrono ContactSolver.update_contacts iterates ``contact_count`` times
        and indexes into ``lambda_[i*3]``. If ``contact_count`` exceeds the lambda
        buffer size (allocated at solver creation time), it crashes with an
        IndexError.  We clamp the count to avoid this.
        """
        try:
            cls._solver.update_contacts(contacts)
        except IndexError:
            # Lambda buffer overflow — too many contacts for the allocated size.
            # Clamp the count so the solver can still partially update.
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            constraint = cls._solver._contact_solver._constraint
            max_lambda = constraint.lambda_.shape[0] // 3
            logger.warning(
                f"Chrono contact buffer overflow: {contact_count} contacts but "
                f"lambda buffer only holds {max_lambda}. Clamping."
            )
            import numpy as np
            contacts.rigid_contact_count.assign(np.array([max_lambda], dtype=np.int32))
            try:
                cls._solver.update_contacts(contacts)
            except Exception:
                pass  # Give up on force reporting for this step
            # Restore the real count
            contacts.rigid_contact_count.assign(np.array([contact_count], dtype=np.int32))

    # ------------------------------------------------------------------
    # Bridge: rigid_contact_force (vec3) → contacts.force (spatial_vector)
    # ------------------------------------------------------------------

    @classmethod
    def _populate_spatial_forces(cls, contacts: Contacts) -> None:
        """Copy rigid_contact_force → contacts.force so SensorContact can read it.

        The Chrono solver only writes per-contact normal forces to
        ``contacts.rigid_contact_force`` (vec3).  Newton's SensorContact reads
        from ``contacts.force`` (spatial_vector) where the *top* 3 components
        carry the linear force and the bottom 3 carry the torque.

        The DVI solver reports constraint impulses (lambda), not forces.
        We scale by 1/dt to convert to Newtons (force = impulse / dt).
        """
        if contacts.force is None:
            return  # no sensor requested it

        # Scale factor: convert DVI impulse to force
        inv_dt = 1.0 / cls._solver_dt if cls._solver_dt > 0 else 1.0

        contact_count = contacts.rigid_contact_count
        wp.launch(
            _copy_rigid_force_to_spatial,
            dim=contacts.rigid_contact_max,
            inputs=[contact_count, contacts.rigid_contact_force, inv_dt],
            outputs=[contacts.force],
            device=contacts.device,
        )


@wp.kernel
def _copy_rigid_force_to_spatial(
    num_contacts: wp.array(dtype=wp.int32),
    rigid_force: wp.array(dtype=wp.vec3f),
    scale: float,
    # output
    spatial_force: wp.array(dtype=wp.spatial_vectorf),
):
    """Write rigid_contact_force (scaled to Newtons) into contacts.force."""
    i = wp.tid()
    if i >= num_contacts[0]:
        # Zero out unused slots
        spatial_force[i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return
    f = rigid_force[i] * scale
    # spatial_vector: top = linear force, bottom = torque (zero for point contacts)
    spatial_force[i] = wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0)
