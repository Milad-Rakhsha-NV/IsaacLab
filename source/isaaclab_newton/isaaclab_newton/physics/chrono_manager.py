# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chrono Newton manager — maximal-coordinate DVI solver for Isaac Lab."""

from __future__ import annotations

import logging
import os

import warp as wp
from newton import Contacts, Control, Model, State, eval_fk, eval_ik
from newton.solvers import FrictionProjection, SolverChrono, SolverType, NumericalSolverConfig

_FRICTION_PROJECTION_MAP = {
    "cone": FrictionProjection.CONE,
    "tangential": FrictionProjection.TANGENTIAL,
}

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
    alpha: float = 0.005,
    recovery_speed: float = -1.0,
    position_correction: bool = False,
    diagonal_precondition: bool = False,
    precond_reg: float = 1e-4,
    friction_projection: FrictionProjection = FrictionProjection.CONE,
    backtrack_iterations: int = 5,
    block_precondition: bool = True,
    iterative_refinement_steps: int = 0,
) -> NumericalSolverConfig:
    """Build a NumericalSolverConfig from string parameters."""
    solver_type = _SOLVER_TYPE_MAP.get(solver_type_str)
    if solver_type is None:
        raise ValueError(
            f"Unknown solver type '{solver_type_str}'. "
            f"Available: {list(_SOLVER_TYPE_MAP.keys())}"
        )
    # Build position correction config if requested (same solver type, fewer iters)
    pos_cfg = None
    if position_correction:
        pos_solver_type = _SOLVER_TYPE_MAP.get(solver_type_str, SolverType.SPARSE_JACOBI)
        pos_cfg = NumericalSolverConfig(
            solver_type=pos_solver_type,
            max_iterations=max(max_iterations // 2, 10),
            alpha=alpha,
            recovery_speed=recovery_speed,
        )
    return NumericalSolverConfig(
        solver_type=solver_type,
        max_iterations=max_iterations,
        omega=omega,
        relax=relax,
        reg=reg,
        alpha=alpha,
        recovery_speed=recovery_speed,
        position_correction=pos_cfg,
        diagonal_precondition=diagonal_precondition,
        precond_reg=precond_reg,
        friction_projection=friction_projection,
        backtrack_iterations=backtrack_iterations,
        block_precondition=block_precondition,
        iterative_refinement_steps=iterative_refinement_steps,
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

        # Propagate rigid_contact_max from collision_cfg to model BEFORE
        # building the solver.  SolverChrono reads model.rigid_contact_max
        # to allocate its internal lambda buffer, but _initialize_contacts
        # (which normally sets this) runs AFTER _build_solver.
        collision_cfg = cls._collision_cfg
        if collision_cfg is not None and getattr(collision_cfg, "rigid_contact_max", None):
            model.rigid_contact_max = collision_cfg.rigid_contact_max
            logger.info(f"Chrono: set model.rigid_contact_max = {model.rigid_contact_max}")

        joint_config = _make_numerical_config(
            solver_type_str=solver_cfg.joint_solver_type,
            max_iterations=solver_cfg.joint_max_iterations,
            omega=solver_cfg.joint_omega,
            relax=solver_cfg.joint_relax,
            reg=solver_cfg.joint_reg,
            alpha=solver_cfg.joint_alpha,
            recovery_speed=solver_cfg.joint_recovery_speed,
            position_correction=solver_cfg.joint_position_correction,
            diagonal_precondition=solver_cfg.diagonal_precondition,
            precond_reg=solver_cfg.precond_reg,
            iterative_refinement_steps=solver_cfg.joint_iterative_refinement_steps,
        )

        # Resolve friction projection mode (env var overrides config)
        fp_str = os.environ.get(
            "NEWTON_FRICTION_PROJECTION",
            solver_cfg.contact_friction_projection,
        ).lower()
        fp = _FRICTION_PROJECTION_MAP.get(fp_str)
        if fp is None:
            raise ValueError(
                f"Unknown contact_friction_projection '{fp_str}'. "
                f"Available: {list(_FRICTION_PROJECTION_MAP.keys())}"
            )

        contact_config = _make_numerical_config(
            solver_type_str=solver_cfg.contact_solver_type,
            max_iterations=solver_cfg.contact_max_iterations,
            omega=solver_cfg.contact_omega,
            relax=solver_cfg.contact_relax,
            reg=solver_cfg.contact_reg,
            alpha=solver_cfg.contact_alpha,
            recovery_speed=solver_cfg.contact_recovery_speed,
            position_correction=solver_cfg.contact_position_correction,
            friction_projection=fp,
            backtrack_iterations=solver_cfg.contact_backtrack_iterations,
            block_precondition=solver_cfg.contact_block_precondition,
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
            joint_limit_ke_scale=solver_cfg.joint_limit_ke_scale,
            enable_timers=False,
        )
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        logger.info(
            f"Chrono solver: joint={solver_cfg.joint_solver_type} "
            f"contact={solver_cfg.contact_solver_type} "
            f"implicit_pd={solver_cfg.use_implicit_pd} "
            f"angular_damping={solver_cfg.angular_damping} "
            f"friction_projection={fp_str}"
        )

    @classmethod
    def initialize_solver(cls) -> None:
        """Override to pre-run prepare_for_capture before CUDA graph capture.

        The block-sparse LDL symbolic factorization reads joint topology
        from the model via ``.numpy()`` (CPU transfer).  This must happen
        before CUDA graph capture starts.  We call it after the base
        ``initialize_solver`` has built the solver and contacts but before
        it captures the graph.
        """
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from isaaclab.physics import PhysicsManager

        cfg = PhysicsManager._cfg
        if cfg is None:
            return

        # --- Run base logic EXCEPT the graph capture ---
        from isaaclab.utils.timer import Timer
        with Timer(name="newton_initialize_solver", msg="Initialize solver took:"):
            NewtonManager._num_substeps = cfg.num_substeps
            NewtonManager._solver_dt = cls.get_physics_dt() / cls._num_substeps
            NewtonManager._collision_cfg = cfg.collision_cfg

            cls._build_solver(cls._model, cfg.solver_cfg)
            if NewtonManager._solver is None:
                raise RuntimeError(
                    f"{cls.__name__}._build_solver did not assign NewtonManager._solver."
                )
            cls._initialize_contacts()

        if cls._usdrt_stage is not None:
            cls._setup_cubric_bindings()

        # --- Pre-run finalize_for_capture (CPU work) BEFORE graph capture ---
        # The block-sparse LDL symbolic factorization reads joint topology
        # via .numpy() (CPU transfer). This must happen before CUDA graph
        # capture starts. SolverChrono.finalize_for_capture() calls
        # prepare_for_capture on all sub-solvers; the methods are idempotent
        # so subsequent calls during capture short-circuit.
        solver = NewtonManager._solver
        state_0 = cls._state_0
        if state_0 is not None and hasattr(solver, 'finalize_for_capture'):
            solver.finalize_for_capture(state_0)
            logger.info("Chrono: pre-ran finalize_for_capture (joint + contact solvers)")

        # --- Now capture CUDA graph (all prepare_for_capture will short-circuit) ---
        device = PhysicsManager._device
        use_cuda_graph = cfg.use_cuda_graph and "cuda" in device
        if use_cuda_graph:
            cls._capture_or_defer_cuda_graph()
        else:
            NewtonManager._graph = None

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

        # Chrono works in maximal coordinates — sync joint coords from body state
        eval_ik(cls._model, state_1, state_1.joint_q, state_1.joint_qd)

        # Sanitize NaN/Inf in body and joint state arrays
        wp.launch(
            _sanitize_transform,
            dim=state_1.body_q.shape[0],
            outputs=[state_1.body_q],
            device=state_1.body_q.device,
        )
        wp.launch(
            _sanitize_float,
            dim=state_1.joint_q.shape[0],
            outputs=[state_1.joint_q],
            device=state_1.joint_q.device,
        )

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
            # Compute contact forces on GPU (replaces solver.update_contacts
            # which uses CPU numpy loops).
            cls._gpu_update_contact_forces(eval_contacts)
            # Bridge rigid_contact_force → contacts.force (spatial_vector)
            # with impulse→force scaling.
            cls._populate_spatial_forces(eval_contacts)
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, eval_contacts)

    # ------------------------------------------------------------------
    # GPU contact force update (replaces solver.update_contacts)
    # ------------------------------------------------------------------

    @classmethod
    def _gpu_update_contact_forces(cls, contacts: Contacts) -> None:
        """Compute rigid_contact_force from solver lambda on GPU.

        Replaces :meth:`SolverChrono.update_contacts` which uses a CPU numpy
        loop.  This version is CUDA-graph safe.
        """
        constraint = cls._solver._contact_solver._constraint
        wp.launch(
            _compute_contact_force_from_lambda,
            dim=constraint.contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                constraint.lambda_,
                constraint.contact_max,
            ],
            outputs=[contacts.rigid_contact_force],
            device=contacts.device,
        )

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
def _compute_contact_force_from_lambda(
    num_contacts: wp.array(dtype=wp.int32),
    normal: wp.array(dtype=wp.vec3f),
    lambda_: wp.array(dtype=wp.float32),
    contact_max: int,
    # output
    force: wp.array(dtype=wp.vec3f),
):
    """Compute rigid_contact_force = normal * lambda (normal component).

    Each contact has 3 lambda rows (normal + 2 friction).  The normal
    impulse is at index ``i * 3``.  Contacts beyond ``num_contacts`` or
    ``contact_max`` are zeroed.
    """
    i = wp.tid()
    if i >= num_contacts[0] or i >= contact_max:
        force[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    # Normal impulse is at lambda_[i*3]; friction at i*3+1, i*3+2
    lam_n = lambda_[i * 3]
    force[i] = normal[i] * lam_n


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


@wp.kernel
def _sanitize_float(
    # in/out
    arr: wp.array(dtype=wp.float32),
):
    """Replace NaN/Inf with zero in a float array."""
    i = wp.tid()
    v = arr[i]
    if wp.isnan(v) or wp.isinf(v):
        arr[i] = 0.0


@wp.kernel
def _sanitize_transform(
    # in/out
    arr: wp.array(dtype=wp.transformf),
):
    """Replace NaN/Inf components with identity transform."""
    i = wp.tid()
    t = arr[i]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    bad = False
    for c in range(3):
        if wp.isnan(p[c]) or wp.isinf(p[c]):
            bad = True
    for c in range(4):
        if wp.isnan(q[c]) or wp.isinf(q[c]):
            bad = True
    if bad:
        arr[i] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
