# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DVI Newton manager — maximal-coordinate DVI solver for Isaac Lab."""

from __future__ import annotations

import logging

import warp as wp
from newton import Contacts, Control, Model, State, eval_fk, eval_ik
from newton.solvers import ActuatorIntegration, FrictionProjection, SolverDVI, SolverType, NumericalSolverConfig

_FRICTION_PROJECTION_MAP = {
    "cone": FrictionProjection.CONE,
    "tangential": FrictionProjection.TANGENTIAL,
}

from isaaclab.physics import PhysicsManager

from .dvi_manager_cfg import DVISolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

# Map string names to SolverType enum values
_SOLVER_TYPE_MAP = {
    "sparse_jacobi": SolverType.SPARSE_JACOBI,
    "sparse_ldl": SolverType.SPARSE_LDL,
    "sparse_apgd": SolverType.SPARSE_APGD,
    "sparse_aspg": SolverType.SPARSE_ASPG,
    "sparse_pspg": SolverType.SPARSE_PSPG,
    "sparse_block_gs": SolverType.SPARSE_GS,
}


def _make_numerical_config(
    solver_type_str: str,
    omega: float,
    relax: float,
    reg: float,
    max_iterations: int = 50,
    alpha: float = 0.005,
    recovery_speed: float = -1.0,
    compliance: float = 0.0,
    diagonal_precondition: bool = False,
    precond_reg: float = 1e-4,
    friction_projection: FrictionProjection = FrictionProjection.CONE,
    block_precondition: bool = False,
    iterative_refinement_steps: int = 0,
    tolerance: float | None = None,
    aspg_no_momentum: bool = False,
    aspg_alpha_max_rel: float | None = None,
    aspg_seed_alpha_max: bool = False,
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
        alpha=alpha,
        recovery_speed=recovery_speed,
        compliance=float(compliance),
        diagonal_precondition=diagonal_precondition,
        precond_reg=precond_reg,
        friction_projection=friction_projection,
        block_precondition=block_precondition,
        iterative_refinement_steps=iterative_refinement_steps,
        aspg_no_momentum=aspg_no_momentum,
        aspg_seed_alpha_max=aspg_seed_alpha_max,
        **({} if tolerance is None else {"tolerance": tolerance}),
        **({} if aspg_alpha_max_rel is None else {"aspg_alpha_max_rel": aspg_alpha_max_rel}),
    )


class NewtonDVIManager(NewtonManager):
    """:class:`NewtonManager` specialization for the DVI solver.

    The DVI solver operates in maximal coordinates (body_q/body_qd) and
    requires:
      - Newton's CollisionPipeline for contact detection (per substep)
      - eval_ik after stepping to sync joint_q/joint_qd from body state
      - CUDA graphs disabled (collision involves variable contact counts)
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: DVISolverCfg) -> None:
        """Construct :class:`SolverDVI` and set base-class slots."""

        # Propagate rigid_contact_max from collision_cfg to model BEFORE
        # building the solver.  SolverDVI reads model.rigid_contact_max
        # to allocate its internal lambda buffer, but _initialize_contacts
        # (which normally sets this) runs AFTER _build_solver.
        collision_cfg = cls._collision_cfg
        if collision_cfg is not None and getattr(collision_cfg, "rigid_contact_max", None):
            model.rigid_contact_max = collision_cfg.rigid_contact_max
            logger.info(f"DVI: set model.rigid_contact_max = {model.rigid_contact_max}")

        # NOTE: joint solver is typically sparse_ldl (a direct factorization),
        # for which max_iterations is unused; it is intentionally not passed here.
        joint_config = _make_numerical_config(
            solver_type_str=solver_cfg.joint_solver_type,
            omega=solver_cfg.joint_omega,
            relax=solver_cfg.joint_relax,
            reg=solver_cfg.joint_reg,
            alpha=solver_cfg.joint_alpha,
            recovery_speed=solver_cfg.joint_recovery_speed,
            diagonal_precondition=solver_cfg.diagonal_precondition,
            precond_reg=solver_cfg.precond_reg,
            iterative_refinement_steps=solver_cfg.joint_iterative_refinement_steps,
        )

        # Resolve friction projection mode from config
        fp_str = solver_cfg.contact_friction_projection.lower()
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
            compliance=solver_cfg.contact_compliance,
            friction_projection=fp,
            block_precondition=solver_cfg.contact_block_precondition,
            tolerance=solver_cfg.contact_tolerance,
            aspg_no_momentum=solver_cfg.contact_aspg_no_momentum,
            aspg_alpha_max_rel=solver_cfg.contact_aspg_alpha_max_rel,
            aspg_seed_alpha_max=solver_cfg.contact_aspg_seed_alpha_max,
        )

        # Build joint limit solver config if constraint-based limits requested
        joint_limit_config = None
        if solver_cfg.joint_limit_solver_type is not None:
            joint_limit_config = _make_numerical_config(
                solver_type_str=solver_cfg.joint_limit_solver_type,
                max_iterations=solver_cfg.joint_limit_max_iterations,
                omega=solver_cfg.joint_limit_omega,
                relax=solver_cfg.joint_limit_relax,
                reg=solver_cfg.joint_limit_reg,
                alpha=solver_cfg.joint_limit_alpha,
                recovery_speed=solver_cfg.joint_limit_recovery_speed,
            )

        # Apply per-joint armature overrides if configured
        if solver_cfg.armature_override and model.joint_armature is not None:
            import numpy as np
            arm = model.joint_armature.numpy()
            qd_start = model.joint_qd_start.numpy()
            dof_dim = model.joint_dof_dim.numpy()
            changed = 0
            for j in range(model.joint_count):
                label = str(model.joint_label[j]).lower()
                qds = qd_start[j]
                ndof = dof_dim[j, 0] + dof_dim[j, 1]
                for pattern, val in solver_cfg.armature_override.items():
                    if pattern.lower() in label:
                        for d in range(ndof):
                            arm[qds + d] = val
                        changed += 1
                        break
            model.joint_armature.assign(wp.array(arm, dtype=wp.float32, device=model.device))
            logger.info(f"DVI: armature_override applied to {changed} joints")

        # Resolve actuator integration mode
        actuator_integration = solver_cfg.actuator_integration
        try:
            ai_mode = ActuatorIntegration(actuator_integration)
        except ValueError:
            raise ValueError(
                f"Unknown actuator_integration '{actuator_integration}'. "
                f"Available: {[e.value for e in ActuatorIntegration]}"
            )

        NewtonManager._solver = SolverDVI(
            model,
            joint_limit_solver=joint_limit_config,
            joint_solver=joint_config,
            contact_solver=contact_config,
            angular_damping=solver_cfg.angular_damping,
            enable_actuation=solver_cfg.enable_actuation,
            coupling_iterations=solver_cfg.coupling_iterations,
            cache_factorization=solver_cfg.cache_factorization,
            post_stabilize_joints=solver_cfg.post_stabilize_joints,
            actuator_integration=ai_mode,
            enable_timers=False,
        )
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        limit_mode = "constraint" if joint_limit_config is not None else "penalty"
        logger.info(
            f"DVI solver: joint={solver_cfg.joint_solver_type} "
            f"contact={solver_cfg.contact_solver_type} "
            f"coupling_iterations={solver_cfg.coupling_iterations} "
            f"cache_factorization={solver_cfg.cache_factorization} "
            f"post_stabilize_joints={solver_cfg.post_stabilize_joints} "
            f"actuator_integration={ai_mode.value} "
            f"angular_damping={solver_cfg.angular_damping} "
            f"friction_projection={fp_str} "
            f"joint_limits={limit_mode}"
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
        # capture starts. SolverDVI.finalize_for_capture() calls
        # prepare_for_capture on all sub-solvers; the methods are idempotent
        # so subsequent calls during capture short-circuit.
        solver = NewtonManager._solver
        state_0 = cls._state_0
        if state_0 is not None and hasattr(solver, 'finalize_for_capture'):
            solver.finalize_for_capture(state_0)
            logger.info("DVI: pre-ran finalize_for_capture (joint + contact solvers)")

        # --- Now capture CUDA graph (all prepare_for_capture will short-circuit) ---
        device = PhysicsManager._device
        use_cuda_graph = cfg.use_cuda_graph and "cuda" in device
        if use_cuda_graph:
            cls._capture_or_defer_cuda_graph()
        else:
            NewtonManager._graph = None

    @classmethod
    def _step_solver(cls, state_0: State, state_1: State, control: Control, substep_dt: float) -> None:
        """Run one DVI substep: collide → step → eval_ik.

        Overrides the base to:
          1. Run collision detection per substep (DVI needs fresh contacts each step)
          2. Step the solver
          3. Run eval_ik to sync joint_q/joint_qd from body_q/body_qd
        """
        # Collision detection per substep
        cls._collision_pipeline.collide(state_0, cls._contacts)

        # Step the solver with contacts
        cls._solver.step(state_0, state_1, control, cls._contacts, substep_dt)

        # DVI works in maximal coordinates — sync joint coords from body state
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
        """Run one physics step — DVI-specific override.

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
            # Write contacts.force (spatial_vector) directly from solver lambda
            # using Newton's own kernel which handles sign convention and
            # friction correctly.
            cls._write_contact_forces_gpu(eval_contacts)
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, eval_contacts)



    # ------------------------------------------------------------------
    # GPU contact force update
    # ------------------------------------------------------------------

    @classmethod
    def _write_contact_forces_gpu(cls, contacts: Contacts) -> None:
        """Write contacts.force (spatial_vector) from solved contact lambda.

        Uses Newton's own ``write_contact_forces`` kernel which correctly:
        - Applies the DVI Jacobian transpose sign (force_on_shape0 = J_a^T * lambda / dt)
        - Includes both normal and friction impulse components
        - Converts impulse to force (divides by dt)

        This replaces the previous two-step pipeline
        (_compute_contact_force_from_lambda + _copy_rigid_force_to_spatial)
        which had a sign error (missing negation) and ignored friction.
        """
        if contacts.force is None:
            return

        from newton._src.solvers.dvi.contact_kernels import write_contact_forces

        constraint = cls._solver._contact_solver._constraint
        wp.launch(
            write_contact_forces,
            dim=constraint.contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                constraint.lambda_,
                cls._solver_dt,
                constraint.contact_max,
            ],
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
