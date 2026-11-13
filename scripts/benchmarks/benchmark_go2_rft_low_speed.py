# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose explicit RFT drift against a scalar implicit static-reaction oracle.

This fixture integrates one vertical coordinate of a 1 kg spherical Go2 foot.
It is not an articulated solver. The static oracle fills the zero-velocity
interval between the two directional flowing-RFT limits. Geometry is evaluated
at the beginning of each step; velocity and reaction satisfy the discrete
momentum equation. No elastic spring or backing floor supplies support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from newton.solvers import SolverRFT

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(args):
    """Compare the same initial states, load schedules and material parameters."""
    args.output.mkdir(parents=True, exist_ok=True)
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label="foot")
    builder.add_shape_sphere(body, radius=0.022, label="foot/sphere", cfg=builder.ShapeConfig(density=0))
    model = builder.finalize(device="cpu")
    surfaces, inventory = collision_surfaces(
        model, ".*", 12, resolutions=[dict(shape_pattern=".*", sphere_subdivisions=2)]
    )
    solvers = {
        name: SolverRFT(
            model,
            surfaces,
            surface_height=0,
            surface_shape_ids=[row["shape"] for row in inventory],
            surface_quadrature_orders=[6],
            config=SolverRFT.Config(bearing=bearing, velocity_regularization=regularization),
        )
        for name, bearing, regularization in (("bearing", 0.87574, 0.04), ("flow", 0.0, 0.0))
    }
    state = model.state()
    results, traces = [], {}
    start = time.monotonic()
    # The starting depths are admissible for the existing bearing capacity:
    # 110% of the two loads. The flowing limits are checked independently below.
    for method in ("bearing", "flow", "implicit_1d"):
        for substeps in args.substeps:
            for nominal_load, initial_depth in ((18.417051, 0.0523165184), (36.834103, 0.0991330367)):
                solver = solvers["flow" if method == "implicit_1d" else method]
                dt = 0.005 / substeps
                position, velocity = 0.022 - initial_depth, 0.0
                state.body_q.assign([[0, 0, position, 0, 0, 0, 1]])
                state.body_qd.assign([[0, 0, 1, 0, 0, 0]])
                initial_lower = float(solvers["flow"].evaluate(state).numpy()[0, 2])
                state.body_qd.assign([[0, 0, -1, 0, 0, 0]])
                initial_upper = float(solvers["flow"].evaluate(state).numpy()[0, 2])
                reverse_load = -min(0.25 * nominal_load, 0.5 * abs(initial_lower))
                history, positive_work = [], 0.0
                for step in range(round(args.seconds / dt)):
                    t = step * dt
                    load = nominal_load
                    if args.load_cycle:
                        # Removal, upward pull, overload and return, in equal fifths.
                        stage = min(4, int(t / (args.seconds / 5)))
                        load = (nominal_load, 0.0, reverse_load, 1.2 * initial_upper, nominal_load)[stage]
                    state.body_q.assign([[0, 0, position, 0, 0, 0, 1]])
                    lower, upper = float("nan"), float("nan")
                    if method == "implicit_1d":
                        state.body_qd.assign([[0, 0, -1, 0, 0, 0]])
                        upper = float(solver.evaluate(state).numpy()[0, 2])
                        state.body_qd.assign([[0, 0, 1, 0, 0, 0]])
                        lower = float(solver.evaluate(state).numpy()[0, 2])
                        if lower > upper:
                            raise RuntimeError("Directional reactions do not define an ordered vertical graph")
                        required = load - velocity / dt
                        force = float(np.clip(required, lower, upper))
                        next_velocity = velocity + dt * (force - load)
                        if lower <= required <= upper:
                            next_velocity = 0.0
                    else:
                        state.body_qd.assign([[0, 0, velocity, 0, 0, 0]])
                        force = float(solver.evaluate(state).numpy()[0, 2])
                        next_velocity = velocity + dt * (force - load)
                    momentum_error = (next_velocity - velocity) - dt * (force - load)
                    if abs(momentum_error) > 1e-10:
                        raise RuntimeError("Vertical momentum balance failed")
                    displacement = dt * next_velocity
                    positive_work += max(0.0, force * displacement)
                    position += displacement
                    velocity = next_velocity
                    if (step + 1) % substeps == 0:
                        history.append([(step + 1) * dt, 0.022 - position, velocity, force, load, lower, upper])
                values = np.asarray(history)
                if not np.isfinite(values[:, :5]).all():
                    raise RuntimeError("Nonfinite scalar trajectory")
                depth_2s = values[np.argmin(abs(values[:, 0] - min(2.0, args.seconds))), 1]
                key = f"{method}_{substeps}_{nominal_load:.0f}"
                traces[key] = values
                results.append(
                    dict(
                        method=method,
                        substeps=substeps,
                        load_N=nominal_load,
                        initial_directional_limits_N=[initial_lower, initial_upper],
                        depth_2s_m=float(depth_2s),
                        depth_final_m=float(values[-1, 1]),
                        creep_m=float(values[-1, 1] - depth_2s),
                        positive_granular_work_J=positive_work,
                    )
                )
                (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = dict(
        results=results,
        seconds=args.seconds,
        load_cycle=args.load_cycle,
        wall_seconds=time.monotonic() - start,
        source_sha256=SOURCE_SHA256,
        newton_kernel_sha256=hashlib.sha256(
            (Path(newton.__file__).parent / "_src/solvers/rft/kernels.py").read_bytes()
        ).hexdigest(),
        trace_columns=["time_s", "sole_depth_m", "velocity_m_s", "reaction_N", "load_N", "lower_N", "upper_N"],
        scope="Scalar vertical integration diagnostic in an unbounded bed; "
        "not an articulated solver or MPM calibration",
    )
    np.savez_compressed(args.output / "trace.npz", **traces)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--substeps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--load_cycle", action="store_true")
    settings = parser.parse_args()
    if settings.seconds <= 0 or any(value <= 0 for value in settings.substeps):
        parser.error("Duration and substep counts must be positive")
    run(settings)
