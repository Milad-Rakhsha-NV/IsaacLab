# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure net work on closed vertical penetration cycles using the exact G1 foot."""

import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverRFT


def main(args):
    """Integrate loading input minus returned work; negative values create energy."""
    geometry = json.loads(args.geometry_file.read_text())
    foot = next(r for r in geometry["surfaces"] if r["body_label"].endswith("/left_ankle_roll_link"))
    mesh = newton.Mesh(
        np.array(foot["vertices"], np.float32), np.array(foot["indices"], np.int32), compute_inertia=False
    )
    samples = 201
    depth = np.linspace(0, 0.04, samples)
    builder = newton.ModelBuilder()
    for _ in range(2 * samples):
        builder.add_body(mass=1, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=args.device)
    state = model.state()
    pose = np.zeros((2 * samples, 7), np.float32)
    pose[:, 6] = 1
    pose[:, 2] = -mesh.vertices[:, 2].min() - np.r_[depth, depth]
    velocity = np.zeros((2 * samples, 6), np.float32)
    state.body_q.assign(pose)
    results = []
    for down, up in ((0.04, 0.001), (0.1, 0.001), (0.2, 0.005), (0.005, 0.0001)):
        velocity[:samples, 2] = -down
        velocity[samples:, 2] = up
        state.body_qd.assign(velocity)
        for bearing in (0.87574, 0.3, 0.0):
            solver = SolverRFT(
                model,
                [(i, mesh) for i in range(2 * samples)],
                surface_height=0,
                surface_bottom=-0.118,
                config=SolverRFT.Config(bearing=bearing, velocity_regularization=0.04),
            )
            force = solver.evaluate(state).numpy()[:, 2]
            loading = float(np.trapezoid(force[:samples], depth))
            returned = float(np.trapezoid(force[samples:], depth))
            results.append(
                dict(
                    bearing=bearing,
                    loading_speed=down,
                    unloading_speed=up,
                    loading_input_work_J=loading,
                    unloading_returned_work_J=returned,
                    net_input_work_J=loading - returned,
                )
            )
    kernel = Path(newton.__file__).parent / "_src/solvers/rft/kernels.py"
    record = dict(
        results=results,
        rft_kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(args.geometry_file.read_bytes()).hexdigest(),
        interpretation="Prescribed closed 40 mm depth cycles. Positive input work dissipates energy.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
