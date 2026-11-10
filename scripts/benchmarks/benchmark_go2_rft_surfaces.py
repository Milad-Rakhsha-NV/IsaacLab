# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surface-load convergence on common prescribed Go2 snapshot kinematics."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from isaaclab_newton.physics.rft_solver import collision_surfaces
from newton.solvers import SolverRFT

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.go2.rft_env import Go2RFTEnv
from isaaclab_tasks.core.velocity.config.go2.rft_env_cfg import Go2RFTEnvCfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", type=Path, default=Path("logs/go2_granular/dr_rft_nominal/rollout.npz"))
parser.add_argument("--output", type=Path, default=Path("logs/go2_granular/report/surface_convergence.json"))
parser.add_argument("--refine_quadrature", action="store_true")
args = parser.parse_args()
cfg = Go2RFTEnvCfg()
cfg.scene.num_envs = 1
cfg.randomize_material = False
cfg.sim.visualizer_cfgs = []
trajectory = np.load(args.trajectory)
poses = trajectory["body_pose"]
indices = [10, 25, 50, 75]
with launch_simulation(cfg):
    env = Go2RFTEnv(cfg)
    try:
        env.reset()
        model = env.rft.model
        com = model.body_com.numpy()
        state = model.state()
        results = []
        for segments in (12, 24, 48):
            surfaces, inventory = collision_surfaces(model, cfg.sim.physics.solver_cfg.body_pattern, segments)
            edge = {12: None, 24: 0.04, 48: 0.02}[segments] if args.refine_quadrature else None
            probe = SolverRFT(
                model,
                surfaces,
                surface_shape_ids=[r["shape"] for r in inventory],
                rigid_solver=env.rft.rigid_solver,
                max_edge_length=edge,
                config=env.rft.config,
            )
            wrenches = []
            for frame in indices:
                before, after = poses[frame - 1], poses[frame]
                twist = np.zeros((model.body_count, 6), dtype=np.float32)
                for body in range(model.body_count):
                    q0, q1 = wp.quat(*before[body, 3:]), wp.quat(*after[body, 3:])
                    c0 = before[body, :3] + np.asarray(wp.quat_rotate(q0, wp.vec3(*com[body])))
                    c1 = after[body, :3] + np.asarray(wp.quat_rotate(q1, wp.vec3(*com[body])))
                    twist[body, :3] = (c1 - c0) / env.step_dt
                    delta = np.asarray(wp.mul(q1, wp.quat_inverse(q0)))
                    if delta[3] < 0:
                        delta = -delta
                    length = np.linalg.norm(delta[:3])
                    if length > 1e-10:
                        twist[body, 3:] = delta[:3] / length * (2 * np.arctan2(length, delta[3]) / env.step_dt)
                state.body_q.assign(after)
                state.body_qd.assign(twist)
                wrenches.append(probe.evaluate(state).numpy().copy())
            results.append(
                dict(
                    segments=segments, max_edge_length=edge, triangles=probe.triangle_count, wrench=np.asarray(wrenches)
                )
            )
        reference = results[-1]["wrench"]
        for result in results:
            w = result.pop("wrench")
            result["force_relative_error"] = float(
                np.linalg.norm(w[:, :, :3] - reference[:, :, :3]) / np.linalg.norm(reference[:, :, :3])
            )
            result["torque_relative_error"] = float(
                np.linalg.norm(w[:, :, 3:] - reference[:, :, 3:]) / np.linalg.norm(reference[:, :, 3:])
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    results=results,
                    reference_segments=48,
                    frames=indices,
                    kinematics="Common poses; COM twists from 20 ms finite differences. No robot integration.",
                    interpretation="Self-convergence of integrated surface loads, not independent physical validation.",
                ),
                indent=2,
            )
        )
        print(json.dumps(results))
    finally:
        env.close()
