# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export the actual finalized G1 collision surfaces for prescribed-motion probes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from isaaclab_newton.physics.rft_solver import collision_surfaces

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.config.g1.granular_env_cfg import G1GranularRFTEnvCfg
from isaaclab_tasks.core.velocity.granular_env import GranularVelocityEnv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cfg = G1GranularRFTEnvCfg()
cfg.scene.num_envs = 1
cfg.randomize_material = False
cfg.curriculum_enabled = False
cfg.sim.visualizer_cfgs = []
with launch_simulation(cfg):
    env = GranularVelocityEnv(cfg)
    try:
        env.reset()
        model = env.rft.model
        surfaces, inventory = collision_surfaces(model, cfg.sim.physics.solver_cfg.body_pattern, 24)
        selected = []
        for (body, mesh), row in zip(surfaces, inventory, strict=True):
            if row["body_label"].endswith(("/left_ankle_roll_link", "/left_knee_link")):
                v = mesh.vertices.astype(float)
                triangles = v[mesh.indices.reshape(-1, 3)]
                volume = float(
                    np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
                )
                assert volume > 0, row
                selected.append(
                    row
                    | dict(
                        vertices=v.tolist(),
                        indices=mesh.indices.tolist(),
                        signed_volume_m3=volume,
                        body_com=model.body_com.numpy()[body].tolist(),
                        bounds_m=[v.min(0).tolist(), v.max(0).tolist()],
                    )
                )
        assert len(selected) == 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            surfaces=selected,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            frame="Authored body-local collision geometry in metres; outward winding verified",
        )
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({row["body_label"]: row["bounds_m"] for row in selected}))
    finally:
        env.close()
