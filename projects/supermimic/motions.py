# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Provenance-preserving G1 LAFAN1 CSV conversion and clip-safe reference sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from .math_utils import JOINT_NAMES

LAFAN1_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"


def prepare(source: Path, output: Path, revision: str):
    states, names, starts, lengths, hashes = [], [], [], [], {}
    for path in sorted(source.glob("*.csv")):
        a = np.loadtxt(path, delimiter=",")
        if a.ndim != 2 or a.shape[1] != 36 or len(a) < 3 or not np.isfinite(a).all():
            raise ValueError(f"Invalid G1 LAFAN1 CSV: {path}")
        ti = np.arange(len(a)) / 30
        to = np.arange(int(np.floor(ti[-1] * 50)) + 1) / 50
        pos = np.stack([np.interp(to, ti, a[:, j]) for j in range(3)], -1)
        q = np.stack([np.interp(to, ti, a[:, j]) for j in range(7, 36)], -1)
        rot = Slerp(ti, Rotation.from_quat(a[:, 3:7]))(to)
        quat = rot.as_quat()
        # Continuous signs avoid a discontinuity in the learned orientation input.
        for t in range(1, len(quat)):
            if np.dot(quat[t], quat[t - 1]) < 0:
                quat[t] *= -1
        vel = np.gradient(pos, 0.02, axis=0)
        qd = np.gradient(q, 0.02, axis=0)
        omega = np.zeros_like(pos)
        omega[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / 0.04
        omega[0] = (rot[1] * rot[0].inv()).as_rotvec() / 0.02
        omega[-1] = (rot[-1] * rot[-2].inv()).as_rotvec() / 0.02
        state = np.concatenate((pos, quat, vel, omega, q, qd), -1).astype(np.float32)
        starts.append(sum(lengths))
        lengths.append(len(state))
        names.append(path.stem)
        states.append(state)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not states:
        raise ValueError(f"No CSV motions found in {source}")
    provenance = {
        "dataset": "lvhaidong/LAFAN1_Retargeting_Dataset",
        "revision": revision,
        "source_sha256": hashes,
        "fps": 50,
        "quaternion": "xyzw",
        "joints": JOINT_NAMES,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.concatenate(states),
        starts=starts,
        lengths=lengths,
        names=names,
        provenance=json.dumps(provenance),
    )
    print(json.dumps({"output": str(output), "clips": len(names), "frames": sum(lengths), "provenance": provenance}))


class MotionLibrary:
    """A compact library whose future references never cross a clip boundary."""

    def __init__(self, path, device, clips=()):
        self.path = str(Path(path).resolve())
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as d:
            self.provenance = json.loads(str(d["provenance"]))
            if self.provenance["quaternion"] != "xyzw" or self.provenance["fps"] != 50:
                raise ValueError("Motion library must contain 50 Hz XYZW states")
            if self.provenance["joints"] != JOINT_NAMES:
                raise ValueError("Motion library joint order differs from the G1 contract")
            indices = [i for i, name in enumerate(d["names"]) if not clips or str(name) in clips]
            if not indices or (clips and len(indices) != len(clips)):
                raise ValueError("Selected motion names do not match the library")
            all_states, starts, lengths = d["states"], d["starts"], d["lengths"]
            arrays = [all_states[starts[i] : starts[i] + lengths[i]] for i in indices]
            self.names = [str(d["names"][i]) for i in indices]
        if any(len(a) < 40 or a.shape[1] != 71 or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid or too-short motion data")
        lengths = np.array([len(a) for a in arrays])
        self.states = torch.tensor(np.concatenate(arrays), device=device)
        self.starts = torch.tensor(np.r_[0, np.cumsum(lengths)[:-1]], device=device)
        self.lengths = torch.tensor(lengths, device=device)
        self.ends = self.starts + self.lengths - 1
        self.device = device

    def reference(self, clip, frame):
        frame = frame.clamp_min(0)
        frame = torch.minimum(frame, self.lengths[clip] - 1)
        return self.states[self.starts[clip] + frame]

    def future(self, clip, frame, count=10, stride=5):
        offsets = torch.arange(count, device=self.device) * stride
        return self.reference(clip[..., None], frame[..., None] + offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("projects/supermimic/data/raw/lafan1/g1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=LAFAN1_REVISION)
    parser.add_argument("--download", action="store_true", help="Download the pinned G1 CSVs before converting")
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "lvhaidong/LAFAN1_Retargeting_Dataset",
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=["g1/*.csv", "README.md"],
            local_dir=args.source.parent,
        )
    prepare(args.source, args.output, args.revision)
