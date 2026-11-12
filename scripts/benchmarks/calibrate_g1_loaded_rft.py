# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate corrected RFT using moving tractions and a simulated loaded-foot response.

The short-time loaded response constrains the fit. It does not validate static
support; prolonged holds and timestep refinement are separate acceptance checks.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import newton
import numpy as np
import warp as wp
from investigate_g1_loaded_foot import run as loaded_foot
from newton.solvers import SolverRFT


def main(args):
    """Fit strength for each bearing candidate, then compare loaded settling."""
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.reference / "summary.json").read_text())
    loaded = json.loads((args.loaded_reference / "summary.json").read_text())
    if loaded["backing_floor_proximity"] or loaded["mean_final_speed_m_s"] > 0.01:
        raise ValueError("The loaded reference must settle without touching the backing floor")
    foot = next(
        r
        for r in json.loads(args.geometry_file.read_text())["surfaces"]
        if r["body_label"].endswith("/left_ankle_roll_link")
    )
    mesh = newton.Mesh(
        np.array(foot["vertices"], np.float32), np.array(foot["indices"], np.int32), compute_inertia=False
    )
    with np.load(args.reference / "trace.npz") as d:
        indices = np.arange(0, len(d["statistics"]), 4)
        t = d["statistics"][indices, 0]
        phase = d["phase"][indices]
        kin = d["kinematics"][indices]
        measured = d["mpm_wrench"][indices]
        original = d["rft_wrench"][indices]
    b = newton.ModelBuilder()
    for _ in indices:
        b.add_body(mass=1, inertia=wp.mat33(np.eye(3)))
    model = b.finalize(device="cuda:0")
    solver = SolverRFT(
        model,
        [(i, mesh) for i in range(len(indices))],
        surface_height=0,
        surface_bottom=-reference["settings"]["bed_depth"],
        max_edge_length=0.02,
        config=SolverRFT.Config(internal_friction=0.4, surface_friction=0.12, velocity_regularization=0.04),
    )
    state = model.state()
    state.body_q.assign(kin[:, :7])
    state.body_qd.assign(kin[:, 7:])
    material = solver.material.numpy()
    forces = []
    for beta in (1.0, 2.0):
        material[:, 6] = beta
        solver.material.assign(material)
        forces.append(solver.evaluate(state).numpy().copy())
    flow, pressure = 2 * forces[0] - forces[1], forces[1] - forces[0]
    names = [r["name"] for r in reference["phases"]]
    fit_names = ("insertion", "drag")
    fit = np.isin(phase, [names.index(n) for n in fit_names])
    weight = np.ones(len(t))
    for i in np.unique(phase):
        weight[phase == i] = 1 / np.sqrt((phase == i).sum())
    units = np.array([1, 1, 1, 12.5, 12.5, 12.5])
    target = (measured * units * weight[:, None])[fit].ravel()
    candidates = []
    predictions = []
    kernel = Path(newton.__file__).parent / "_src/solvers/rft/kernels.py"
    for beta in (0.1, 0.2, 0.3, 0.45, 0.6, 0.87574, 1.2):
        basis = flow + beta * pressure
        design = (basis * units * weight[:, None])[fit].ravel()
        strength = float(np.clip(np.dot(design, target) / np.dot(design, design), 0.05, 3.0))
        prediction = strength * basis
        error = float(np.linalg.norm(strength * design - target) / np.linalg.norm(target))
        folder = args.output / f"bearing-{beta:g}"
        settings = dict(
            output=folder,
            geometry_file=args.geometry_file,
            field="rft",
            mass=loaded["settings"]["mass"],
            load=loaded["settings"]["load"],
            seconds=2.0,
            passes=1,
            relaxation=1.0,
            bearing=beta,
            strength_multiplier=strength,
            voxel_size=0.02,
            substeps=4,
            iterations=200,
            device="cuda:0",
            geometry="g1",
            bed_depth=0.118,
            density=1600.0,
            friction=0.4,
            roughness=0.3,
            modulus=1e15,
            domain_scale=1.0,
            settle=0.5,
            particles_per_cell=2,
        )
        loaded_foot(SimpleNamespace(**settings))
        result = json.loads((folder / "summary.json").read_text())
        depth_error = abs(result["final_sole_depth_m"] - loaded["final_sole_depth_m"]) / loaded["final_sole_depth_m"]
        score = error**2 + 4 * depth_error**2 + 0.05 * (result["mean_final_speed_m_s"] / 0.01) ** 2
        score += float(result["backing_floor_proximity"])
        candidates.append(
            dict(
                internal_friction=0.4,
                roughness=0.3,
                regularization=0.04,
                bearing=beta,
                strength_multiplier=strength,
                fit_normalized_error=error,
                loaded_depth_error=depth_error,
                loaded_result=result,
                score=score,
                folder=str(folder),
            )
        )
        predictions.append(prediction)
        (args.output / "search-status.json").write_text(
            json.dumps(dict(completed=len(candidates), total=7, candidates=candidates), indent=2) + "\n"
        )
        gc.collect()
    best = min(range(len(candidates)), key=lambda i: candidates[i]["score"])
    selected = candidates[best]
    prediction = predictions[best]
    rows = []
    for i, name in enumerate(names):
        if name in ("preparation", "released"):
            continue
        mask = phase == i
        mean = measured[mask].mean(0)
        fitted = prediction[mask].mean(0)
        rows.append(
            dict(
                phase=name,
                split="fit" if fit[mask][0] else "heldout",
                measured_wrench=mean.tolist(),
                fitted_wrench=fitted.tolist(),
                original_wrench=original[mask].mean(0).tolist(),
                force_error=float(np.linalg.norm(fitted[:3] - mean[:3]) / max(np.linalg.norm(mean[:3]), 20)),
                torque_error=float(np.linalg.norm(fitted[3:] - mean[3:]) / max(np.linalg.norm(mean[3:]), 1)),
            )
        )
    record = dict(
        selected=selected,
        candidates=candidates,
        reference=str(args.reference),
        loaded_reference=str(args.loaded_reference),
        loaded_measurement=loaded,
        phases=rows,
        fit_phases=fit_names,
        holdout_accepted=all(max(r["force_error"], r["torque_error"]) < 0.2 for r in rows if r["split"] == "heldout"),
        rft_kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        selection=(
            "Moving-wrench NRMSE squared + 4 times loaded-depth relative error squared "
            "+ 0.05 times final-speed/0.01 squared; add 1 for backing-floor contact. "
            "Fixed bulk friction and regularization."
        ),
        interpretation=(
            "Corrected dissipative bearing. Unloaded history remains held out; "
            "matching a short-time loaded depth does not establish static or force equivalence."
        ),
    )
    (args.output / "calibration.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        args.output / "prediction.npz", time=t, measured=measured, original=original, fitted=prediction, phase=phase
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, (ax, label) in enumerate(
        zip(axes.flat, ("Fx [N]", "Fy [N]", "Fz [N]", "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"), strict=True)
    ):
        ax.plot(t, measured[:, j], label="MPM")
        ax.plot(t, original[:, j], label="Original RFT before fix", alpha=0.6)
        ax.plot(t, prediction[:, j], label="Corrected force-informed RFT")
        ax.set(xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / "force-fit.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(selected=selected, holdout_accepted=record["holdout_accepted"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--loaded_reference", type=Path, required=True)
    p.add_argument("--geometry_file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
