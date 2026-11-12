# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit RFT to prescribed G1-foot MPM wrenches and retain independent motion holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverRFT
from scipy.optimize import lsq_linear


def run(args):
    """Select by insertion/hold/drag only; report rotation and unloading separately."""
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.reference / "summary.json").read_text())
    geometry = json.loads(args.geometry_file.read_text())
    foot = next(s for s in geometry["surfaces"] if s["body_label"].endswith("/left_ankle_roll_link"))
    mesh = newton.Mesh(
        np.array(foot["vertices"], np.float32), np.array(foot["indices"], np.int32), compute_inertia=False
    )
    with np.load(args.reference / "trace.npz") as data:
        index = np.arange(0, len(data["statistics"]), 4)
        t = data["statistics"][index, 0]
        kinematics = data["kinematics"][index]
        phase = data["phase"][index]
        measured = data["mpm_wrench"][index]
        original = data["rft_wrench"][index]
    builder = newton.ModelBuilder()
    for _ in index:
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device="cuda:0")
    rft = SolverRFT(
        model,
        [(i, mesh) for i in range(len(index))],
        surface_height=0,
        surface_bottom=-summary["settings"]["bed_depth"],
        max_edge_length=0.02,
    )
    state = model.state()
    state.body_q.assign(kinematics[:, :7])
    state.body_qd.assign(kinematics[:, 7:])
    material = rft.material.numpy()
    settings = summary["settings"]

    def evaluate(mu, roughness, epsilon, bearing):
        material[:, 0] = settings["density"]
        material[:, 1] = mu
        material[:, 2] = roughness * mu
        material[:, 3] = settings["density"] * 9.81 * (894 * mu**3 - 386 * mu**2 + 89 * mu)
        material[:, 5] = epsilon
        material[:, 6] = bearing
        rft.material.assign(material)
        return rft.evaluate(state).numpy().copy()

    names = [p["name"] for p in summary["phases"]]
    train = np.isin(phase, [names.index(p) for p in ("insertion", "hold", "drag")])
    heldout = np.isin(phase, [names.index(p) for p in ("drag_hold", "pitch", "roll", "withdrawal")])
    units = np.array([1, 1, 1, 1 / 0.08, 1 / 0.08, 1 / 0.08])
    weight = np.ones(len(t))
    for p in np.unique(phase):
        weight[phase == p] = 1 / np.sqrt((phase == p).sum())
    target = (measured * units * weight[:, None])[train].ravel()
    candidates = []
    predictions = []
    for mu in (0.25, 0.3, 0.4, 0.5):
        for roughness in (0.3, 0.6, 1.0):
            for epsilon in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
                pure = evaluate(mu, roughness, epsilon, 0)
                one, two = evaluate(mu, roughness, epsilon, 1), evaluate(mu, roughness, epsilon, 2)
                flow, bearing = 2 * one - two, two - one
                for family in ("flowing", "bearing"):
                    basis = [pure] if family == "flowing" else [flow, bearing]
                    design = np.stack([(b * units * weight[:, None])[train].ravel() for b in basis], axis=1)
                    result = lsq_linear(design, target, bounds=(1e-6, np.inf), tol=1e-10)
                    coefficients = result.x
                    strength = float(coefficients[0])
                    beta = 0.0 if family == "flowing" else max(float(coefficients[1] / strength), 1e-6)
                    prediction = strength * (pure if family == "flowing" else flow + beta * bearing)
                    fit = float(
                        np.linalg.norm((prediction * units * weight[:, None])[train].ravel() - target)
                        / np.linalg.norm(target)
                    )
                    error = float(
                        np.linalg.norm((prediction - measured)[heldout, :3])
                        / max(np.linalg.norm(measured[heldout, :3]), 1)
                    )
                    candidates.append(
                        dict(
                            internal_friction=mu,
                            roughness=roughness,
                            regularization=epsilon,
                            bearing=beta,
                            strength_multiplier=strength,
                            family=family,
                            fit_normalized_error=fit,
                            heldout_force_relative_l2=error,
                        )
                    )
                    predictions.append(prediction)
    unconstrained = min(range(len(candidates)), key=lambda i: candidates[i]["fit_normalized_error"])
    comparable = [
        i
        for i, candidate in enumerate(candidates)
        if candidate["fit_normalized_error"] <= 1.01 * candidates[unconstrained]["fit_normalized_error"]
        and candidate["internal_friction"] == settings["friction"]
        and candidate["regularization"] == settings["regularization"]
    ]
    # Keep known friction and the original regularizer when the fitting objective
    # cannot distinguish them. No held-out measurement participates in selection.
    flowing = [i for i in comparable if candidates[i]["family"] == "flowing"]
    best = (
        min(flowing or comparable, key=lambda i: candidates[i]["fit_normalized_error"]) if comparable else unconstrained
    )
    prediction = predictions[best]
    phase_rows = []
    for i, name in enumerate(names):
        if name in ("preparation", "released"):
            continue
        mask = phase == i
        mean = measured[mask].mean(0)
        fit = prediction[mask].mean(0)
        phase_rows.append(
            dict(
                phase=name,
                split="fit" if train[mask][0] else "heldout",
                measured_wrench=mean.tolist(),
                fitted_wrench=fit.tolist(),
                original_wrench=original[mask].mean(0).tolist(),
                force_error=float(np.linalg.norm(fit[:3] - mean[:3]) / max(np.linalg.norm(mean[:3]), 20)),
                torque_error=float(np.linalg.norm(fit[3:] - mean[3:]) / max(np.linalg.norm(mean[3:]), 1)),
            )
        )
    record = dict(
        reference=str(args.reference),
        selected=candidates[best],
        unconstrained_best=candidates[unconstrained],
        selection=(
            "Within 1% of the best training objective, retain known internal friction and the original regularizer "
            "when available; prefer zero bearing among comparable fits."
        ),
        candidates=candidates,
        phases=phase_rows,
        fit_phases=["insertion", "hold", "drag"],
        heldout_phases=["drag_hold", "pitch", "roll", "withdrawal"],
        holdout_target=0.2,
        holdout_accepted=all(
            max(row["force_error"], row["torque_error"]) < 0.2 for row in phase_rows if row["split"] == "heldout"
        ),
        interpretation="Fit to an MPM numerical fixture, not physical sand. Reference convergence is a separate gate.",
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
        ax.plot(t, original[:, j], label="Original RFT", alpha=0.65)
        ax.plot(t, prediction[:, j], label="Fitted RFT")
        ax.set(xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("G1 foot calibration: fit insertion/hold/drag; hold out rotation and withdrawal")
    fig.tight_layout()
    fig.savefig(args.output / "force-fit.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(selected=candidates[best], holdout_accepted=record["holdout_accepted"], phases=phase_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--geometry_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
