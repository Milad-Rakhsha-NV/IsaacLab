# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize numerical sensitivity, held-out force matching and frozen Go2 policy comparisons."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverRFT
from scipy.spatial.transform import Rotation

ROOT = Path("logs/go2_granular/investigation")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def records():
    """Load completed probes, including held-out rate and withdrawal cases."""
    return {
        path.parent.name: dict(json.loads(path.read_text()), folder=str(path.parent))
        for category in ("numerics", "transfer", "calibration", "tolerance", "rate")
        for path in (ROOT / category).glob("*/summary.json")
    }


def force_component(record):
    """Return the resisting component used by the convergence gate."""
    axis = 2 if record["settings"]["mode"] == "vertical" else 0
    return abs(record["mean_force_N"][axis])


def compare(data, first, second, limit=0.10):
    """Compare a refinement against its finer reference without changing the target."""
    if first not in data or second not in data:
        return None
    a, b = force_component(data[first]), force_component(data[second])
    relative = abs(a - b) / max(b, 0.05)
    return dict(first=first, reference=second, relative_change=relative, target=limit, passed=relative <= limit)


def numerical_report(data):
    """Plot independent spatial and temporal refinements and report their gates."""
    checks = {}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for col, mode in enumerate(("vertical", "drag")):
        for row, shape in ((0, "sphere"), (2, "plate")):
            for scheme, grids, naming in (
                ("PIC", (0.02, 0.01, 0.005), lambda g: f"{shape}-{mode}-grid-{g}"),
                (
                    "APIC",
                    (0.01, 0.005),
                    lambda g: f"sphere-{mode}-apic-time-16"
                    if shape == "sphere" and g == 0.01
                    else f"sphere-{mode}-apic-fine"
                    if shape == "sphere"
                    else f"plate-{mode}-apic-grid-{g}",
                ),
            ):
                available = [(1000 * g, data[naming(g)]) for g in grids if naming(g) in data]
                if available:
                    axes[row, col].plot(
                        [x for x, _ in available], [force_component(v) for _, v in available], "o-", label=scheme
                    )
            axes[row, col].set(
                title=f"{shape.capitalize()}: {mode}", xlabel="MPM grid spacing [mm]", ylabel="Mean force [N]"
            )
            axes[row, col].invert_xaxis()
            axes[row, col].legend()
        for scheme, values, naming in (
            ("PIC", (4, 8, 16), lambda s: f"sphere-{mode}-grid-0.01" if s == 4 else f"sphere-{mode}-time-{s}"),
            ("APIC", (8, 16, 32), lambda s: f"sphere-{mode}-apic-time-{s}"),
        ):
            available = [(5 / s, data[naming(s)]) for s in values if naming(s) in data]
            if available:
                axes[1, col].plot(
                    [x for x, _ in available], [force_component(v) for _, v in available], "o-", label=scheme
                )
        axes[1, col].set(title=f"Sphere: {mode}, 10 mm grid", xlabel="Solver step [ms]", ylabel="Mean force [N]")
        axes[1, col].invert_xaxis()
        axes[1, col].legend()
        for shape in ("sphere", "plate"):
            checks[f"{shape}_{mode}_pic_grid"] = compare(
                data, f"{shape}-{mode}-grid-0.01", f"{shape}-{mode}-grid-0.005"
            )
            checks[f"{shape}_{mode}_apic_grid"] = compare(
                data,
                f"sphere-{mode}-apic-time-16" if shape == "sphere" else f"plate-{mode}-apic-grid-0.01",
                f"sphere-{mode}-apic-fine" if shape == "sphere" else f"plate-{mode}-apic-grid-0.005",
            )
        checks[f"{mode}_pic_time"] = compare(data, f"sphere-{mode}-time-8", f"sphere-{mode}-time-16")
        checks[f"{mode}_apic_time"] = compare(data, f"sphere-{mode}-apic-time-16", f"sphere-{mode}-apic-time-32")
        checks[f"{mode}_iterations"] = compare(data, f"sphere-{mode}-iterations-100", f"sphere-{mode}-iterations-200")
        checks[f"{mode}_domain"] = compare(data, f"sphere-{mode}-iterations-200", f"sphere-{mode}-domain")
        checks[f"{mode}_projection"] = compare(data, f"sphere-{mode}-iterations-200", f"sphere-{mode}-projection")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("MPM force sensitivity: sphere diameter 44 mm; plate width 50 mm")
    fig.tight_layout()
    fig.savefig(ROOT / "convergence.png", dpi=150)
    plt.close(fig)
    checks["vertical_tolerance"] = compare(data, "sphere-vertical-apic-time-16", "sphere-vertical-apic-tight")
    return checks


def measured_rows(record):
    """Select loading or steady-drag frames independently of the force values."""
    with np.load(Path(record["folder"]) / "trace.npz") as data:
        trace = data["trace"]
    c = record["settings"]
    insertion = (c["depth"] + 0.01) / (0.1 if c["mode"] == "drag" else c["speed"])
    end = c["settle"] + insertion
    if c["mode"] == "drag":
        mask = (trace[:, 0] > end + 0.1 + 0.01 / c["speed"]) & (trace[:, 0] < end + 0.1 + 0.04 / c["speed"] - 0.005)
    else:
        penetration = (0.022 if c["geometry"] == "sphere" else 0.0025) - trace[:, 3]
        mask = (penetration > 0.005) & (trace[:, 0] < end)
    return trace[np.flatnonzero(mask)[::3]]


def candidate_forces(record):
    """Batch actual training-surface RFT predictions over independent candidate worlds."""
    c = record["settings"]
    trace = measured_rows(record)
    candidates = np.array([(mu, ratio) for mu in np.linspace(0.3, 0.5, 9) for ratio in np.linspace(0.6, 1.0, 5)])
    builder = newton.ModelBuilder()
    n = len(trace)
    # Average over the same solver poses as the 5 ms MPM impulse interval.
    # Horizontal translation leaves RFT unchanged in this uniform field.
    samples = c["substeps"] if c["mode"] == "vertical" else 1
    for mu, ratio in candidates:
        builder.begin_world()
        for _ in range(n * samples):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.001))
        builder.end_world()
    model = builder.finalize(device="cuda:0")
    mesh = (
        newton.Mesh.create_sphere(0.022, num_latitudes=6, num_longitudes=12, compute_inertia=False)
        if c["geometry"] == "sphere"
        else newton.Mesh.create_box(0.025, 0.025, 0.0025, compute_inertia=False)
    )
    solver = SolverRFT(
        model,
        [(i, mesh) for i in range(model.body_count)],
        surface_bottom=-c["bed_depth"],
        config=SolverRFT.Config(velocity_regularization=0.04, bearing=0.87574),
    )
    material = solver.material.numpy()
    for world, (mu, ratio) in enumerate(candidates):
        material[world, :4] = [c["density"], mu, mu * ratio, -1]
    solver.material.assign(material)
    q = np.c_[trace[:, 1:4], np.tile([0, 0, 0, 1], (n, 1))]
    v = np.zeros((n, 6), dtype=np.float32)
    t = trace[:, 0] - 0.005 / c["substeps"]
    insertion_speed = 0.1 if c["mode"] == "drag" else c["speed"]
    insertion_end = c["settle"] + (c["depth"] + 0.01) / insertion_speed
    v[:, 2] = np.where((t >= c["settle"]) & (t < insertion_end), -insertion_speed, 0)
    if c["mode"] == "drag":
        start = insertion_end + 0.1
        v[:, 0] = np.where((t >= start) & (t < start + 0.04 / c["speed"]), c["speed"], 0)
    q = np.repeat(q, samples, axis=0)
    v = np.repeat(v, samples, axis=0)
    offsets = np.tile((np.arange(samples) - samples + 1) * 0.005 / c["substeps"], n)
    q[:, :3] += v[:, :3] * offsets[:, None]
    state = model.state()
    state.body_q.assign(np.tile(q, (len(candidates), 1)).astype(np.float32))
    state.body_qd.assign(np.tile(v, (len(candidates), 1)))
    force = solver.evaluate(state).numpy()[:, :3].reshape(len(candidates), n, samples, 3).mean(axis=2)
    return dict(trace=trace, candidates=candidates, forces=force)


def bounded_force_fit(predictions):
    """Fit a single amplitude across equally weighted penetration and drag traces."""
    lhs, rhs = [], []
    for p in predictions.values():
        observed = p["trace"][:, 4:7]
        scale = max(float(np.linalg.norm(observed)), 0.05)
        lhs.append(p["forces"] / scale)
        rhs.append(observed / scale)
    a, b = np.concatenate(lhs, axis=1), np.concatenate(rhs, axis=0)
    optimum = np.sum(a * b[None], axis=(1, 2)) / np.maximum(np.sum(a * a, axis=(1, 2)), 1e-12)
    bounded = np.clip(optimum, 0.8, 1.2)
    error = np.sqrt(np.sum((a * bounded[:, None, None] - b[None]) ** 2, axis=(1, 2)) / len(predictions))
    return int(np.argmin(error)), bounded, optimum, error


def fit_response(data):
    """Fit nominal sphere loading and drag, then evaluate held-out responses."""
    names = ("sphere-vertical-apic-time-16", "sphere-drag-apic-time-16")
    if not all(name in data for name in names):
        return None
    predictions = {name: candidate_forces(data[name]) for name in names}
    best, bounded, optimum, error = bounded_force_fit(predictions)
    mu, ratio = next(iter(predictions.values()))["candidates"][best]
    result = dict(
        fitted_on=list(names),
        density=1600,
        internal_friction=float(mu),
        roughness_ratio=float(ratio),
        strength_multiplier=float(bounded[best]),
        unconstrained_strength_multiplier=float(optimum[best]),
        fit_normalized_rms=float(error[best]),
        candidate_count=len(error),
        held_out=[],
        interpretation="Provisional force matching within the RFT training bounds; MPM numerical convergence "
        "is checked separately.",
    )
    candidate_table = next(iter(predictions.values()))["candidates"]
    nominal_index = int(np.argmin(np.linalg.norm(candidate_table - [0.4, 1.0], axis=1)))
    result["same_material_unconstrained_strength"] = float(optimum[nominal_index])
    result["same_material_bounded_fit_error"] = float(error[nominal_index])
    result["material_fits"] = []
    for label in ("lower-material", "higher-material"):
        pair = [f"sphere-{mode}-{label}" for mode in ("vertical", "drag")]
        if all(name in data for name in pair):
            values = {name: candidate_forces(data[name]) for name in pair}
            selected, amplitude, free_amplitude, errors = bounded_force_fit(values)
            mu_fit, ratio_fit = candidate_table[selected]
            c = data[pair[0]]["settings"]
            result["material_fits"].append(
                dict(
                    material=label,
                    mpm_density=c["density"],
                    mpm_friction=c["friction"],
                    mpm_roughness=c["roughness"],
                    bed_depth=c["bed_depth"],
                    rft_friction=float(mu_fit),
                    rft_roughness=float(ratio_fit),
                    rft_strength=float(amplitude[selected]),
                    unconstrained_strength=float(free_amplitude[selected]),
                    fit_normalized_rms=float(errors[selected]),
                )
            )
    heldout = [
        name
        for name, r in data.items()
        if r["settings"].get("transfer_scheme") == "apic"
        and r["settings"]["density"] == 1600
        and ("shallow-slow" in name or "deep-fast" in name or name.startswith("plate-") and "0.01" in name)
    ]
    for name in heldout:
        predictions[name] = candidate_forces(data[name])
    plot_names = [
        *names,
        "plate-vertical-apic-grid-0.01",
        "plate-drag-apic-grid-0.01",
        "sphere-vertical-shallow-slow",
        "sphere-drag-shallow-slow",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for name, p in predictions.items():
        observed = p["trace"][:, 4:7]
        fitted = p["forces"][best] * bounded[best]
        nominal = p["forces"][np.argmin(np.linalg.norm(p["candidates"] - [0.4, 1.0], axis=1))]
        scale = max(float(np.linalg.norm(observed)), 0.05)
        entry = dict(
            case=name,
            fitted_relative_l2=float(np.linalg.norm(fitted - observed) / scale),
            nominal_relative_l2=float(np.linalg.norm(nominal - observed) / scale),
            mean_mpm_N=observed.mean(0).tolist(),
            mean_fitted_N=fitted.mean(0).tolist(),
        )
        if name not in names:
            result["held_out"].append(entry)
        if name in plot_names:
            ax = axes.flat[plot_names.index(name)]
            component = 2 if data[name]["settings"]["mode"] == "vertical" else 0
            ax.plot(p["trace"][:, 0], observed[:, component], label="MPM")
            ax.plot(p["trace"][:, 0], nominal[:, component], label="Nominal RFT")
            ax.plot(p["trace"][:, 0], fitted[:, component], label="Fitted RFT")
            c = data[name]["settings"]
            ax.set(
                title=f"{c['geometry']} {c['mode']}: depth {1000 * c['depth']:g} mm, speed {c['speed']:g} m/s",
                xlabel="Simulation time [s]",
                ylabel="Force [N]",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    for index, name in enumerate(plot_names):
        if name not in predictions:
            axes.flat[index].text(0.5, 0.5, "Measurement pending", ha="center", transform=axes.flat[index].transAxes)
            axes.flat[index].set_axis_off()
    fig.suptitle("Provisional force matching: APIC, 10 mm grid, 0.3125 ms solver step")
    fig.tight_layout()
    fig.savefig(ROOT / "force-response.png", dpi=150)
    plt.close(fig)
    return result


def rate_report(data):
    """Show rate sensitivity at fixed depth and a complete load/unload cycle."""
    if "sphere-vertical-unloading" not in data:
        return {}
    result = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for column, mode in enumerate(("vertical", "drag")):
        names = [f"sphere-{mode}-rate-0.05", f"sphere-{mode}-apic-time-16", f"sphere-{mode}-rate-0.2"]
        available = [data[name] for name in names if name in data]
        speed = [case["settings"]["speed"] for case in available]
        forces = [force_component(case) for case in available]
        axes[column].plot(speed, forces, "o-", label="MPM")
        axis = 2 if mode == "vertical" else 0
        axes[column].plot(speed, [abs(case["mean_rft_force_N"][axis]) for case in available], "s--", label="RFT")
        axes[column].set(title=f"Sphere {mode}, 40 mm depth", xlabel="Probe speed [m/s]", ylabel="Mean force [N]")
        axes[column].legend()
        result[mode] = dict(speeds_m_s=speed, mpm_forces_N=forces)
    record = data.get("sphere-vertical-unloading")
    if record:
        trace = np.load(Path(record["folder"]) / "trace.npz")["trace"]
        c = record["settings"]
        end = c["settle"] + (c["depth"] + 0.01) / c["speed"]
        penetration = 0.022 - trace[:, 3]
        for mask, label, style in (
            (trace[:, 0] < end, "MPM insertion", "-"),
            (trace[:, 0] > end + 0.1, "MPM withdrawal", "--"),
        ):
            axes[2].plot(penetration[mask] * 1000, trace[mask, 6], style, label=label)
        hold = (trace[:, 0] > end + 0.03) & (trace[:, 0] < end + 0.09)
        result["cycle"] = dict(
            hold_mpm_vertical_N=float(trace[hold, 6].mean()),
            hold_rft_vertical_N=float(trace[hold, 9].mean()),
            mpm_cycle_input_work_J=float(-np.trapezoid(trace[:, 6], trace[:, 3])),
            rft_cycle_input_work_J=float(-np.trapezoid(trace[:, 9], trace[:, 3])),
        )
    axes[2].set(title="MPM insertion / withdrawal", xlabel="Sole penetration [mm]", ylabel="Vertical force [N]")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "rate-and-history.png", dpi=150)
    plt.close(fig)
    return result


def foot_geometry(phase, row, *, data_root=ROOT):
    """Measure the task's 10 mm lateral-foot margin from recorded body poses."""
    folder = data_root / phase / row["name"]
    with np.load(folder / "rollout.npz") as data:
        poses = data["body_pose"]
    labels = row["summary"]["body_labels"]
    summary = row["summary"]
    root_name = summary.get("root_body_name", "base")
    foot_names = summary.get("foot_body_names", [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")])
    sides = summary.get("foot_side_signs", [1, -1, 1, -1])
    root = next(i for i, name in enumerate(labels) if name.endswith(f"/{root_name}"))
    feet = [next(i for i, name in enumerate(labels) if name.endswith(f"/{foot}")) for foot in foot_names]
    offsets = poses[:, feet, :3] - poses[:, root, None, :3]
    orientation = Rotation.from_quat(np.repeat(poses[:, root, 3:7], len(feet), axis=0))
    local = orientation.inv().apply(offsets.reshape(-1, 3)).reshape(-1, len(feet), 3)
    signed_y = local[:, :, 1] * sides
    minimum = float(signed_y.min())
    recorded = row["summary"].get("min_signed_foot_y")
    if recorded is not None:
        np.testing.assert_allclose(minimum, recorded, atol=1e-5)
    return dict(minimum_signed_foot_y_m=minimum, foot_placement_pass=minimum >= 0.01)


def policy_report(policy_root=ROOT, output_root=ROOT, *, speed=0.5):
    """Read frozen-policy studies and quantify tracking and load support."""
    result = {}
    for phase in ("coupling", "materials", "reference"):
        path = policy_root / f"{phase}-status.json"
        result[phase] = json.loads(path.read_text())["results"] if path.exists() else []
        if phase in ("materials", "reference"):
            for row in result[phase]:
                row["foot_geometry"] = foot_geometry(phase, row, data_root=policy_root)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    labels = ("lower-density-friction", "nominal", "higher-density-friction")
    x = np.arange(3)
    for policy, offset, label in (("rigid", -0.18, "Rigid-trained / MPM"), ("rft", 0.18, "RFT-trained / MPM")):
        values = {row["name"]: row["summary"] for row in result["materials"] if row["policy"] == policy}
        axes[0].bar(
            x + offset,
            [values.get(f"{name}-{policy}", {}).get("mean_velocity_error", np.nan) for name in labels],
            width=0.35,
            label=label,
        )
    reference = {row["name"]: row["summary"] for row in result["reference"]}
    axes[0].plot(
        x,
        [reference.get(f"{name}-rft", {}).get("mean_velocity_error", np.nan) for name in labels],
        "ko--",
        label="RFT-trained / RFT",
    )
    axes[0].set(
        xticks=x,
        xticklabels=("Lower density / friction", "Nominal", "Higher density / friction"),
        ylabel="Mean planar tracking error [m/s]",
        title=f"Frozen policies: {speed:.2f} m/s command, 3 s",
    )
    axes[0].legend(fontsize=8)
    rows = result["materials"]
    positions = np.arange(len(rows))
    granular = np.array([row["summary"].get("mean_external_vertical_N", np.nan) for row in rows])
    rigid = np.array([row["summary"].get("mean_rigid_contact_vertical_N", np.nan) for row in rows])
    axes[1].bar(positions, granular, label="Granular coupling")
    axes[1].bar(positions, rigid, bottom=granular, label="Rigid contacts")
    weight = rows[0]["summary"].get("robot_weight_N", 147.336) if rows else 147.336
    axes[1].axhline(weight, color="k", ls="--", lw=1, label="Robot weight")
    axes[1].set(
        xticks=positions,
        xticklabels=[f"{row['name'].split('-')[0].title()}\n{row['policy'].upper()}" for row in rows],
        ylabel="Mean upward force [N]",
        title="Support after the first 0.5 s",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_root / "policy-and-support.png", dpi=150)
    plt.close(fig)
    if result["coupling"]:
        fig, ax = plt.subplots(figsize=(11, 4))
        labels = []
        for row in result["coupling"]:
            c = row["settings"]
            labels.append(
                f"α={c.get('proxy_relaxation', 0.1):g}, passes={c.get('proxy_iterations', 1)}\n"
                f"substeps={c.get('substeps', 4)}"
            )
        ax.bar(np.arange(len(labels)), [row["summary"]["mean_velocity_error"] for row in result["coupling"]])
        ax.set(
            xticks=np.arange(len(labels)),
            xticklabels=labels,
            ylabel="Mean planar error [m/s]",
            title=f"Coupling sensitivity: same frozen RFT policy, 2 s at {speed:.2f} m/s",
        )
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_root / "coupling.png", dpi=150)
        plt.close(fig)
    return result


def main(final=False):
    """Publish numerical gates and exploratory comparisons with explicit limits."""
    data = records()
    policy_video_href = (
        "../previous_0p5_report/" if (ROOT.parent / "previous_0p5_report/index.html").exists() else "../report/"
    )
    checks = numerical_report(data)
    fit = fit_response(data)
    rate = rate_report(data)
    policies = policy_report()
    required = [
        checks.get(f"{shape}_{mode}_apic_grid") for shape in ("sphere", "plate") for mode in ("vertical", "drag")
    ]
    required += [checks.get(f"{mode}_apic_time") for mode in ("vertical", "drag")]
    accepted = all(c is not None and c["passed"] for c in required)
    quality = {
        name: max(row[name] for row in data.values())
        for name in ("maximum_floor_penetration_m", "maximum_probe_penetration_m", "momentum_balance_mean_relative")
    }
    force_error_gate_passed = bool(
        fit
        and len(fit["held_out"]) == 6
        and fit["fit_normalized_rms"] <= 0.20
        and all(row["fitted_relative_l2"] <= 0.20 for row in fit["held_out"])
    )
    result = dict(
        case_count=len(data),
        convergence_checks=checks,
        mpm_reference_accepted=accepted,
        force_fit=fit,
        force_fit_accepted=accepted and force_error_gate_passed,
        force_error_gate_passed=force_error_gate_passed,
        force_error_target=0.20,
        rate_and_history=rate,
        quality_worst_cases=quality,
        frozen_policy_studies=policies,
        completion=dict(
            probes=len(data),
            probes_planned=50,
            coupling=len(policies["coupling"]),
            coupling_planned=8,
            materials=len(policies["materials"]),
            materials_planned=6,
        ),
    )
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2))
    (ROOT / "probe-records.json").write_text(json.dumps(data, indent=2))
    rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' '))}</td><td>{100 * c['relative_change']:.1f}%</td>"
        f"<td>{'Pass' if c['passed'] else 'Fail'}</td></tr>"
        for name, c in checks.items()
        if c is not None
    )
    fit_text = "Force matching awaits the nominal APIC penetration and drag records."
    if fit:
        fit_text = (
            f"Provisional bounded fit: density {fit['density']} kg/m³, "
            f"internal friction {fit['internal_friction']:.3f}, surface/internal ratio {fit['roughness_ratio']:.2f}, "
            f"strength multiplier {fit['strength_multiplier']:.3f}. "
            f"Fit normalized RMS error: {100 * fit['fit_normalized_rms']:.1f}%."
        )
    heldout_rows = []
    for row in fit["held_out"] if fit else []:
        c = data[row["case"]]["settings"]
        shape = "Foot sphere" if c["geometry"] == "sphere" else "Plate"
        motion = "penetration" if c["mode"] == "vertical" else "horizontal drag"
        label = f"{shape}: {motion}, depth {1000 * c['depth']:g} mm, speed {c['speed']:g} m/s"
        heldout_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{100 * row['nominal_relative_l2']:.1f}%</td>"
            f"<td>{100 * row['fitted_relative_l2']:.1f}%</td></tr>"
        )
    material_rows = []
    for row in policies["materials"]:
        failure = row["summary"]["first_failure_seconds"]
        failure_text = "None in 3 s" if failure is None else f"{failure:.2f} s"
        material_rows.append(
            f"<tr><td>{html.escape(row['name'].rsplit('-', 1)[0].replace('-', ' '))}</td>"
            f"<td>{'Rigid-ground policy' if row['policy'] == 'rigid' else 'RFT-trained policy'}</td>"
            f"<td>{row['summary']['mean_velocity_error']:.3f} m/s</td><td>{failure_text}</td>"
            f"<td>{1000 * row['foot_geometry']['minimum_signed_foot_y_m']:.1f} mm</td></tr>"
        )
    page = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:25px auto;padding:20px;background:#f1f5f9;"
        "color:#0f172a}img{width:100%}table{border-collapse:collapse;background:white}"
        "th,td{padding:8px;border:1px solid #aaa}"
        ".status{padding:16px;background:#fff7ed}a{color:#1d4ed8}</style><h1>Go2: material-model investigation</h1>"
        f'<p><a href="{policy_video_href}">Original 0.5 m/s policy videos</a> · '
        '<a href="coupling.html">Coupling sensitivity</a> · '
        '<a href="materials.html">Material-bank policy evaluation</a></p>'
        f"<p class='status'>Completed: {len(data)}/50 force probes, "
        f"{len(policies['materials'])}/6 material-policy trials, "
        f"{len(policies['coupling'])}/8 coupling trials. "
        "Pending cases are excluded from numerical results.</p>"
        "<p id='live-progress'>Live measurement status is loading.</p>"
        f"<p class='status'><strong>MPM numerical reference accepted: {'yes' if accepted else 'no'}.</strong> "
        "The target is ≤10% change in mean force under the final tested grid and timestep "
        "refinements for both probes.</p>"
        "<p>The original videos used PIC transfer, a 20 mm grid, four substeps and one "
        "coupling pass with relaxation 0.1. "
        "These results investigate numerical sensitivity before treating that MPM setup as a material reference.</p>"
        "<h2>Numerical convergence</h2><p>Sphere diameter 44 mm; plate width 50 mm and thickness 5 mm. "
        "Prepared bed depth 118 mm, fixed bottom, frictionless container walls; domain-size "
        "and projection checks are included. "
        "Grid series hold timestep fixed within each curve; PIC/APIC curves can use different timesteps. "
        "Vertical force is averaged during insertion; drag is averaged after the first 10 mm of horizontal travel.</p>"
        '<img src="convergence.png" alt="MPM grid and timestep sensitivity">'
        "<table><tr><th>Check</th><th>Change relative to finer case</th><th>10% target</th></tr>" + rows + "</table>"
        "<p>The stress-solve tolerance check increases the cap to 500 iterations and tightens tolerance to 10⁻⁶. "
        "Momentum closure and submillimetre contact checks help detect implementation errors; "
        "they do not establish force convergence.</p>"
        f"<p>Across {len(data)} probes: largest floor penetration "
        f"{1000 * quality['maximum_floor_penetration_m']:.3f} mm; "
        f"largest probe penetration {1000 * quality['maximum_probe_penetration_m']:.3f} mm; "
        "worst case-mean momentum residual "
        f"{100 * quality['momentum_balance_mean_relative']:.4f}% of the bed gravitational impulse per frame.</p>"
        "<h2>Material-response matching</h2><p>" + fit_text + "</p>"
        "<p>The fit uses only the nominal sphere penetration and drag records. Other depths, "
        "speeds and the plate are held out. "
        "The RFT candidate grid covers friction 0.3–0.5 and surface/internal ratios 0.6–1.0, "
        "with strength constrained to 0.8–1.2. "
        "The actual training sphere quadrature is used: 6 latitudes and 12 longitudes. "
        "Density stays matched at 1,600 kg/m³. "
        "Loading and drag have equal weight after each trace is normalized by its measured force norm. "
        "The exploratory force-error target is 20% on the fitted and every held-out trace. "
        "A successful fit also requires an accepted numerical reference. No fitted values "
        "have been applied to either policy.</p>"
        + ('<img src="force-response.png" alt="Measured MPM versus nominal and fitted RFT forces">' if fit else "")
        + "<table><tr><th>Held-out force trace</th><th>Nominal RFT vector error</th><th>Fitted "
        "RFT vector error</th></tr>"
        + "".join(heldout_rows)
        + "</table><p>Errors are the L2 norm of the three-component force residual over the trace, "
        "divided by the measured MPM trace norm. Drag errors include the vertical load component.</p>"
        + "<h2>Rate and material history</h2><p>Speed changes hold the maximum penetration fixed at 40 mm. "
        "MPM includes inertia and evolving deformation; the RFT training closure has no "
        "inertial term and uses a fixed free surface. "
        "Its 0.04 m/s regularization and experimental bearing term still affect near-rest forces. "
        "The probe is held kinematically, without an imposed robot-weight load. "
        "Rate differences can include numerical effects.</p>"
        + (
            '<img src="rate-and-history.png" alt="Fixed-depth rate sensitivity and loading/withdrawal force loop">'
            if rate
            else "<p>Rate and withdrawal measurements are pending.</p>"
        )
        + (
            f"<p>During the final part of the fixed-depth hold, this prescribed-foot test gives "
            f"{rate['cycle']['hold_mpm_vertical_N']:.3f} N from MPM versus "
            f"{rate['cycle']['hold_rft_vertical_N']:.2f} N from the nominal RFT bearing closure. "
            "The models therefore also differ in their near-rest response.</p>"
            if "cycle" in rate
            else ""
        )
        + "<h2>Parameter correspondence</h2><p>Density, initial depth and the internal friction "
        "invariant agree as inputs. "
        "A direct constitutive check gives τ = 399.99997 Pa at P = 1,000 Pa and μ = 0.4, using "
        "τ = √(s:s/2). MPM collider friction is μ times the same surface/internal ratio used by RFT. "
        "This establishes the meaning of μ, rather than equivalence of the complete material response.</p>"
        "<p>The MPM bed uses E = 10¹⁵ Pa, ν = 0.3, compressive cap 10¹⁵ Pa, zero cohesion/tensile strength, "
        "viscosity, hardening and dilatancy. These settings approximate a stiff frictional continuum. "
        "RFT has no equivalent elastic stiffness, stress history or packing state; its 0.8–1.2 strength factor "
        "and bearing closure have no direct MPM input counterpart.</p>"
        "<h2>Frozen-policy material comparison</h2><p>Both policies are evaluated for 3 s, "
        "with the same 0.50 m/s forward "
        "command and seed 123. These are short parameter-sensitivity trials, not success-rate estimates. "
        "Lower settings: 1,400 kg/m³, μ=0.3, ratio=0.6, depth=60 mm. Nominal: 1,600, 0.4, 1.0, 118 mm. "
        "Higher settings: 1,800, 0.5, 0.8, 100 mm. Each lies within the RFT reset-randomization ranges; "
        "three selected settings do not cover their full joint distribution. "
        "The RFT comparison uses strength factor 1.0.</p>"
        "<p>The frozen rigid policy was trained for 600 iterations and the RFT policy for 1,500, "
        "both with 4,096 environments and one training seed. These different budgets prevent a controlled "
        "causal attribution to material training alone.</p>"
        "<p>Policy trials retain the original coarse PIC setup to isolate parameter changes: 20 mm grid, 4 substeps, "
        "50 stress iterations, one coupling pass and relaxation 0.1. Beds are prepared for "
        "0.5 s with the robot held fixed, "
        "preserving the actual MPM entry history. The short bed is 3 m long. These trials remain exploratory because "
        "the MPM reference failed the numerical gates.</p>"
        '<img src="policy-and-support.png" alt="Frozen-policy tracking error and granular versus rigid support">'
        "<table><tr><th>MPM setting</th><th>Policy training terrain</th><th>Tracking "
        "error</th><th>First failure check</th><th>Minimum signed lateral foot margin</th></tr>"
        + "".join(material_rows)
        + "</table>"
        "<p>Failure checks cover body height below 0.16 m relative to the initial free surface, "
        "projected-gravity z above −0.3, and nonfinite body positions. Trials continue without resetting.</p>"
        "<p>The foot-margin audit uses recorded poses: left feet must remain at least 10 mm left of the base, "
        "and right feet at least 10 mm right, in the base frame. Negative values indicate centerline crossing. "
        "It does not certify contact slip or stepping quality.</p>"
        + "<h2>Coupling audit</h2><p>Multi-pass solves now restore the entire persistent entry "
        "state before re-solving the same "
        "interval. A native MPM/APIC regression failed before this correction and passes "
        "afterward. Preparation also retains "
        "the entry material history. All cases below use the corrected implementation; the "
        "four original one-pass, unprepared "
        "videos are retained as the original exploratory recordings.</p>"
        "<p>Relaxation α blends new proxy feedback with its previous value. At α=0.1, the recurrence is "
        "0.9 × previous + 0.1 × new; the steady force is not simply multiplied by 0.1. "
        "Changing timestep also changes this temporal filter. The sweep changes relaxation, pass count and substeps "
        "with an otherwise fixed policy and a 2 m prepared bed.</p>"
        '<img src="coupling.png" alt="Frozen-policy sensitivity to coupling controls">'
        + "<h2>Interpretation</h2><p>These are simulation comparisons, without measured physical-sand data. "
        "The current robot test uses a shallow sand layer over a rigid floor. Whole-robot momentum audits quantify "
        "granular and backing-floor support; the rigid-floor control validates the load-accounting method.</p>"
        "<p>The RFT-trained policy handles this MPM setup better than the rigid-trained policy in the selected tests. "
        "The evidence supports useful training exposure to granular resistance. It does not "
        "yet show that the two material "
        "models represent the same sand. A quantitative transfer benchmark needs a converged "
        "MPM contact/transfer setup, "
        "then independent force calibration and longer multi-seed policy evaluations.</p>"
        '<p><a href="analysis.json">Numerical gates and held-out fit errors</a> · '
        '<a href="probe-records.json">All probe records</a> · '
        '<a href="constitutive-parameters.json">Native friction-invariant check</a></p>'
        f"""<script>
async function updateProgress() {{
  try {{
    const response = await fetch('finalization-status.json', {{cache: 'no-store'}});
    if (!response.ok) return;
    const status = await response.json();
    const count = status.completion?.probes ??
      Object.values(status.probe_counts ?? {{}}).reduce((a, b) => a + b, 0);
    document.getElementById('live-progress').textContent =
      `Live measurements: ${{count}}/50 complete. Figures currently include {len(data)} probes.`;
    if (status.stage === 'Reports published' && count !== {len(data)}) location.reload();
  }} catch (_) {{}}
}}
updateProgress();
setInterval(updateProgress, 10000);
</script>"""
    )
    (ROOT / "analysis.html").write_text(page)
    if final:
        (ROOT / "index.html").write_text(page)
    print(json.dumps(dict(cases=len(data), accepted=accepted, fit=fit)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    main(parser.parse_args().final)
