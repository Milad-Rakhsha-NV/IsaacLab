# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Create synthetic mathematical illustrations; these are not paper results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    a = np.linspace(-0.1, 0.1, 1600)
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True, constrained_layout=True)
    ax[0].plot(a, a, color="#14263d", label="Reference map: F(a) = a")
    ax[0].plot(a, a + 0.01 * np.sin(200 * a), color="#2563eb", label="Small-error map: a + 0.01 sin(200a)")
    ax[0].plot(a, a + 0.02, color="#0f766e", ls="--", label="Larger-bias smooth map: a + 0.02")
    ax[0].set(
        ylabel="Predicted next quantity", title="Synthetic example: state-fit error does not bound derivative error"
    )
    ax[0].legend(fontsize=8)
    ax[1].plot(a, np.ones_like(a), color="#14263d")
    ax[1].plot(a, 1 + 2 * np.cos(200 * a), color="#2563eb")
    ax[1].plot(a, np.ones_like(a), color="#0f766e", ls="--")
    ax[1].set(xlabel="Scalar action a", ylabel="Derivative with respect to a")
    for axis in ax:
        axis.grid(alpha=0.2)
    fig.savefig(out / "prediction_and_gradient.pdf")
    plt.close(fig)
    a = np.linspace(-0.09, 0.09, 800)
    tau = 0.015
    constrained = np.maximum(a, 0)
    smooth = tau * np.logaddexp(0, a / tau)
    target = 0.06
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.7), sharex=True, constrained_layout=True)
    ax[0].plot(a, constrained, label="Unilateral template: max(a, 0)", color="#14263d")
    ax[0].plot(a, smooth, label="Smooth surrogate: softplus, scale 0.015 m", color="#2563eb")
    ax[0].axhline(target, color="#0f766e", ls="--", label="Desired height: 0.06 m")
    ax[0].set(
        ylabel="Next height (m)", title="Synthetic constraint example: local derivatives and finite changes differ"
    )
    ax[0].legend(fontsize=8)
    exact_gradient = -2 * (constrained - target) * (a > 0)
    surrogate_gradient = -2 * (smooth - target) / (1 + np.exp(-a / tau))
    ax[1].plot(a, exact_gradient, color="#14263d", label="Local return derivative")
    ax[1].plot(a, surrogate_gradient, color="#2563eb", label="Surrogate return derivative")
    ax[1].set(xlabel="Candidate free-motion height a (m), not a G1 action", ylabel="Return derivative")
    ax[1].legend(fontsize=8)
    for axis in ax:
        axis.grid(alpha=0.2)
    fig.savefig(out / "contact_gradient.pdf")
    plt.close(fig)
    print("Created two synthetic vector figures; no empirical measurements used.")


if __name__ == "__main__":
    main()
