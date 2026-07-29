#!/usr/bin/env python3
"""Publication-style smoothed PPO reward curves with variability bands.

Each environment plot uses one distinct color per solver/budget and reports
median steady-state environment steps/s in the legend. The primary curve is an
EMA-smoothed reward trace; the translucent band is a rolling within-run
variability band, not a cross-seed confidence interval.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "figures"

RUNS = [
    ("Jacobi, 5", "newton_dvi", "contact_5", "blue", "-"),
    ("APGD, 5", "newton_dvi_apgd", "contact_5", "black", "-"),
    ("P-SPG-FB, 5", "newton_dvi_pspg", "contact_5", "red", "-"),
    ("Jacobi, 10", "newton_dvi", "contact_10", "blue", "--"),
    ("APGD, 10", "newton_dvi_apgd", "contact_10", "black", "--"),
    ("P-SPG-FB, 10", "newton_dvi_pspg", "contact_10", "red", "--"),
    ("MJWarp (native)", "newton_mjwarp", "contact_10", "green", "-"),
]

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
REWARD_RE = re.compile(r"Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FPS_RE = re.compile(r"Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(path: Path):
    iterations, rewards, fps = [], [], []
    current = None
    text = path.read_text(errors="replace")
    if not (path.parent / "COMPLETED").exists():
        raise RuntimeError(f"not completed: {path}")
    if "Training time:" not in text or "exit=0" not in text:
        raise RuntimeError(f"incomplete log: {path}")
    for line in text.splitlines():
        m = ITER_RE.search(line)
        if m:
            current = int(m.group(1))
        m = REWARD_RE.search(line)
        if m and current is not None:
            iterations.append(current)
            rewards.append(float(m.group(1)))
        m = FPS_RE.search(line)
        if m and current is not None and current >= 1:
            fps.append(float(m.group(1)))
    unique = {}
    for i, r in zip(iterations, rewards):
        unique[i] = r
    xy = sorted(unique.items())
    if not xy:
        raise RuntimeError(f"no reward samples: {path}")
    return (np.array([x for x, _ in xy]), np.array([y for _, y in xy]),
            float(np.median(fps)))


def smooth_band(y: np.ndarray, span: int = 35):
    """Return EMA and rolling +/- one standard-deviation band."""
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(y, dtype=float)
    ema[0] = y[0]
    for i in range(1, len(y)):
        ema[i] = alpha * y[i] + (1.0 - alpha) * ema[i - 1]
    half = max(3, span // 2)
    lo = np.empty_like(y, dtype=float)
    hi = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        a, b = max(0, i - half), min(len(y), i + half + 1)
        sd = np.std(y[a:b])
        lo[i], hi[i] = ema[i] - sd, ema[i] + sd
    return ema, lo, hi


def make_plot(env: str):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for label, solver, budget, color, linestyle in RUNS:
        path = RAW / env / solver / budget / "run.log"
        it, reward, fps = parse_log(path)
        mean, lo, hi = smooth_band(reward)
        ax.plot(it, mean, color=color, linestyle=linestyle, linewidth=2.0,
                label=f"{label}  [{fps / 1000:.1f}k steps/s]")
        ax.fill_between(it, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.set_title(env.upper())
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.set_xlim(left=0)
    out = OUT / f"{env}_reward_curves"
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 220})):
        fig.savefig(out.with_suffix(suffix), **kwargs)
    plt.close(fig)
    return out.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="+", default=["g1", "h1", "go2"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for env in args.env:
        print(make_plot(env))


if __name__ == "__main__":
    main()
