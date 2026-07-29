#!/usr/bin/env python3
"""Overlay available G1 contact-5/10/15 and native MJWarp reward traces.

Important: the contact-15 Jacobi/APGD logs were stopped/completed at 1000 PPO
iterations, whereas the required G1 duration is 1500. They are included only
as provisional visual overlays and are explicitly labeled as 1000-iteration
runs; they must not be used as final 1500-iteration results. Incomplete runs
are excluded.
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/g1"
OUT = ROOT / "figures"
RUNS = [
    ("Jacobi, 5", "newton_dvi", "contact_5", "blue", "-"),
    ("APGD, 5", "newton_dvi_apgd", "contact_5", "black", "-"),
    ("P-SPG-FB, 5", "newton_dvi_pspg", "contact_5", "red", "-"),
    ("Jacobi, 10", "newton_dvi", "contact_10", "blue", "--"),
    ("APGD, 10", "newton_dvi_apgd", "contact_10", "black", "--"),
    ("P-SPG-FB, 10", "newton_dvi_pspg", "contact_10", "red", "--"),
    ("Jacobi, 15 [1000 it. provisional]", "newton_dvi", "contact_15", "blue", ":"),
    ("APGD, 15 [1000 it. provisional]", "newton_dvi_apgd", "contact_15", "black", ":"),
    ("MJWarp (native)", "newton_mjwarp", "contact_10", "green", "-"),
]
ITER = re.compile(r"Learning iteration\s+(\d+)/")
REWARD = re.compile(r"Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FPS = re.compile(r"Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse(path):
    text = path.read_text(errors="replace")
    if not ((path.parent / "COMPLETED").exists() and "Training time:" in text and "exit=0" in text):
        raise RuntimeError(f"incomplete log: {path}")
    current = None; rewards = {}; fps = []
    for line in text.splitlines():
        m = ITER.search(line)
        if m: current = int(m.group(1))
        m = REWARD.search(line)
        if m and current is not None: rewards[current] = float(m.group(1))
        m = FPS.search(line)
        if m and current is not None and current >= 1: fps.append(float(m.group(1)))
    x = np.array(sorted(rewards)); y = np.array([rewards[i] for i in x])
    return x, y, float(np.median(fps))

def smooth_band(y, span=35):
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(y, dtype=float); ema[0] = y[0]
    for i in range(1, len(y)): ema[i] = alpha*y[i] + (1-alpha)*ema[i-1]
    lo = np.empty_like(y); hi = np.empty_like(y); half = max(3, span//2)
    for i in range(len(y)):
        a, b = max(0, i-half), min(len(y), i+half+1)
        sd = np.std(y[a:b]); lo[i], hi[i] = ema[i]-sd, ema[i]+sd
    return ema, lo, hi

fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
for label, solver, budget, color, style in RUNS:
    try: x, y, fps = parse(RAW / solver / budget / "run.log")
    except RuntimeError: continue
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=1.9,
            label=f"{label}  [{fps/1000:.1f}k steps/s]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.08, linewidth=0)
ax.set_title("G1")
ax.set_xlabel("PPO iteration"); ax.set_ylabel("Mean episode reward")
ax.set_xlim(left=0); ax.grid(True, alpha=0.22); ax.legend(loc="best", fontsize=8)
ax.text(0.01, 0.01, "Dotted contact-15 traces are provisional 1000-iteration runs; required G1 duration is 1500.",
        transform=ax.transAxes, fontsize=8, color="#555555")
OUT.mkdir(parents=True, exist_ok=True)
for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 220})):
    fig.savefig(OUT / f"g1_reward_curves_with_contact15_provisional{suffix}", **kwargs)
plt.close(fig)
print(OUT / "g1_reward_curves_with_contact15_provisional.pdf")
