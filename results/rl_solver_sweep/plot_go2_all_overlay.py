#!/usr/bin/env python3
"""Overlay validated Go2 baseline and coupling-2 Jacobi reward curves."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/go2"
OUT = ROOT / "figures"
RUNS = [
    ("Jacobi, 5", "newton_dvi", "contact_5", "blue", "-", 1.8),
    ("APGD, 5", "newton_dvi_apgd", "contact_5", "black", "-", 1.6),
    ("P-SPG-FB, 5", "newton_dvi_pspg", "contact_5", "red", "-", 1.6),
    ("Jacobi, 10", "newton_dvi", "contact_10", "blue", "--", 1.8),
    ("APGD, 10", "newton_dvi_apgd", "contact_10", "black", "--", 1.6),
    ("P-SPG-FB, 10", "newton_dvi_pspg", "contact_10", "red", "--", 1.6),
    ("MJWarp (native)", "newton_mjwarp", "contact_10", "green", "-", 1.8),
]
COUPLING_RUNS = [
    ("Jacobi, 5; coupling=2", 5, "blue", ":"),
    ("Jacobi, 10; coupling=2", 10, "blue", "-."),
    ("Jacobi, 15; coupling=2", 15, "blue", (0, (6, 2, 1, 2))),
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

fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
for label, solver, budget, color, style, width in RUNS:
    x, y, fps = parse(RAW / solver / budget / "run.log")
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=width,
            label=f"{label}  [{fps/1000:.1f}k steps/s]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.04, linewidth=0)
for label, contact, color, style in COUPLING_RUNS:
    path = RAW / "newton_dvi_coupling2" / f"contact_{contact}" / "run.log"
    x, y, fps = parse(path)
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=2.2,
            label=f"{label}  [{fps/1000:.1f}k steps/s]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.07, linewidth=0)
ax.set_title("Go2")
ax.set_xlabel("PPO iteration"); ax.set_ylabel("Mean episode reward")
ax.set_xlim(left=0); ax.grid(True, alpha=0.22); ax.legend(loc="best", fontsize=8, ncol=2)
OUT.mkdir(parents=True, exist_ok=True)
for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 220})):
    fig.savefig(OUT / f"go2_all_reward_curves_overlay{suffix}", **kwargs)
plt.close(fig)
print(OUT / "go2_all_reward_curves_overlay.pdf")
