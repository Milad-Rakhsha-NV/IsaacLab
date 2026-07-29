#!/usr/bin/env python3
"""Plot the completed Ant contact-10 reward curves with FPS in the legend."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/ant/initial_contact_10"
OUT = ROOT / "figures"
RUNS = [
    ("Jacobi, 10", "jacobi.log", "blue", "-"),
    ("APGD, 10", "apgd.log", "black", "-"),
    ("P-SPG-FB, 10", "pspg.log", "red", "-"),
    ("MJWarp (native)", "mjwarp.log", "green", "-") ,
]
ITER = re.compile(r"Learning iteration\s+(\d+)/")
REWARD = re.compile(r"Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FPS = re.compile(r"Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse(path):
    text = path.read_text(errors="replace")
    # The Ant pilot predates the restart-safe COMPLETED-marker wrapper; its
    # terminal `END ... exit=0` line is the completion gate.
    assert "Training time:" in text and "exit=0" in text
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
    ema = np.empty_like(y, dtype=float)
    ema[0] = y[0]
    for i in range(1, len(y)):
        ema[i] = alpha * y[i] + (1.0 - alpha) * ema[i - 1]
    half = max(3, span // 2)
    lo = np.empty_like(y, dtype=float); hi = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        a, b = max(0, i-half), min(len(y), i+half+1)
        sd = np.std(y[a:b])
        lo[i], hi[i] = ema[i] - sd, ema[i] + sd
    return ema, lo, hi

fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
for label, filename, color, linestyle in RUNS:
    x, y, fps = parse(RAW / filename)
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=2.0,
            label=f"{label}  [{fps/1000:.1f}k steps/s]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
ax.set_title("ANT")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("Mean episode reward")
ax.set_xlim(left=0)
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=9)
OUT.mkdir(exist_ok=True)
for suffix, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 220})]:
    fig.savefig(OUT / f"ant_reward_curves{suffix}", **kwargs)
plt.close(fig)
print(OUT / "ant_reward_curves.pdf")
