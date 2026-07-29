#!/usr/bin/env python3
"""Go2 reward curves for the validated contact-15 solver comparison."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/go2"
OUT = ROOT / "figures"
RUNS = [
    ("DVI Jacobi, coupling=2 (baseline)", "newton_dvi_coupling2/contact_15", "#0000FF", "-", 2.0),
    ("DVI Jacobi, coupling=2, cache off", "newton_dvi_coupling2_cache_compare/cache_false_contact_15", "#0000FF", "--", 1.8),
    ("DVI Jacobi, coupling=2, cache on", "newton_dvi_coupling2_cache_compare/cache_true_contact_15", "#0000FF", ":", 2.4),
    ("DVI Jacobi, coupling=1 + post-stabilize", "newton_dvi_post_stabilize/contact_15", "#FF1700", "-", 2.2),
    ("MJWarp (native)", "newton_mjwarp/contact_15", "#00FF00", "-", 2.2),
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
        if m and current is not None: fps.append(float(m.group(1)))
    xy = sorted(rewards.items())
    x = np.array([i for i, _ in xy]); y = np.array([r for _, r in xy])
    return x, y, float(np.median(fps))

def smooth_band(y, span=35):
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(y); ema[0] = y[0]
    for i in range(1, len(y)): ema[i] = alpha*y[i] + (1-alpha)*ema[i-1]
    lo = np.empty_like(y); hi = np.empty_like(y); half = max(3, span//2)
    for i in range(len(y)):
        a, b = max(0, i-half), min(len(y), i+half+1)
        sd = np.std(y[a:b]); lo[i], hi[i] = ema[i]-sd, ema[i]+sd
    return ema, lo, hi

fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
for label, rel, color, style, width in RUNS:
    x, y, fps = parse(RAW / rel / "run.log")
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=width,
            label=f"{label} [{fps/1000:.1f}k steps/s]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.10 if "MJWarp" not in label else 0.07, linewidth=0)
ax.set_title("Go2")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("Mean episode reward")
ax.set_xlim(left=0)
ax.grid(True, alpha=0.22)
ax.legend(loc="best", fontsize=9)
for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 220})):
    fig.savefig(OUT / f"go2_contact15_mjwarp_reward_curves{suffix}", **kwargs)
plt.close(fig)
print(OUT / "go2_contact15_mjwarp_reward_curves.png")
