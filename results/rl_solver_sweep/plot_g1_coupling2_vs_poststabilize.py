#!/usr/bin/env python3
"""G1 Jacobi comparison: full coupling-2 versus one sweep + post-stabilization."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/g1"
OUT = ROOT / "figures"
RUNS = [
    ("Jacobi, coupling 2, contact 5", "newton_dvi_coupling2/contact_5", "#0000FF", "-"),
    ("Jacobi, coupling 2, contact 10", "newton_dvi_coupling2/contact_10", "#0000FF", "--"),
    ("Jacobi, coupling 2, contact 15", "newton_dvi_coupling2/contact_15", "#0000FF", ":"),
    ("Jacobi, coupling 1 + post-stabilization, contact 10", "newton_dvi_post_stabilize/contact_10", "#FF1700", "--"),
]
ITER = re.compile(r"Learning iteration\s+(\d+)/")
REWARD = re.compile(r"Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FPS = re.compile(r"Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse(rel):
    d = RAW / rel
    p = d / "run.log"
    text = p.read_text(errors="replace")
    valid = (d / "COMPLETED").exists() and "Training time:" in text and "exit=0" in text
    if not valid:
        raise RuntimeError(f"incomplete: {p}")
    current = None; rewards = {}; fps = []
    for line in text.splitlines():
        m = ITER.search(line)
        if m: current = int(m.group(1))
        m = REWARD.search(line)
        if m and current is not None: rewards[current] = float(m.group(1))
        m = FPS.search(line)
        if m and current is not None: fps.append(float(m.group(1)))
    x = np.array(sorted(rewards)); y = np.array([rewards[i] for i in x])
    return x, y, float(np.median(fps)), float(y[-1])

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
for label, rel, color, style in RUNS:
    try: x, y, fps, final = parse(rel)
    except RuntimeError as e:
        print(f"SKIP {e}"); continue
    mean, lo, hi = smooth_band(y)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=2.0,
            label=f"{label} [{fps/1000:.1f}k steps/s; final {final:.2f}]")
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
ax.set_title("G1")
ax.set_xlabel("PPO iteration"); ax.set_ylabel("Mean episode reward")
ax.set_xlim(left=0); ax.grid(True, alpha=0.22); ax.legend(loc="best", fontsize=8)
ax.text(0.01, 0.01, "EMA smoothing (span 35); bands show within-run rolling variability. Blue = full coupling-2; red = one coupling sweep + post-stabilization.", transform=ax.transAxes, fontsize=8, color="#555555")
OUT.mkdir(parents=True, exist_ok=True)
base = OUT / "g1_coupling2_vs_poststabilize_reward_curves"
for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 220})):
    fig.savefig(base.with_suffix(suffix), **kwargs)
plt.close(fig)
print(base.with_suffix('.png'))
