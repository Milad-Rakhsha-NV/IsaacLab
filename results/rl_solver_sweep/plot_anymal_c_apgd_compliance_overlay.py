#!/usr/bin/env python3
"""Plot Anymal-C APGD reward curves for zero and nonzero contact compliance."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw' / 'anymal_c'
OUT = ROOT / 'paper_clean' / 'anymal_c_progress'
OUT.mkdir(parents=True, exist_ok=True)
ITER = re.compile(r'Learning iteration\s+(\d+)/')
REWARD = re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
FPS = re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

runs = {
    '0': RAW / 'apgd_coupling2_cache_on_contact20',
    '1e-7': RAW / 'apgd_coupling2_cache_on_contact20_compliance_1e-7',
    '1e-6': RAW / 'apgd_coupling2_cache_on_contact20_compliance_1e-6',
    '1e-5': RAW / 'apgd_coupling2_cache_on_contact20_compliance_1e-5',
}
colors = {'0': '#000000', '1e-7': '#0000FF', '1e-6': '#FF1700', '1e-5': '#00AA00'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.linewidth': .75, 'lines.linewidth': .8,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
})

def read_run(path):
    text = (path / 'run.log').read_text(errors='replace')
    current = None; rewards = {}; fps = []
    for line in text.splitlines():
        m = ITER.search(line)
        if m: current = int(m.group(1))
        m = REWARD.search(line)
        if m and current is not None: rewards[current] = float(m.group(1))
        m = FPS.search(line)
        if m and current is not None: fps.append(float(m.group(1)))
    x = np.array(sorted(rewards), dtype=float)
    y = np.array([rewards[int(i)] for i in x])
    return x, y, float(np.median(fps))

def smooth_band(y, span=35):
    a = 2.0 / (span + 1.0)
    ema = np.empty_like(y); ema[0] = y[0]
    for i in range(1, len(y)): ema[i] = a*y[i] + (1-a)*ema[i-1]
    half = max(3, span // 2)
    sd = np.array([np.std(y[max(0,i-half):min(len(y),i+half+1)]) for i in range(len(y))])
    return ema, ema-sd, ema+sd

fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=False)
for compliance, path in runs.items():
    x, y, fps = read_run(path)
    ema, lo, hi = smooth_band(y)
    label = f'c={compliance} [{fps/1000:.0f}k FPS]'
    ax.plot(x, ema, color=colors[compliance], label=label)
    ax.fill_between(x, lo, hi, color=colors[compliance], alpha=.22, linewidth=0)

ax.set_title('Anymal C')
ax.set_xlabel('PPO iteration')
ax.set_ylabel('Mean episode reward')
ax.set_xlim(left=0)
ax.grid(True, which='major', color='#999', alpha=.35, linewidth=.5)
ax.grid(True, which='minor', color='#BBB', alpha=.18, linewidth=.35)
ax.tick_params(direction='in', top=True, right=True)
ax.legend(frameon=False, fontsize=6.8, loc='best', handlelength=1.6,
          handletextpad=.35, borderaxespad=.25)
fig.subplots_adjust(left=.17, right=.98, bottom=.19, top=.84)
base = OUT / 'anymal_c_apgd_compliance_overlay'
for suffix, kwargs in [('.png', {'dpi': 600}), ('.pdf', {}), ('.svg', {})]:
    fig.savefig(base.with_suffix(suffix), **kwargs)
plt.close(fig)
print(f'output={base}')
