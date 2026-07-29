#!/usr/bin/env python3
"""Plot the finished G1 coupling-2/cache-on contact-15 run against prior G1 curves."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw/g1'; OUT=ROOT/'figures'
ITER=re.compile(r'Learning iteration\s+(\d+)/'); REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'); FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
RUNS=[
 ('Current: coupling 2, cache on, contact 15','newton_dvi_coupling2_cache_on/contact_15','#0000FF','-',True),
 ('Current: coupling 1 + post-stabilization, cache on, contact 15','newton_dvi_post_stabilize_cache_on/contact_15','#FF1700','--',True),
 ('Current: coupling 2, cache off, contact 15','newton_dvi_coupling2_cache_off/contact_15','#ff7f0e','-.',True),
 ('Current: coupling 1, no post-stabilization, contact 15','newton_dvi_coupling1_no_post_cache_on/contact_15','#9467bd',(0,(5,2)),True),
 ('Prior: coupling 2, contact 15','newton_dvi_coupling2/contact_15','#111111',':',True),
 ('Prior: coupling 2, contact 10','newton_dvi_coupling2/contact_10','#9467bd','-.',True),
 ('Prior: coupling 1 + post-stabilization, contact 10','newton_dvi_post_stabilize/contact_10','#00FF00',(0,(3,1,1,1)),True),
]
def parse(rel, marker):
 d=RAW/rel; p=d/'run.log'; t=p.read_text(errors='replace')
 if 'Training time:' not in t or 'exit=0' not in t or (marker and not (d/'COMPLETED').exists()): raise RuntimeError(str(p))
 cur=None;r={};f=[]
 for line in t.splitlines():
  m=ITER.search(line)
  if m:cur=int(m.group(1))
  m=REWARD.search(line)
  if m and cur is not None:r[cur]=float(m.group(1))
  m=FPS.search(line)
  if m and cur is not None and cur>=1:f.append(float(m.group(1)))
 x=np.array(sorted(r)); y=np.array([r[i] for i in x]); return x,y,np.median(f),y[-1]
def smooth(y,span=35):
 a=2/(span+1);e=np.empty_like(y);e[0]=y[0]
 for i in range(1,len(y)):e[i]=a*y[i]+(1-a)*e[i-1]
 lo=np.empty_like(y);hi=np.empty_like(y);h=max(3,span//2)
 for i in range(len(y)):
  s=y[max(0,i-h):min(len(y),i+h+1)];sd=np.std(s);lo[i]=e[i]-sd;hi[i]=e[i]+sd
 return e,lo,hi
fig,ax=plt.subplots(figsize=(11.5,6.8),constrained_layout=True)
for label,rel,c,ls,mark in RUNS:
 try:x,y,fps,final=parse(rel,mark)
 except RuntimeError as e: print('SKIP',e);continue
 e,lo,hi=smooth(y); ax.plot(x,e,color=c,ls=ls,lw=2,label=f'{label} [{fps/1000:.1f}k steps/s; final {final:.2f}]');ax.fill_between(x,lo,hi,color=c,alpha=.09,lw=0)
ax.set_title('G1');ax.set_xlabel('PPO iteration');ax.set_ylabel('Mean episode reward');ax.set_xlim(left=0);ax.grid(alpha=.22);ax.legend(fontsize=8,loc='best')
ax.text(.01,.01,'EMA smoothing (span 35); bands show rolling within-run variability.',transform=ax.transAxes,fontsize=8,color='#555')
OUT.mkdir(exist_ok=True);base=OUT/'g1_finished_coupling2_cache_on_reward_curve'
for s,k in [('.png',{'dpi':220}),('.pdf',{}),('.svg',{})]:fig.savefig(base.with_suffix(s),**k)
plt.close(fig);print(base.with_suffix('.png'))
