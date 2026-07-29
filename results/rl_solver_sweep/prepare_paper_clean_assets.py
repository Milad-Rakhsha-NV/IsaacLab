#!/usr/bin/env python3
"""Create clean MJWarp-vs-selected-Jacobi reward figures from explicit runs."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'paper_clean'; FIG=OUT/'figures'
plt.rcParams.update({
 'font.family':'serif', 'font.serif':['STIXGeneral','DejaVu Serif'],
 'mathtext.fontset':'stix', 'font.size':9, 'axes.labelsize':9,
 'axes.titlesize':9, 'legend.fontsize':7, 'xtick.labelsize':8,
 'ytick.labelsize':8, 'axes.linewidth':0.75, 'lines.linewidth':0.75,
 'lines.markersize':3.5, 'figure.facecolor':'white',
 'axes.facecolor':'white', 'savefig.facecolor':'white',
 'savefig.bbox':'tight'})
ITER=re.compile(r'Learning iteration\s+(\d+)/')
REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
RUNS={
'ant': [('Jacobi','ant/newton_dvi_post_stabilize/contact_10/run.log','#0000FF','-',True),('MJWarp','archive_2026-07-24_nonpaper_runs/raw/ant/initial_contact_10/mjwarp.log','#00FF00','-',False)],
'humanoid': [('Jacobi','humanoid/newton_dvi_post_stabilize/contact_10/run.log','#0000FF','-',True),('MJWarp','humanoid/newton_mjwarp/contact_10/run.log','#00FF00','-',True)],
'h1': [('Jacobi','archive_2026-07-24_nonpaper_runs/raw/h1/newton_dvi/contact_10/run.log','#0000FF','-',True),('MJWarp','h1/newton_mjwarp/contact_10/run.log','#00FF00','-',True)],
'g1': [('Jacobi','g1/newton_dvi_coupling2_cache_on/contact_15/run.log','#0000FF','-',True),('MJWarp','g1/newton_mjwarp/contact_15/run.log','#00FF00','-',True)],
'go2': [('Jacobi','go2/newton_dvi_post_stabilize/contact_15/run.log','#0000FF','-',True),('MJWarp','go2/newton_mjwarp/contact_15/run.log','#00FF00','-',True)],
}
def parse(rel, marker):
 p=(ROOT/rel) if rel.startswith('archive_') else (RAW/rel); d=p.parent; t=p.read_text(errors='replace')
 if 'Training time:' not in t or 'exit=0' not in t or (marker and not (d/'COMPLETED').exists()): raise RuntimeError(p)
 cur=None;r={};fps=[]
 for line in t.splitlines():
  m=ITER.search(line)
  if m: cur=int(m.group(1))
  m=REWARD.search(line)
  if m and cur is not None:r[cur]=float(m.group(1))
  m=FPS.search(line)
  if m and cur is not None and cur>=1:fps.append(float(m.group(1)))
 x=np.array(sorted(r)); y=np.array([r[i] for i in x]); return x,y,float(np.median(fps)) if fps else float('nan')
def smooth(y,span=35):
 a=2/(span+1);e=np.empty_like(y);e[0]=y[0]
 for i in range(1,len(y)):e[i]=a*y[i]+(1-a)*e[i-1]
 lo=np.empty_like(y);hi=np.empty_like(y);h=max(3,span//2)
 for i in range(len(y)):
  s=y[max(0,i-h):min(len(y),i+h+1)];sd=np.std(s);lo[i]=e[i]-sd;hi[i]=e[i]+sd
 return e,lo,hi
FIG.mkdir(parents=True,exist_ok=True)
combined_data=[]
for env,runs in RUNS.items():
 fig,ax=plt.subplots(figsize=(3.35,2.65),constrained_layout=False); plotted=0
 combined_curves=[]
 for label,rel,c,ls,marker in runs:
  try:x,y,fps=parse(rel,marker)
  except Exception as e: print('SKIP',e);continue
  e,lo,hi=smooth(y); ax.plot(x,e,color=c,ls=ls,lw=0.75,label=f'{label} [{fps/1000:.0f}k FPS]');ax.fill_between(x,lo,hi,color=c,alpha=.38,lw=0);combined_curves.append((x,e,lo,hi,c,label,fps));plotted+=1
 ax.set_title(env[:1].upper()+env[1:]);ax.set_xlabel('PPO iteration');ax.set_ylabel('Mean episode reward');ax.set_xlim(left=0);ax.grid(True,which='major',color='#999999',alpha=.35,linewidth=.5);ax.grid(True,which='minor',color='#BBBBBB',alpha=.18,linewidth=.35);ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False,fontsize=7,loc='best',handlelength=1.6,handletextpad=.35,borderaxespad=.25)
 fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.84)
 base=FIG/f'{env}_mjwarp_vs_selected_jacobi_reward'
 for s,k in [('.png',{'dpi':600}),('.pdf',{}),('.svg',{})]:fig.savefig(base.with_suffix(s),**k)
 plt.close(fig); combined_data.append((env,combined_curves)); print(env,plotted,base.with_suffix('.png'))

# Compact five-panel review figure for paper layout inspection.
fig,axes=plt.subplots(2,3,figsize=(7.1,4.55),constrained_layout=False)
for ax,(env,curves) in zip(axes.flat,combined_data):
 for x,e,lo,hi,c,label,fps in curves:
  ax.plot(x,e,color=c,lw=0.75,label=f'{label} [{fps/1000:.0f}k FPS]')
  ax.fill_between(x,lo,hi,color=c,alpha=.38,lw=0)
 ax.set_title(env[:1].upper()+env[1:],fontsize=9)
 ax.set_xlabel('PPO iteration',fontsize=8); ax.set_ylabel('Mean episode reward',fontsize=8)
 ax.tick_params(direction='in',top=True,right=True,labelsize=7)
 ax.grid(True,which='major',color='#999999',alpha=.35,linewidth=.5)
 ax.legend(frameon=False,fontsize=6.3,loc='best',handlelength=1.3,handletextpad=.3,borderaxespad=.2)
axes.flat[-1].axis('off')
fig.subplots_adjust(left=.085,right=.99,bottom=.10,top=.94,wspace=.28,hspace=.42)
for s,k in [('.png',{'dpi':600}),('.pdf',{}),('.svg',{})]:
 fig.savefig(FIG/f'all_envs_jacobi_mjwarp_reward_review{s}',**k)
plt.close(fig)
