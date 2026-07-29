#!/usr/bin/env python3
"""Regenerate Anymal-C Jacobi/APGD/MJWarp progress plots from available logs."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw'/'anymal_c'; OUT=ROOT/'paper_clean'/'anymal_c_progress'; OUT.mkdir(parents=True,exist_ok=True)
ITER=re.compile(r'Learning iteration\s+(\d+)/'); REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'); FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
COL={'Jacobi':'#0000FF','APGD':'#000000','P-SPG-FB':'#FF1700','MJWarp':'#00FF00'}
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.75,'lines.linewidth':.75,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.bbox':'tight'})
def read(name,partial=True):
 p=(RAW/name/'run.log') if name == 'mjwarp' else ((RAW/'jacobi_coupling2_cache_on_contact20'/'run.log') if name == 'jacobi' else ((RAW/'apgd_coupling2_cache_on_contact20'/'run.log') if name == 'apgd' else (RAW/'newton_dvi_pspg'/'contact_20_coupling2_post_false'/'run.log')))
 if not p.exists(): return None
 t=p.read_text(errors='replace'); cur=None; r={}; f=[]
 for line in t.splitlines():
  m=ITER.search(line)
  if m: cur=int(m.group(1))
  m=REWARD.search(line)
  if m and cur is not None:r[cur]=float(m.group(1))
  m=FPS.search(line)
  if m and cur is not None:f.append(float(m.group(1)))
 if not r:return None
 x=np.array(sorted(r)); y=np.array([r[i] for i in x]); return x,y,float(np.median(f)) if f else float('nan')
def smooth(y,span=35):
 a=2/(span+1); e=np.empty_like(y); e[0]=y[0]
 for i in range(1,len(y)):e[i]=a*y[i]+(1-a)*e[i-1]
 h=max(3,span//2); sd=np.array([np.std(y[max(0,i-h):min(len(y),i+h+1)]) for i in range(len(y))]); return e,e-sd,e+sd
fig,ax=plt.subplots(figsize=(3.35,2.65),constrained_layout=False); n=0
for label in ('Jacobi','APGD','P-SPG-FB','MJWarp'):
 z=read(label.lower())
 if z is None: continue
 x,y,f=z; e,lo,hi=smooth(y); fps=f'{f/1000:.0f}k FPS' if np.isfinite(f) else 'FPS pending'; ax.plot(x,e,color=COL[label],lw=.75,label=f'{label} [{fps}]'); ax.fill_between(x,lo,hi,color=COL[label],alpha=.38,lw=0); n+=1
ax.set_title('Anymal C'); ax.set_xlabel('PPO iteration'); ax.set_ylabel('Mean episode reward'); ax.set_xlim(left=0); ax.grid(True,which='major',color='#999',alpha=.35,linewidth=.5); ax.grid(True,which='minor',color='#BBB',alpha=.18,linewidth=.35); ax.tick_params(direction='in',top=True,right=True); ax.legend(frameon=False,fontsize=7,loc='best',handlelength=1.6,handletextpad=.35,borderaxespad=.25); fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.84)
base=OUT/'anymal_c_jacobi_mjwarp_apgd_progress'
for s,k in [('.png',{'dpi':600}),('.pdf',{}),('.svg',{})]: fig.savefig(base.with_suffix(s),**k)
plt.close(fig); print(f'curves={n} output={base}')
