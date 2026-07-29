#!/usr/bin/env python3
"""Regenerate separate Jacobi/MJWarp/APGD progress plots from available logs."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw'; OUT=ROOT/'paper_clean'/'apgd_progress'; OUT.mkdir(parents=True,exist_ok=True)
ITER=re.compile(r'Learning iteration\s+(\d+)/'); REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'); FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.75,'lines.linewidth':0.75,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.bbox':'tight'})
BASE={'ant':('ant/newton_dvi_post_stabilize/contact_10/run.log','archive_2026-07-24_nonpaper_runs/raw/ant/initial_contact_10/mjwarp.log',10,1,True),'humanoid':('humanoid/newton_dvi_post_stabilize/contact_10/run.log','humanoid/newton_mjwarp/contact_10/run.log',10,1,True),'h1':('archive_2026-07-24_nonpaper_runs/raw/h1/newton_dvi/contact_10/run.log','h1/newton_mjwarp/contact_10/run.log',10,1,False),'g1':('g1/newton_dvi_coupling2_cache_on/contact_15/run.log','g1/newton_mjwarp/contact_15/run.log',15,2,False),'go2':('go2/newton_dvi_post_stabilize/contact_15/run.log','go2/newton_mjwarp/contact_15/run.log',15,1,True)}
COL={'Jacobi':'#0000FF','APGD':'#000000','P-SPG-FB':'#FF1700','MJWarp':'#00FF00'}
def read(rel,allow_partial=False):
 p=ROOT/rel if rel.startswith('archive_') else RAW/rel
 if not p.exists(): return None
 t=p.read_text(errors='replace'); cur=None; r={}; f=[]
 for line in t.splitlines():
  m=ITER.search(line)
  if m: cur=int(m.group(1))
  m=REWARD.search(line)
  if m and cur is not None:r[cur]=float(m.group(1))
  m=FPS.search(line)
  if m and cur is not None:f.append(float(m.group(1)))
 if not r or (not allow_partial and ('Training time:' not in t or 'exit=0' not in t)): return None
 x=np.array(sorted(r)); y=np.array([r[i] for i in x]); fps=float(np.median(f)) if f else float('nan'); return x,y,fps
def smooth(y,span=35):
 a=2/(span+1); e=np.empty_like(y); e[0]=y[0]
 for i in range(1,len(y)): e[i]=a*y[i]+(1-a)*e[i-1]
 h=max(3,span//2); sd=np.array([np.std(y[max(0,i-h):min(len(y),i+h+1)]) for i in range(len(y))]); return e,e-sd,e+sd
for env,(jac,mjw,budget,coupling,post) in BASE.items():
 runs=[('Jacobi',jac),('MJWarp',mjw),('APGD',f'{env}/newton_dvi_apgd/contact_{budget}/run.log'),('P-SPG-FB',f'{env}/newton_dvi_pspg/contact_{budget}_coupling{coupling}_post_{str(post).lower()}/run.log')]; fig,ax=plt.subplots(figsize=(3.35,2.65),constrained_layout=False); n=0
 for name,rel in runs:
  z=read(rel, name in ('APGD','P-SPG-FB'))
  if z is None: continue
  x,y,f=z; e,lo,hi=smooth(y); label=f'{name} [{f/1000:.0f}k FPS]' if np.isfinite(f) else name+' [FPS pending]'; ax.plot(x,e,color=COL[name],lw=0.75,label=label); ax.fill_between(x,lo,hi,color=COL[name],alpha=.38,lw=0); n+=1
 ax.set_title(env[:1].upper()+env[1:]); ax.set_xlabel('PPO iteration'); ax.set_ylabel('Mean episode reward'); ax.set_xlim(left=0); ax.grid(True,which='major',color='#999',alpha=.35,lw=.5); ax.grid(True,which='minor',color='#bbb',alpha=.18,lw=.35); ax.tick_params(direction='in',top=True,right=True); ax.legend(frameon=False,fontsize=7,loc='best',handlelength=1.6,handletextpad=.35,borderaxespad=.25); fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.84)
 base=OUT/f'{env}_jacobi_mjwarp_apgd_progress'; [fig.savefig(base.with_suffix(s),dpi=600) if s=='.png' else fig.savefig(base.with_suffix(s)) for s in ('.png','.pdf','.svg')]; plt.close(fig); print(env,n)
