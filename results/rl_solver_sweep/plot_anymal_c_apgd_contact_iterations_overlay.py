#!/usr/bin/env python3
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw'/'anymal_c'; OUT=ROOT/'paper_clean'/'anymal_c_progress'; OUT.mkdir(parents=True,exist_ok=True)
ITER=re.compile(r'Learning iteration\s+(\d+)/'); REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'); FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+)')
colors={'5':'#00A000','10':'#800080','20':'#000000','40':'#0000FF','60':'#FF1700'}
def read(p):
 t=(p/'run.log').read_text(errors='replace'); cur=None;r={};f=[]
 for line in t.splitlines():
  m=ITER.search(line)
  if m:cur=int(m.group(1))
  m=REWARD.search(line)
  if m and cur is not None:r[cur]=float(m.group(1))
  m=FPS.search(line)
  if m and cur is not None:f.append(float(m.group(1)))
 x=np.array(sorted(r),float);y=np.array([r[int(i)] for i in x]);return x,y,np.median(f)
def smooth(y,span=35):
 a=2/(span+1);e=np.empty_like(y);e[0]=y[0]
 for i in range(1,len(y)):e[i]=a*y[i]+(1-a)*e[i-1]
 h=max(3,span//2);sd=np.array([np.std(y[max(0,i-h):min(len(y),i+h+1)]) for i in range(len(y))]);return e,e-sd,e+sd
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.75,'lines.linewidth':.8,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.bbox':'tight'})
fig,ax=plt.subplots(figsize=(3.45,2.65));
for c in ('5','10','20','40','60'):
 x,y,f=read(RAW/f'apgd_coupling2_cache_on_contact{c}_compliance_0');e,lo,hi=smooth(y);ax.plot(x,e,color=colors[c],label=f'{c} contacts [{f/1000:.0f}k FPS]');ax.fill_between(x,lo,hi,color=colors[c],alpha=.22,lw=0)
ax.set_title('Anymal C');ax.set_xlabel('PPO iteration');ax.set_ylabel('Mean episode reward');ax.set_xlim(left=0);ax.grid(True,color='#999',alpha=.35,lw=.5);ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False,fontsize=7,loc='best',handlelength=1.6,handletextpad=.35,borderaxespad=.25);fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.84)
base=OUT/'anymal_c_apgd_contact_iterations_all_overlay'
for s,k in [('.png',{'dpi':600}),('.pdf',{}),('.svg',{})]:fig.savefig(base.with_suffix(s),**k)
plt.close(fig);print(base)
