#!/usr/bin/env python3
"""Plot newly validated Jacobi/post-stabilization runs against prior validated runs."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw'; OUT=ROOT/'figures'
ITER=re.compile(r'Learning iteration\s+(\d+)/'); REWARD=re.compile(r'Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'); FPS=re.compile(r'Steps per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def parse(rel, marker=False):
    p=RAW/rel; d=p.parent; text=p.read_text(errors='replace')
    if 'Training time:' not in text or 'exit=0' not in text or (marker and not (d/'COMPLETED').exists()): raise RuntimeError(str(p))
    cur=None; r={}; f=[]
    for line in text.splitlines():
        m=ITER.search(line)
        if m: cur=int(m.group(1))
        m=REWARD.search(line)
        if m and cur is not None: r[cur]=float(m.group(1))
        m=FPS.search(line)
        if m and cur is not None and cur>=1: f.append(float(m.group(1)))
    x=np.array(sorted(r)); y=np.array([r[i] for i in x])
    return x,y,float(np.median(f)) if f else float('nan'),y[-1]

def band(y,span=35):
    a=2/(span+1); e=np.empty_like(y); e[0]=y[0]
    for i in range(1,len(y)): e[i]=a*y[i]+(1-a)*e[i-1]
    lo=np.empty_like(y); hi=np.empty_like(y); h=max(3,span//2)
    for i in range(len(y)):
        s=y[max(0,i-h):min(len(y),i+h+1)]; lo[i]=e[i]-np.std(s); hi[i]=e[i]+np.std(s)
    return e,lo,hi

specs={
'ant': [('Prior Jacobi, contact 10','ant/initial_contact_10/jacobi.log','#0000FF','- ',False),('Prior APGD, contact 10','ant/initial_contact_10/apgd.log','#111111','-',False),('Prior P-SPG-FB, contact 10','ant/initial_contact_10/pspg.log','#FF1700','-',False),('Prior MJWarp (native)','ant/initial_contact_10/mjwarp.log','#00FF00','-',False),('New Jacobi, coupling 1 + post-stabilization, contact 10','ant/newton_dvi_post_stabilize/contact_10/run.log','#9467bd','--',True)],
'humanoid':[('New Jacobi, coupling 1 + post-stabilization, contact 10','humanoid/newton_dvi_post_stabilize/contact_10/run.log','#9467bd','--',True)],
'h1':[('Prior Jacobi, contact 5','h1/newton_dvi/contact_5/run.log','#0000FF','-',True),('Prior Jacobi, contact 10','h1/newton_dvi/contact_10/run.log','#0000FF','--',True),('Prior APGD, contact 5','h1/newton_dvi_apgd/contact_5/run.log','#111111','-',True),('Prior APGD, contact 10','h1/newton_dvi_apgd/contact_10/run.log','#111111','--',True),('Prior P-SPG-FB, contact 5','h1/newton_dvi_pspg/contact_5/run.log','#FF1700','-',True),('Prior P-SPG-FB, contact 10','h1/newton_dvi_pspg/contact_10/run.log','#FF1700','--',True),('Prior MJWarp (native)','h1/newton_mjwarp/contact_10/run.log','#00FF00','-',True),('New Jacobi, coupling 1 + post-stabilization, contact 10','h1/newton_dvi_post_stabilize/contact_10/run.log','#9467bd','--',True)],
'go2':[('Prior Jacobi, coupling 1, contact 5','go2/newton_dvi/contact_5/run.log','#0000FF','-',True),('Prior Jacobi, coupling 1, contact 10','go2/newton_dvi/contact_10/run.log','#0000FF','--',True),('Prior Jacobi, coupling 2, contact 5','go2/newton_dvi_coupling2/contact_5/run.log','#17becf','-',True),('Prior Jacobi, coupling 2, contact 10','go2/newton_dvi_coupling2/contact_10/run.log','#17becf','--',True),('Prior Jacobi, coupling 2, contact 15','go2/newton_dvi_coupling2/contact_15/run.log','#17becf',':',True),('Prior APGD, contact 5','go2/newton_dvi_apgd/contact_5/run.log','#111111','-',True),('Prior APGD, contact 10','go2/newton_dvi_apgd/contact_10/run.log','#111111','--',True),('Prior P-SPG-FB, contact 5','go2/newton_dvi_pspg/contact_5/run.log','#FF1700','-',True),('Prior P-SPG-FB, contact 10','go2/newton_dvi_pspg/contact_10/run.log','#FF1700','--',True),('Prior MJWarp (native)','go2/newton_mjwarp/contact_10/run.log','#00FF00','-',True),('Prior post-stabilization, contact 15','go2/newton_dvi_post_stabilize/contact_15/run.log','#9467bd',':',True)]}

OUT.mkdir(exist_ok=True)
for env,runs in specs.items():
    fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True); n=0
    for label,rel,c,ls,need in runs:
        try: x,y,fps,final=parse(rel,need)
        except Exception as e: print('SKIP',e); continue
        e,lo,hi=band(y); ax.plot(x,e,color=c,ls=ls.strip(),lw=1.9,label=f'{label} [{fps/1000:.1f}k; final {final:.2f}]'); ax.fill_between(x,lo,hi,color=c,alpha=.08,lw=0); n+=1
    ax.set_title(env.upper()); ax.set_xlabel('PPO iteration'); ax.set_ylabel('Mean episode reward'); ax.set_xlim(left=0); ax.grid(alpha=.22); ax.legend(fontsize=7,loc='best')
    if env=='humanoid': ax.text(.01,.01,'No prior validated Humanoid-direct solver curves are present in the results tree.',transform=ax.transAxes,fontsize=8,color='#555')
    base=OUT/f'{env}_requested_vs_prior_reward_curves'
    for s,k in [('.png',{'dpi':220}),('.pdf',{}),('.svg',{})]: fig.savefig(base.with_suffix(s),**k)
    plt.close(fig); print(env,n,base.with_suffix('.png'))
