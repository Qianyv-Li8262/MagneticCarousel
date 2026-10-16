"""Refit original eight tracked free-release segments at prescribed coefficients.
Only new audit outputs are written. Each hypothesis refits downhill direction
and all segment initial positions and velocities, using forward trajectories.
"""
from pathlib import Path
import sys,json,inspect
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'mu_tst'))
import rolling_resistance as rr
OUT=Path(__file__).parent/'current_blow'
source=inspect.getsource(rr.forward_joint)
source=source.replace('def forward_joint(', 'def fixed_forward(')
source=source.replace('tilt_deg=None):','tilt_deg=None,fixed_c=None):')
source=source.replace('lower[nb:ng]=0', 'lower[nb:ng]=0\n if fixed_c is not None:\n  lower[nb]=fixed_c;upper[nb]=fixed_c+1e-12;p[nb]=fixed_c+5e-13')
scope=rr.__dict__.copy();exec(source,scope);fit=scope['fixed_forward']
segments=rr.load_segments();initial=rr.fit_integral(segments,'constant')
angle=float(np.degrees(np.arctan(.002/1.05))); factor=7/(5*9.80665)
results=[];predictions={}
for mu in [.0001,.0013,.0049,.0063]:
    result,preds=fit(segments,'constant',initial,tilt_deg=angle,fixed_c=mu/factor,max_nfev=100)
    result['mu_fixed']=mu
    result['launch_rms_mm']=[float(np.sqrt(np.mean(np.sum((pr-s['q'])**2,axis=1)))*1000) for pr,s in zip(preds,segments)]
    results.append(result);predictions[mu]=preds
    print(json.dumps({k:result[k] for k in ['mu_fixed','rms_mm','success','launch_rms_mm']}),flush=True)
for mu in [.0013,.0049]:
    result,preds=fit(segments,'constant',initial,fixed_c=mu/factor,max_nfev=100)
    result['mu_fixed']=mu;result['tilt_free']=True
    result['tilt_inferred_deg']=float(np.degrees(np.arcsin(np.linalg.norm(result['params'][:2])*1.4/9.80665)))
    results.append(result)
    print(json.dumps({k:result[k] for k in ['mu_fixed','rms_mm','success','tilt_inferred_deg']}),flush=True)
(OUT/'video_resistance_profile.json').write_text(json.dumps(results,indent=2),encoding='utf8')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,k in zip(axs.flat,[0,1,3,5]):
    s=segments[k]; t=s['t'];q=s['q'];direction=np.mean(s['v'],axis=0);direction/=np.linalg.norm(direction)
    straight=np.stack([np.polyval(np.polyfit(t,q[:,j],1),t) for j in [0,1]],axis=1)
    ax.plot(t,(q-straight)@direction*1000,'o',ms=3,label='Observed position curvature')
    for mu in [.0001,.0013,.0049]:
        ax.plot(t,(predictions[mu][k]-straight)@direction*1000,label=f'C = {mu:g}')
    ax.set_title(f'Original release {k+1}');ax.set_xlabel('Time in segment (s)');ax.set_ylabel('Along-track curvature (mm)');ax.grid(alpha=.25)
axs[0,0].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'video_resistance_profile.png',dpi=150)
