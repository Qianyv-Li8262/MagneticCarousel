"""Refine material only: zero COM speed, zero spin, zero magnetic memory at release."""
import sys,json,numpy as np
from scipy.optimize import least_squares
import fit_video56c6 as f
OLD=f.OUT;B=json.loads((OLD/'startup_candidate.json').read_text())['parameters'];f.OUT=f.ROOT/'runs/video56c6-rest-release-20261017';f.OUT.mkdir(exist_ok=True)
base=dict(B,wz=0.,hold=0.,vt=0.,vr=0.,qz=0.,r=67.4)
count=0;best=1e9

def fun(x):
 global count,best
 cfg=dict(base,sigma=x[0]*1e7,mu=x[1]*f.BASE['mu'],c=x[2],K=x[3]*f.BASE['K'])
 rec=f.run(cfg,tag='rest_refine'+str(count));count+=1
 means=np.array(rec['curve_means']);target=np.array([b['speed'] for b in f.BINS])
 rr=list((means-target)/25.)
 # Secondary constraints: permit the few-percent period bias accepted by the user.
 rr += [(rec['holdout']['period']-f.TARGET_PERIOD)/.12,(rec['holdout']['radius']-75.8)/3]
 if rec['status'].split('status=')[-1]!='0':rr=[1e3]*len(rr)
 rr=np.nan_to_num(rr,nan=1e3,posinf=1e3,neginf=-1e3);score=float(np.mean(rr**2))
 if score<best:
  best=score;rec['startup_priority_score']=score;(f.OUT/'best_rest.json').write_text(json.dumps(rec,indent=2))
 return rr
x0=[B['sigma']/1e7,B['mu']/f.BASE['mu'],B['c'],B['K']*1.05/f.BASE['K']]
r=least_squares(fun,x0,bounds=([.6,400/f.BASE['mu'],.19,.7],[1.4,1000/f.BASE['mu'],.30,1.5]),diff_step=.005,max_nfev=25,verbose=1,x_scale='jac',ftol=.001,xtol=.0002,gtol=.01)
(f.OUT/'optimization.json').write_text(json.dumps(dict(x=r.x.tolist(),cost=r.cost,message=r.message,nfev=r.nfev,objective='startup curve first; period secondary',fixed_initial_state=dict(v=0,omega=0,q=0,eddy_modes=0,held_turns=0,radius_mm=67.4)),indent=2))
