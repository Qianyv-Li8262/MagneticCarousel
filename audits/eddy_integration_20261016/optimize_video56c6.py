import sys,json,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
base=dict(r=68.5,wz=-20,hold=2)
count=0;best=1e9

def fun(x):
 global count,best
 cfg=dict(base,sigma=x[0]*1e7,mu=x[1]*f.BASE['mu'],c=x[2],K=x[3]*f.BASE['K'])
 rec=f.run(cfg,tag='localfit'+str(count));count+=1
 if rec['score']<best:
  best=rec['score'];(f.OUT/'best_local.json').write_text(json.dumps(rec,indent=2))
 return np.nan_to_num(rec['residual'],nan=1e3,posinf=1e3,neginf=-1e3)
r=least_squares(fun,[.8,1,.22415155,1],bounds=([.4,.65,.15,.5],[2.5,2,.4,1.8]),diff_step=.015,max_nfev=25,verbose=1,x_scale='jac',ftol=.002,xtol=.001,gtol=.01)
(f.OUT/'optimization.json').write_text(json.dumps(dict(x=r.x.tolist(),cost=r.cost,message=r.message,nfev=r.nfev),indent=2))
