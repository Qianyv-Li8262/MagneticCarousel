import sys,json,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
count=0;best=1e9

def fun(x):
 global count,best
 cfg=dict(r=68.5,wz=x[4]*20,hold=2,sigma=x[0]*1e7,mu=x[1]*f.BASE['mu'],c=x[2],K=x[3]*f.BASE['K'])
 rec=f.run(cfg,tag='jointfit'+str(count));count+=1
 rr=np.nan_to_num(rec['residual'],nan=1e3,posinf=1e3,neginf=-1e3);rr[-3:-1]*=2.5
 score=float(np.mean(rr**2))
 if score<best:
  best=score;rec['constrained_score']=score;(f.OUT/'best_joint.json').write_text(json.dumps(rec,indent=2))
 return rr
r=least_squares(fun,[1.4,1200/f.BASE['mu'],.20,1,-1],bounds=([.4,.65,.18,.5,-4],[4,4,.3,1.8,1]),diff_step=.01,max_nfev=20,verbose=1,x_scale='jac',ftol=.003,xtol=.0005,gtol=.01)
(f.OUT/'joint_optimization.json').write_text(json.dumps(dict(x=r.x.tolist(),cost=r.cost,message=r.message,nfev=r.nfev),indent=2))
