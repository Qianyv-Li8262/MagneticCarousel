"""Deterministic local refinements from startup, small-overshoot and waiting branches."""
import json,numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares
import fit_video56c6_scale as g
seeds=[('startup',[.84280685,554.30314/g.f.BASE['mu'],.23816504,.00581625713/g.f.BASE['K'],1.03]),('smallover',[1.2,1,.4,.2,1.03]),('waiting',[1.2,1200/g.f.BASE['mu'],g.f.BASE['c'],.70,1.03])]
def optimize(seed):
 name,start=seed;count=0;best=1e9;history=[]
 def fun(x):
  nonlocal count,best
  r=g.evaluate(list(x)+[.58],name+'_'+str(count));count+=1;history.append(dict(key=r['key'],native_key=r['native_key'],score=r['score']))
  if r['score']<best:
   best=r['score'];(g.OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
  return np.asarray(r['residual'])
 result=least_squares(fun,start,bounds=([.4,300/g.f.BASE['mu'],.15,.08,.90],[3,2000/g.f.BASE['mu'],.6,1.8,1.15]),diff_step=.008,max_nfev=20,x_scale='jac',ftol=.002,xtol=.0005,gtol=.01)
 rec=dict(seed=name,x=result.x.tolist(),cost=result.cost,nfev=result.nfev,message=result.message,history=history)
 (g.OUT/('optimization_'+name+'.json')).write_text(json.dumps(rec,indent=2));return rec
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(optimize,seeds))
(g.OUT/'optimizations.json').write_text(json.dumps(results,indent=2))
