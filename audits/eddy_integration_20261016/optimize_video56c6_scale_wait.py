"""Block-normalized objective: explicitly require the observed quiet interval."""
import json,numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor
import fit_video56c6_scale as g
seeds=[('lowmu_wait',[.6,300/g.f.BASE['mu'],.29,1,1.03]),('highmu_wait',[1.2,1200/g.f.BASE['mu'],g.f.BASE['c'],.75,1.03]),('smallover_wait',[1.2,1,.4,.2,1.03])]
def optimize(seed):
 name,start=seed;count=0;best=1e9;hist=[]
 def fun(x):
  nonlocal count,best
  r=g.evaluate(list(x)+[.58],name+'_'+str(count));count+=1
  v=np.array(r['model_speed_CV']);radius=np.array(r['model_radius_CV']);quiet=(g.TARGETS[:,0]>=9)&(g.TARGETS[:,0]<18);launch=(g.TARGETS[:,0]>=18)&(g.TARGETS[:,0]<35);late=g.TARGETS[:,0]>=35
  rr=np.r_[((v-g.TARGETS[:,2])[quiet]/5/np.sqrt(quiet.sum())),((v-g.TARGETS[:,2])[launch]/25/np.sqrt(launch.sum())),((v-g.TARGETS[:,2])[late]/30/np.sqrt(late.sum())),(radius-g.TARGETS[:,3])/1.2/np.sqrt(len(radius)),(r['period_s']-g.f.TARGET_PERIOD)/.035,max(0,r['overshoot_CV_mm']-1.5)/1.]
  if r['status'].split('status=')[-1]!='0':rr[:]=1e3
  rr=np.nan_to_num(rr,nan=1e3,posinf=1e3,neginf=-1e3);score=float(np.sum(rr**2));hist.append(dict(key=r['key'],score=score))
  print('WAIT',name,count,round(score,2),flush=True)
  if score<best:
   best=score;r['block_wait_score']=score;(g.OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
  return rr
 result=least_squares(fun,start,bounds=([.35,150/g.f.BASE['mu'],.15,.08,.90],[3.,2000/g.f.BASE['mu'],.65,2.,1.15]),diff_step=.006,max_nfev=22,x_scale='jac',ftol=.002,xtol=.0004,gtol=.01)
 rec=dict(seed=name,x=result.x.tolist(),cost=result.cost,nfev=result.nfev,message=result.message,history=hist)
 (g.OUT/('optimization_'+name+'.json')).write_text(json.dumps(rec,indent=2));return rec
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:res=list(pool.map(optimize,seeds))
 (g.OUT/'wait_optimizations.json').write_text(json.dumps(res,indent=2))
