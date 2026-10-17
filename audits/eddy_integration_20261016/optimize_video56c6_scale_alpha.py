"""Release alpha/K degeneracy to check whether loop shape can improve post-wait acceleration."""
import json,numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares
import fit_video56c6_scale as g
b=json.loads((g.OUT/'best_weakeddy_wait.json').read_text())['x'][:5]
seeds=[('alpha_wait',b+[1.]),('alpha_pair_wait',[*b[:3],b[3]*1.8,b[4],2.])]
def optimize(seed):
 name,start=seed;count=0;best=1e9
 def fun(x):
  nonlocal count,best
  r=g.evaluate(list(x[:5])+[.58,x[5]],name+'_'+str(count));count+=1
  v=np.array(r['model_speed_CV']);radius=np.array(r['model_radius_CV']);quiet=(g.TARGETS[:,0]>=9)&(g.TARGETS[:,0]<18);launch=(g.TARGETS[:,0]>=18)&(g.TARGETS[:,0]<35);late=g.TARGETS[:,0]>=35
  rr=np.r_[((v-g.TARGETS[:,2])[quiet]/5/np.sqrt(quiet.sum())),((v-g.TARGETS[:,2])[launch]/25/np.sqrt(launch.sum())),((v-g.TARGETS[:,2])[late]/30/np.sqrt(late.sum())),(radius-g.TARGETS[:,3])/1.2/np.sqrt(len(radius)),(r['period_s']-g.f.TARGET_PERIOD)/.035,max(0,r['overshoot_CV_mm']-1.5)]
  if r['status'].split('status=')[-1]!='0':rr[:]=1e3
  rr=np.nan_to_num(rr,nan=1e3,posinf=1e3,neginf=-1e3);score=float(np.sum(rr**2))
  if score<best:
   best=score;r['block_wait_score']=score;(g.OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
  print('ALPHAWAIT',name,count,round(score,2),flush=True);return rr
 result=least_squares(fun,start,bounds=([.35,100/g.f.BASE['mu'],.15,.05,.90,.3],[3,2000/g.f.BASE['mu'],.7,2.5,1.15,4]),diff_step=.006,max_nfev=15,x_scale='jac',ftol=.003,xtol=.0005,gtol=.01)
 (g.OUT/('optimization_'+name+'.json')).write_text(json.dumps(dict(x=result.x.tolist(),cost=result.cost,nfev=result.nfev,message=result.message),indent=2))
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(optimize,seeds))
