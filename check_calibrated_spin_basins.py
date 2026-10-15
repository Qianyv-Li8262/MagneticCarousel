"""Check the inferred damping from both release and preexisting high spin."""
import json
from concurrent.futures import ProcessPoolExecutor
import spin_damping_response as s

if __name__=='__main__':
 h,name,high=next(q for q in s.seeds() if q[0]==39.62)
 s.a.m=s.m;s.a.P=s.P;s.config(39.62,15.)
 fresh=s.b.initial(dict(velocity_m_s=0,spin_factor=0)).tolist()
 tasks=[(39.62,label,state,tau,600,.00025) for label,state in [('parked_high_spin',high),('parked_fresh',fresh)] for tau in [12.,15.,20.]]
 records=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for q in pool.map(s.run,tasks):records.append(q);(s.OUT/'calibrated_basins.json').write_text(json.dumps(records,indent=2))
