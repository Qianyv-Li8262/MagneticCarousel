"""Long/half-step checks and prepared-release checks for viscous Z damping."""
import json
from concurrent.futures import ProcessPoolExecutor
import spin_damping_response as s

if __name__=='__main__':
 tasks=[(h,name,state,1.,900,.00025) for h,name,state in s.seeds()]
 # Initial velocities are SI; held magnetic memory is prepared by the same model.
 s.a.m=s.m;s.a.P=s.P
 for v in [-.11,-.25]:
  for tau in [1.,3.]:
   s.config(39.62,tau)
   state=s.b.initial(dict(velocity_m_s=v,spin_factor=1)).tolist()
   tasks.append((39.62,f'fresh_v{v:g}',state,tau,600,.00025))
 records=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for q in pool.map(s.run,tasks):records.append(q);(s.OUT/'verification.json').write_text(json.dumps(records,indent=2))
