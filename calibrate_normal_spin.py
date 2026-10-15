"""Infer damping from the observed nearly parked 1--2 s spin period."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import spin_damping_response as s

if __name__=='__main__':
 candidates=[]
 for p in s.OUT.glob('*.json'):
  try:q=json.loads(p.read_text())
  except Exception:continue
  if isinstance(q,dict) and q.get('height_mm')==39.62 and q.get('seed_branch')=='parked' and q.get('tau_s')==1. and q.get('seconds')==900:
   state=np.load(p.with_suffix('.npz'))['final_state'].tolist();break
 else:raise RuntimeError('verified low-spin seed unavailable')
 tasks=[(39.62,'parked_low_spin',state,tau,600,.00025) for tau in [10.,12.,15.,18.,20.,22.,25.,30.]]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for q in pool.map(s.run,tasks):
   q['spin_period_s']=2*np.pi/abs(q['spin_z_mean_rad_s'])
   candidates.append(q);(s.OUT/'observed_spin_calibration.json').write_text(json.dumps(candidates,indent=2))
