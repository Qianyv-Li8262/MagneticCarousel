"""Continue the separately discovered fast branch upwards in small height steps."""
import json,numpy as np
import branch_height_audit as b
if __name__=='__main__':
 height0=30.908847694813392;spec={'velocity_m_s':-.25,'spin_factor':1};record,state=b.evolve(height0,spec,seconds=900,dt=.00025,tag='long_confirmation');records=[]
 for height in np.r_[height0,np.arange(31.5,39.51,.5),39.62]:
  q,state=b.evolve(float(height),state=state,seconds=240,tag='fast_up');records.append(q);(b.OUT/'fast_continuation.json').write_text(json.dumps(records,indent=2))
  if q['error']:break
 # Revisit the last supported height with a longer, finer-step continuation.
 good=[q for q in records if not q['error']]
 if good:
  h=good[-1]['height_mm']
  paths=[]
  for p in b.OUT.glob('*.json'):
   r=json.loads(p.read_text())
   if isinstance(r,dict) and r.get('tag')=='fast_up' and r.get('height_mm')==h:paths.append(p)
  seed=np.load(paths[0].with_suffix('.npz'))['final_state'];q,s=b.evolve(h,state=seed,seconds=900,dt=.00025,tag='fast_up_last_fine');(b.OUT/'fast_continuation_fine.json').write_text(json.dumps(q,indent=2))
