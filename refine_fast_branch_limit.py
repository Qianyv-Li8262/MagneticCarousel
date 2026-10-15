import json,numpy as np
import branch_height_audit as b
def state_for(height,tag):
 paths=[]
 for p in b.OUT.glob('*.json'):
  q=json.loads(p.read_text())
  if isinstance(q,dict) and q.get('height_mm')==height and q.get('tag')==tag:paths.append(p)
 return np.load(paths[0].with_suffix('.npz'))['final_state']
if __name__=='__main__':
 seed=state_for(36.,'fast_up');s=seed.copy();results=[]
 for height in [36.1,36.2,36.3,36.4]:
  q,s=b.evolve(height,state=s,seconds=300,tag='fast_limit_fine_height');results.append(q);(b.OUT/'fast_limit_refinement.json').write_text(json.dumps(results,indent=2))
  if q['error']:break
 # Revisit close-to-video candidates with smaller step and long observation.
 near=[q for q in results if q['error'] is None and q.get('period_s')]
 if near:
  q=min(near,key=lambda q:abs(q['period_s']-2.48587994094));s=state_for(q['height_mm'],'fast_limit_fine_height');r,s=b.evolve(q['height_mm'],state=s,seconds=900,dt=.00025,tag='fast_limit_long');(b.OUT/'fast_limit_long.json').write_text(json.dumps(r,indent=2))
