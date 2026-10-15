"""Restart stable branches from zero spin at half step for 1200 seconds."""
import json
from concurrent.futures import ProcessPoolExecutor
import low_spin_field_grid as g

if __name__=='__main__':
 records=json.loads((g.OUT/'scan_summary.json').read_text())
 selected=[]
 for field in [100,200,300]:
  for tau in [12.,15.,23.]:
   candidates=[r for r in records if r['field_G']==field and r['tau_z_s']==tau and r['stable_clockwise_orbit']]
   # Verify all different periods, not multiple repetitions of the same branch.
   clusters=[]
   for r in candidates:
    if not any(abs(r['period_s']/q['period_s']-1)<.01 for q in clusters):clusters.append(r)
   selected.extend(clusters)
 tasks=[(r['field_G'],r['height_mm'],r['tau_z_s'],r['protocol'],1200,.00025) for r in selected]
 confirmed=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for r in pool.map(g.run,tasks):confirmed.append(r);(g.OUT/'long_confirmation.json').write_text(json.dumps(confirmed,indent=2))
