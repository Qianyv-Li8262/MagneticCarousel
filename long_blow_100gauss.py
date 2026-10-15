"""Longer finite air pushes from zero initial spin at 100 G."""
import json
from concurrent.futures import ProcessPoolExecutor
import low_spin_field_grid as g
OUT=g.a.ROOT/'runs/long-blow-100gauss-20261013';OUT.mkdir(exist_ok=True)
g.OUT=OUT
for acc in [.1,.2,.5]:
 for duration in [1.,3.,10.]:
  g.PROTOCOLS[f'blow_a{acc:g}_d{duration:g}']=(1.,acc,duration)

if __name__=='__main__':
 fields=json.loads((g.a.ROOT/'runs/low-spin-field-grid-20261013/field_heights.json').read_text())
 z=next(f['height_mm'] for f in fields if f['target_G']==100)
 results=[];tasks=[(100,z,tau,f'blow_a{acc:g}_d{duration:g}',600,.0005) for acc in [.1,.2,.5] for duration in [1.,3.,10.] for tau in [12.,15.,23.]]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for r in pool.map(g.run,tasks):results.append(r);(OUT/'scan_summary.json').write_text(json.dumps(results,indent=2))
