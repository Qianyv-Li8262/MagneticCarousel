"""Height-Hermite LUT refinement; validate field against independent surface integral."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import height_response_audit as a
sys.path.insert(0,str(a.ROOT.parent/'research/q235'))
import assess_apparatus_gap as exact
H=np.array([25.,27.5,30.,32.5,35.,37.5,39.62]);norm=H*4/70
tables=np.array([np.fromfile(a.OUT/f'height_{h:g}_fp64.bin',dtype='<f8').reshape(5,600,4) for h in H])
hermite=CubicHermiteSpline(norm,tables[:,:,:,:2],tables[:,:,:,2:4],axis=0)
def make_table(height):
 path=a.OUT/f'height_{height:g}_fp64.bin'
 if path.exists():return
 z=height*4/70;v=hermite(z);dv=hermite(z,1);np.concatenate([v,dv],axis=2).astype('<f8').tofile(path)
def validate(height):
 a.config(height);points=np.array([[r*4/70*np.cos(phi),r*4/70*np.sin(phi)] for r in [65,70,75] for phi in [0,.09,.2,.35]])
 reference=exact.field(points,height*4/70,32,128);state=np.zeros(15);state[5]=1;calc=[]
 for p in points:state[1:3]=p;calc.append(a.m.material_diagnostics(state,a.P)[:3])
 calc=np.array(calc);return dict(height_mm=height,relative_field_norm_error=float(np.linalg.norm(calc-reference)/np.linalg.norm(reference)),max_field_error_G=float(np.linalg.norm(calc-reference,axis=1).max()*a.common.MS*a.common.MU0*1e4))
if __name__=='__main__':
 heights=[26.,26.5,26.75,27.];runs=[];checks=[];stats={}
 for h in heights:
  make_table(h);checks.append(validate(h));stats[str(h)]=a.fieldstats(h);runs.append(a.run(h,300));(a.OUT/'height_refine_summary.json').write_text(json.dumps(dict(runs=runs,checks=checks,field_stats=stats),indent=2))
 good=sorted([q for q in runs if q['error'] is None and q['period_s'] and q['stationary']],key=lambda q:abs(q['period_s']-2.48587994094))
 long=[a.run(q['height_mm'],600,.00025,True) for q in good[:2]];(a.OUT/'height_refine_long.json').write_text(json.dumps(long,indent=2))
