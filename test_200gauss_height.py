import json,numpy as np
from scipy.optimize import brentq
import refine_height_response as r
def field_rms(h):
 r.make_table(h);r.a.config(h);state=np.zeros(15);state[5]=1;state[1]=4;B=[]
 for phi in np.arange(256)*np.pi/2/256:state[0]=phi;d=r.a.m.material_diagnostics(state,r.a.P);B.append(np.linalg.norm(d[:3])*r.a.common.MS*r.a.common.MU0*1e4)
 return np.sqrt(np.mean(np.array(B)**2))
if __name__=='__main__':
 h=float(brentq(lambda h:field_rms(h)-200,30,32.5,xtol=1e-4));r.make_table(h);rec=dict(height_for_200G_rms_at_70mm=h,field_stats=r.a.fieldstats(h),validation=r.validate(h),run=r.a.run(h,600,.00025,True));(r.a.OUT/'200gauss_result.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
