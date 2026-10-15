"""Fixed reference magnet moment: joint material scan and long-run confirmation."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import qmc
import material_response_audit as a
def worker(item):
 label,material=item
 try:return a.run(np.array(material),seconds=180,label=label)
 except Exception as e:return dict(label=label,material=material,error=str(e))
if __name__=='__main__':
 tasks=[]
 for mur in [450,650,1000]:
  for c in [.16,.20,.25,.30]:
   for kfactor in [.3,1,3]:
    mat=a.BASE.copy();mat[0]=mur;mat[2]=c;mat[3]*=kfactor;tasks.append((f'joint mur{mur} C{c:g} Kx{kfactor:g}',mat.tolist()))
 sample=qmc.LatinHypercube(4,seed=20261013).random(24)
 for i,x in enumerate(sample):
  material=[10**(2+1.5*x[0]),10**(-3.7+2.2*x[1]),.02+.43*x[2],10**(-4+2.5*x[3])];tasks.append((f'joint Latin {i}',material))
 records=[]
 with ProcessPoolExecutor(max_workers=3) as executor:
  for rec in executor.map(worker,tasks):
   records.append(rec);(a.OUT/'joint_scan_summary.json').write_text(json.dumps(records,indent=2))
 good=[r for r in records if r.get('error') is None and r.get('period_s') and r.get('reverse_fraction',0)>.99 and r.get('radius_mean_mm',0)<90]
 ranked=sorted(good,key=lambda r:abs(np.log(r['period_s']/2.48587994094)))
 confirm=[]
 for r in ranked[:5]:
  material=np.array([r['material'][key] for key in ['mu_r','alpha_JA','C_JA','K']]);confirm.append(a.run(material,seconds=600,dt=.00025,save=True,label='600s '+r['label']))
 (a.OUT/'joint_long_confirm.json').write_text(json.dumps(confirm,indent=2))
 print('CONFIRMED',json.dumps([{k:r.get(k) for k in ['label','material','period_s','radius_mean_mm','stationary','error']} for r in confirm]),flush=True)
