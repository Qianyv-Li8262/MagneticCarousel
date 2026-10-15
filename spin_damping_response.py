"""SI-calibrated Z-spin damping sensitivity, preserving branch seed states."""
from pathlib import Path
import importlib.util,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import branch_height_audit as b
a=b.a;ROOT=a.ROOT;OUT=ROOT/'runs/spin-damping-20261013';OUT.mkdir(exist_ok=True)
binary=next((ROOT/'scratch/spin-damping-20261013-source').glob('spin_damping_audit*.pyd'))
spec=importlib.util.spec_from_file_location('spin_damping_audit',binary)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P=m.SolverParams(*[getattr(a.P,key) for key in ['alpha','delta','gamma_g','gamma','gamma_m']]);P.reverse_push_acceleration=0
INERTIA=.4*a.common.MASS*.006**2
# Provisional calibration from the user's 1--2 s spin-period observation.
# This is an isolated trial value, conditional on the original field/material.
TRIAL_TAU_Z_S=15.
TRIAL_C_Z_N_M_S=INERTIA/TRIAL_TAU_Z_S
TRIAL_LAMBDA_Z=1/(a.OMEGA*TRIAL_TAU_Z_S)

def config(height,tau):
 m.configure_height_table(str(a.OUT/f'height_{height:g}_fp64.bin'))
 m.configure_material(*a.common.BASE);m.configure_magnet_moment_scale(1.)
 m.configure_laser_tilt(a.common.TILT,np.pi/2);m.configure_rolling_resistance(1e-5,1e-4,.004)
 m.configure_normal_spin_damping(0 if tau is None else 1/(a.OMEGA*tau))

def seeds():
 records=json.loads((b.OUT/'long_confirmation.json').read_text());out=[]
 for q in records:
  if not q['stationary']:continue
  height=q['height_mm'];T=q['period_s'];name='parked' if T is None else ('fast' if T<2 else 'slow')
  for p in b.OUT.glob('*.json'):
   if p.name in ['long_confirmation.json','initial_grid.json','continuation.json']:continue
   try:rec=json.loads(p.read_text())
   except Exception:continue
   if isinstance(rec,dict) and rec.get('tag')=='long_confirmation' and rec.get('height_mm')==height and rec.get('initial_spec')==q['initial_spec']:
    state=np.load(p.with_suffix('.npz'))['final_state'];out.append((height,name,state.tolist()));break
 return out

def run(task):
 height,name,initial,tau,seconds,dt=task
 key=hashlib.sha256(json.dumps(task).encode()).hexdigest()[:20];path=OUT/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 config(height,tau);state=np.array(initial);mode=direction=None;step=0;chunk=round(.02*a.OMEGA/dt);nmax=round(seconds*a.OMEGA/dt);rows=[np.r_[0,state,0]];error=None;start=time.perf_counter()
 try:
  while step<nmax:
   n=min(1 if mode is None else chunk,nmax-step)
   state,mode,direction=m.evolve_trajectory(state,dt,n,P,mode,direction,step);step+=n;rows.append(np.r_[step*dt/a.OMEGA,state,float(mode)])
 except RuntimeError as e:error=str(e)
 rows=np.array(rows);u=rows[rows[:,0]>=max(0,rows[-1,0]-60)];z=u[:,12]*a.OMEGA
 rec=dict(height_mm=height,seed_branch=name,tau_s=tau,lambda_z=0 if tau is None else 1/(a.OMEGA*tau),c_z_N_m_s=0 if tau is None else INERTIA/tau,seconds=seconds,dt=dt,completed_s=float(rows[-1,0]),error=error,spin_z_mean_rad_s=float(z.mean()),spin_z_rms_rad_s=float(np.sqrt(np.mean(z*z))),spin_z_max_abs_rad_s=float(abs(z).max()),wall_s=time.perf_counter()-start,**b.classify(rows,error))
 path.write_text(json.dumps(rec,indent=2));np.savez_compressed(path.with_suffix('.npz'),rows=rows,final_state=state)
 print(json.dumps({k:rec[k] for k in ['height_mm','seed_branch','tau_s','period_s','spin_z_mean_rad_s','stationary','error']}),flush=True);return rec

def free_decay():
 config(39.62,1.);m.configure_magnet_moment_scale(1e-12);m.configure_laser_tilt(0,0)
 state=np.zeros(15);state[1]=.07*a.K;state[5]=1;state[11]=10/a.OMEGA
 state,mode,direction=m.evolve_trajectory(state,.0005,round(3*a.OMEGA/.0005),P,None,None,0)
 elapsed=round(3*a.OMEGA/.0005)*.0005/a.OMEGA
 got=float(state[11]*a.OMEGA);expected=10*np.exp(-elapsed)
 rec=dict(tau_s=1,elapsed_s=elapsed,observed_rad_s=got,expected_rad_s=float(expected),relative_error=float(abs(got/expected-1)),inertia_kg_m2=INERTIA,omega_ref_rad_s=a.OMEGA)
 assert rec['relative_error']<1e-6,rec
 for bad in [-1,float('nan'),float('inf')]:
  try:m.configure_normal_spin_damping(bad)
  except ValueError:pass
  else:raise AssertionError('invalid damping accepted')
 (OUT/'free_decay.json').write_text(json.dumps(rec,indent=2));return rec

if __name__=='__main__':
 print(free_decay(),flush=True)
 tasks=[(height,name,state,tau,300,.0005) for height,name,state in seeds() for tau in [None,100,30,10,3,1,.3,.1]]
 results=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for r in pool.map(run,tasks):results.append(r);(OUT/'scan_summary.json').write_text(json.dumps(results,indent=2))
