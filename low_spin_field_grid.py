"""Zero-spin release, three fields and damping times, finite blowing only."""
import json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import brentq
import spin_damping_response as s
h=s.b.h;a=s.a;m=s.m;P=s.P
OUT=a.ROOT/'runs/low-spin-field-grid-20261013';OUT.mkdir(exist_ok=True)
PROTOCOLS={
 'release':(1.,0.),
 'blow_weak':(1.,.5),
 'blow_strong':(1.,1.),
 'half_tilt_blow':(.5,.5),
 'flat_blow':(0.,.5),
}

def heights():
 path=OUT/'field_heights.json'
 if path.exists():return json.loads(path.read_text())
 def rms(z):h.make_table(z);return a.fieldstats(z)[1]['field_rms_G']
 records=[]
 for gauss in [100,200,300]:
  z=brentq(lambda z:rms(z)-gauss,25.,40.2,xtol=1e-8)
  records.append(dict(target_G=gauss,height_mm=z,field_stats=a.fieldstats(z),independent_check=h.validate(z),height_interpolation_extrapolated=z>39.62))
 path.write_text(json.dumps(records,indent=2));return records

def run(task):
 gauss,z,tau,protocol,seconds,dt=task
 definition=PROTOCOLS[protocol]
 key=hashlib.sha256(json.dumps([task,list(definition)]).encode()).hexdigest()[:20];path=OUT/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 tilt,acceleration=definition[:2]
 duration=definition[2] if len(definition)>2 else .3
 s.config(z,tau);m.configure_laser_tilt(a.common.TILT*tilt,np.pi)
 P.reverse_push_acceleration=0
 a.m=m;a.P=P
 # Zero center velocity and zero spin in all three axes; held JA preparation.
 state=s.b.initial(dict(velocity_m_s=0,spin_factor=0,spin_z_rad_s=0))
 mode=direction=None;step=0;nmax=round(seconds*a.OMEGA/dt);chunk=round(.02*a.OMEGA/dt)
 on=round(30*a.OMEGA/dt);off=round((30+duration)*a.OMEGA/dt)
 rows=[np.r_[0,state,0]];error=None;start=time.perf_counter()
 try:
  while step<nmax:
   active=acceleration>0 and on<=step<off
   P.reverse_push_acceleration=acceleration*a.K/a.OMEGA**2 if active else 0
   n=min(1 if mode is None else chunk,nmax-step)
   if step<on:n=min(n,on-step)
   elif step<off:n=min(n,off-step)
   state,mode,direction=m.evolve_trajectory(state,dt,n,P,mode,direction,step)
   step+=n;rows.append(np.r_[step*dt/a.OMEGA,state,float(mode)])
 except RuntimeError as e:error=str(e)
 finally:P.reverse_push_acceleration=0
 rows=np.array(rows);cl=s.b.classify(rows,error)
 steady=cl['stationary'] and cl['branch']=='clockwise' and cl.get('period_s') is not None and rows[-1,0]>seconds-.1
 tail=rows[:,0]>=max(0,rows[-1,0]-180);u=rows[tail]
 theta=np.unwrap(np.arctan2(u[:,3],u[:,2]));reverse=-theta
 crossings=[]
 for level in np.arange(np.ceil(reverse.min()/(2*np.pi)),np.floor(reverse.max()/(2*np.pi))+1)*2*np.pi:
  idx=np.where((reverse[:-1]<level)&(reverse[1:]>=level))[0]
  if len(idx)==1:
   i=idx[0];crossings.append(float(u[i,0]+(level-reverse[i])/(reverse[i+1]-reverse[i])*(u[i+1,0]-u[i,0])))
 lap=np.diff(sorted(crossings));r=np.linalg.norm(u[:,2:4],axis=1);v=-(u[:,2]*u[:,5]-u[:,3]*u[:,4])/r
 X=np.c_[np.ones(len(u)),np.cos(theta),np.sin(theta),np.cos(2*theta),np.sin(2*theta)]
 co=np.linalg.lstsq(X,v,rcond=None)[0]
 record=dict(field_G=gauss,height_mm=z,tau_z_s=tau,c_z_N_m_s=s.INERTIA/tau,lambda_z=1/(a.OMEGA*tau),protocol=protocol,tilt_rad=a.common.TILT*tilt,downhill_direction='+y',initial_center_velocity_m_s=0,initial_spin_rad_s=[0,0,0],blow_acceleration_m_s2=acceleration,blow_start_s=30 if acceleration else None,blow_duration_s=duration if acceleration else 0,blow_force_N=a.common.MASS*acceleration,seconds=seconds,dt=dt,completed_s=float(rows[-1,0]),error=error,stable_clockwise_orbit=bool(steady),spin_z_last180_mean_rad_s=float(u[:,12].mean()*a.OMEGA),orbit_lap_periods_s=lap.tolist(),orbit_lap_period_mean_s=float(lap.mean()) if len(lap) else None,orbit_lap_period_cv_percent=float(lap.std()/lap.mean()*100) if len(lap)>1 else None,speed_first_harmonic_percent=float(np.hypot(co[1],co[2])/abs(co[0])*100) if abs(co[0])>1e-8 else None,wall_s=time.perf_counter()-start,**cl)
 # A spatial harmonic fit is ill-conditioned for a parked ball with no laps.
 if not steady:record['speed_first_harmonic_percent']=None
 path.write_text(json.dumps(record,indent=2));np.savez_compressed(path.with_suffix('.npz'),rows=rows,final_state=state)
 print(json.dumps({k:record[k] for k in ['field_G','tau_z_s','protocol','period_s','stable_clockwise_orbit','spin_z_last180_mean_rad_s','completed_s']}),flush=True)
 return record

if __name__=='__main__':
 fields=heights();results=[]
 # Protocols are predefined: no ongoing force or inherited high-spin seed.
 tasks=[(f['target_G'],f['height_mm'],tau,p,600,.0005) for p in PROTOCOLS for f in fields for tau in [12.,15.,23.]]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for r in pool.map(run,tasks):results.append(r);(OUT/'scan_summary.json').write_text(json.dumps(results,indent=2))
