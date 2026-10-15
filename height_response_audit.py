"""Fixed magnet moment/material; change the actual height-specific field only."""
from pathlib import Path
import importlib.util,json,time,hashlib
import numpy as np
import material_response_audit as common
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/height-response-20261013'
BINARY=next((ROOT/'scratch/height-response-20261013-source').glob('height_audit*.pyd'));spec=importlib.util.spec_from_file_location('height_audit',BINARY);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P=m.SolverParams(*[getattr(common.params,key) for key in ['alpha','delta','gamma_g','gamma','gamma_m']]);P.reverse_push_acceleration=0;OMEGA=common.OMEGA;K=common.KSPACE
def config(height):
 m.configure_height_table(str(OUT/f'height_{height:g}_fp64.bin'));m.configure_material(*common.BASE);m.configure_magnet_moment_scale(1.);m.configure_laser_tilt(common.TILT,np.pi/2);m.configure_rolling_resistance(1e-5,1e-4,.004)
def fieldstats(height):
 config(height);stats=[]
 for radius in [65,70,75]:
  state=np.zeros(15);state[5]=1.;state[1]=radius*.001*K;H=[];grad=[];force=[]
  for phi in np.arange(256)*np.pi/2/256:
   state[0]=phi;d=m.material_diagnostics(state,P);H.append(d[:3]*common.MS);grad.append(d[18:]*common.MS*K);force.append(d[6:9]*common.MASS*OMEGA**2/K)
  H=np.array(H);g=np.array(grad);F=np.array(force);B=np.linalg.norm(H,axis=1)*common.MU0*1e4
  stats.append(dict(radius_mm=radius,field_min_G=float(B.min()),field_max_G=float(B.max()),field_rms_G=float(np.sqrt(np.mean(B*B))),field_mean_G=float(B.mean()),radial_force_mean_q0_N=float(F[:,0].mean()),normal_attraction_mean_q0_N=float(-F[:,2].mean()),gradient_components_rms_T_m=float(np.sqrt(np.mean(np.sum(g*g,axis=1)))*common.MU0)))
 return stats
def run(height,seconds=300,dt=.0005,save=False,moment_scale=1.):
 suffix='' if moment_scale==1 else f'_moment{moment_scale:.12g}'
 path=OUT/f'run_h{height:g}_s{seconds:g}_dt{dt:g}{suffix}.json'
 if path.exists():return json.loads(path.read_text())
 config(height);m.configure_magnet_moment_scale(moment_scale);state=common.animation.prepare_initial_state(m,P,dt,v0=-.075);mode=direction=None;step=0;chunk=round(.02*OMEGA/dt);maxsteps=round(seconds*OMEGA/dt);rows=[np.r_[0,state,0]];err=None;start=time.perf_counter()
 try:
  while step<maxsteps:
   n=min(1 if mode is None else chunk,maxsteps-step);state,mode,direction=m.evolve_trajectory(state,dt,n,P,mode,direction,step);step+=n;rows.append(np.r_[step*dt/OMEGA,state,float(mode)])
 except RuntimeError as e:err=str(e)
 rows=np.array(rows);completed=rows[-1,0];width=min(60,completed/3);wins=[]
 for j in [3,2,1]:
  u=rows[(rows[:,0]>=completed-j*width)&(rows[:,0]<=completed-(j-1)*width)];angle=np.unwrap(np.arctan2(u[:,3],u[:,2]));rate=np.polyfit(u[:,0]-u[:,0].mean(),angle,1)[0];r=np.linalg.norm(u[:,2:4],axis=1);v=-(u[:,2]*u[:,5]-u[:,3]*u[:,4])/r;X=np.c_[np.ones(len(u)),np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)];co=np.linalg.lstsq(X,v,rcond=None)[0]
  wins.append(dict(period_s=float(-2*np.pi/rate) if rate<0 else None,radius_mm=float(r.mean()/K*1000),radius_range_mm=[float(r.min()/K*1000),float(r.max()/K*1000)],reverse_fraction=float(np.mean(v>0)),amplitude_percent=float(np.hypot(co[1],co[2])/abs(co[0])*100) if abs(co[0])>1e-10 else None))
 periods=[w['period_s'] for w in wins];stationary=bool(err is None and all(p is not None for p in periods) and np.ptp(periods)/np.mean(periods)<.01 and np.ptp([w['radius_mm'] for w in wins])<.5 and min(w['reverse_fraction'] for w in wins)>.99)
 rec=dict(height_mm=height,magnet_moment_scale=moment_scale,material=common.BASE.tolist(),rpm=OMEGA*60/(2*np.pi),seconds=seconds,dt=dt,completed_s=float(completed),error=err,stationary=stationary,windows=wins,period_s=wins[-1]['period_s'],radius_mm=wins[-1]['radius_mm'],amplitude_percent=wins[-1]['amplitude_percent'],wall_s=time.perf_counter()-start)
 if save:np.savez_compressed(path.with_suffix('.npz'),rows=rows)
 path.write_text(json.dumps(rec,indent=2));print(json.dumps({k:rec[k] for k in ['height_mm','seconds','period_s','radius_mm','stationary','completed_s','error']}),flush=True);return rec
if __name__=='__main__':
 heights=[39.62,37.5,35,32.5,30,27.5,25];stats={str(h):fieldstats(h) for h in heights};(OUT/'field_stats.json').write_text(json.dumps(stats,indent=2))
 runs=[]
 for h in heights:runs.append(run(h));(OUT/'scan_summary.json').write_text(json.dumps(runs,indent=2))
 ranked=sorted([q for q in runs if q['error'] is None and q['period_s'] and q['windows'][-1]['reverse_fraction']>.99],key=lambda q:abs(np.log(q['period_s']/2.48587994094)))
 confirm=[run(q['height_mm'],600,.00025,True) for q in ranked[:3]];(OUT/'long_confirm.json').write_text(json.dumps(confirm,indent=2))
