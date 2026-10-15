"""Multistability search: different SI initial states and height continuation."""
import json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import refine_height_response as h
a=h.a;OUT=a.ROOT/'runs/branch-height-20261013';OUT.mkdir(exist_ok=True)
def initial(spec):
 radius=spec.get('radius_mm',70)*.001;velocity=spec.get('velocity_m_s',-.11);spin_factor=spec.get('spin_factor',1);phase=spec.get('phase',.58);turns=spec.get('held_turns',2)
 s=np.zeros(15);s[5]=1;s[1]=radius*a.K;dt=.0005;n=round(turns*2*np.pi/dt);s[0]=phase-n*dt
 if n:s=a.m.magnetize_held_ball(s,dt,n,a.P)
 s[0]=phase;s[3:5]=[0,velocity*a.K/a.OMEGA];s[9:12]=[-velocity/(.006*a.OMEGA)*spin_factor,0,spec.get('spin_z_rad_s',0)/a.OMEGA]
 return s
def classify(rows,error):
 elapsed=rows[-1,0];width=min(60,elapsed/3);wins=[]
 for j in [3,2,1]:
  u=rows[(rows[:,0]>=elapsed-j*width)&(rows[:,0]<=elapsed-(j-1)*width)]
  if len(u)<8:continue
  angle=np.unwrap(np.arctan2(u[:,3],u[:,2]));rate=np.polyfit(u[:,0]-u[:,0].mean(),angle,1)[0];radius=np.linalg.norm(u[:,2:4],axis=1);v=-(u[:,2]*u[:,5]-u[:,3]*u[:,4])/radius
  wins.append(dict(angular_rate_rad_s=float(rate),signed_period_s=float(2*np.pi/rate) if abs(rate)>1e-8 else None,period_s=float(2*np.pi/abs(rate)) if abs(rate)>1e-8 else None,radius_mm=float(radius.mean()/a.K*1000),radius_std_mm=float(radius.std()/a.K*1000),tangent_speed_mean_m_s=float(v.mean()*a.OMEGA/a.K),reverse_fraction=float(np.mean(v>0)),rolling_fraction=float(np.mean(u[:,16]==0)),spin_z_mean_rad_s=float(u[:,12].mean()*a.OMEGA)))
 if not wins:return dict(windows=wins,branch='failed',stationary=False)
 rates=np.array([q['angular_rate_rad_s'] for q in wins]);radii=np.array([q['radius_mm'] for q in wins]);last=wins[-1]
 rotating=abs(last['angular_rate_rad_s'])>.1
 stable=bool(error is None and len(wins)==3 and np.ptp(radii)<.5 and (np.ptp(rates)/abs(rates.mean())<.01 if rotating else np.max(abs(rates))<.02))
 label='counterclockwise' if last['angular_rate_rad_s']>0 else 'clockwise'
 if not rotating:label='parked/very_slow'
 return dict(windows=wins,branch=label,stationary=stable,period_s=last['period_s'] if rotating else None,radius_mm=last['radius_mm'],angular_rate_rad_s=last['angular_rate_rad_s'])
def evolve(height,spec=None,state=None,seconds=300,dt=.0005,tag='',save=True):
 state_hash=None if state is None else hashlib.sha256(np.asarray(state,dtype='<f8').tobytes()).hexdigest()
 key=hashlib.sha256(json.dumps([height,spec,seconds,dt,tag,state_hash]).encode()).hexdigest()[:18];path=OUT/f'{key}.json'
 if path.exists():
  record=json.loads(path.read_text());data=np.load(path.with_suffix('.npz'));return record,data['final_state']
 h.make_table(height);a.config(height);s=initial(spec or {}) if state is None else state.copy();mode=direction=None;step=0;maxsteps=round(seconds*a.OMEGA/dt);chunk=round(.02*a.OMEGA/dt);rows=[np.r_[0,s,0]];error=None;began=time.perf_counter()
 try:
  while step<maxsteps:
   n=min(1 if mode is None else chunk,maxsteps-step);s,mode,direction=a.m.evolve_trajectory(s,dt,n,a.P,mode,direction,step);step+=n;rows.append(np.r_[step*dt/a.OMEGA,s,float(mode)])
 except RuntimeError as e:error=str(e)
 rows=np.array(rows);record=dict(height_mm=height,initial_spec=spec,input_state_sha256=state_hash,tag=tag,seconds=seconds,completed_s=float(rows[-1,0]),dt=dt,error=error,wall_s=time.perf_counter()-began,**classify(rows,error))
 path.write_text(json.dumps(record,indent=2));np.savez_compressed(path.with_suffix('.npz'),rows=rows if save else rows[::10],final_state=s)
 print(json.dumps({k:record.get(k) for k in ['height_mm','initial_spec','tag','period_s','branch','stationary','completed_s','error']}),flush=True);return record,s
def worker(task):
 height,spec=task
 try:return evolve(height,spec,tag='initial_grid')[0]
 except Exception as e:return dict(height_mm=height,initial_spec=spec,error=str(e),stationary=False,branch='exception')
def continuation():
 seed=np.load(a.OUT/'run_h26.91_s600_dt0.00025.npz')['rows'][-1,1:16].copy();records=[]
 # Both histories start from the same verified target-period orbit. State,
 # attitude and magnetic memory are carried forward, with fixed SI time units.
 for label,sequence in [('up',np.r_[26.91,np.arange(27.5,39.51,.5),39.62]),('down_then_up',[26.91,26.5,26.,25.5,25.,25.5,26.,26.5,26.91,27.5,30.908847694813392,35.,39.62])]:
  state=seed.copy()
  for height in sequence:
   q,state=evolve(float(height),state=state,seconds=180,tag='continuation_'+label);records.append(q);(OUT/'continuation.json').write_text(json.dumps(records,indent=2))
   if q['error']:break
 return records
if __name__=='__main__':
 specs=[dict(velocity_m_s=v,spin_factor=s) for v in [0,-.11,-.18,-.25,.18] for s in [0,1]]
 specs += [dict(radius_mm=r,velocity_m_s=-.18,spin_factor=1) for r in [65,75]]
 specs += [dict(velocity_m_s=-.18,spin_factor=1,spin_z_rad_s=z) for z in [-50,50]]
 specs += [dict(velocity_m_s=-.18,spin_factor=1,held_turns=n) for n in [0,8]]
 specs += [dict(velocity_m_s=-.18,spin_factor=1,phase=phase) for phase in [0,np.pi/8]]
 tasks=[(height,spec) for height in [39.62,30.908847694813392,26.91] for spec in specs];results=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for q in pool.map(worker,tasks):results.append(q);(OUT/'initial_grid.json').write_text(json.dumps(results,indent=2))
 # One representative of each observed rotating/parked cluster per height,
 # plus the nearest surviving trajectory to the video period, gets a restart
 # from its prepared initial state at half step and 900 s total duration.
 representatives=[]
 for height in [39.62,30.908847694813392,26.91]:
  found=[q for q in results if q['height_mm']==height and not q['error']]
  clusters=[]
  for q in sorted(found,key=lambda q:q.get('period_s') or 0):
   if not any(q['branch']==old['branch'] and ((q.get('period_s') is None and old.get('period_s') is None) or (q.get('period_s') and old.get('period_s') and abs(q['period_s']/old['period_s']-1)<.03)) for old in clusters):clusters.append(q)
  representatives.extend(clusters[:5])
 confirmed=[]
 for q in representatives:
  r,s=evolve(q['height_mm'],q['initial_spec'],seconds=900,dt=.00025,tag='long_confirmation');confirmed.append(r);(OUT/'long_confirmation.json').write_text(json.dumps(confirmed,indent=2))
 continuation()
