"""Finite centre-force pushes of the fixed, horizontal low-spin candidate."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import fit_horizontal_spin_period as s
f=s.f
OUT=f.ROOT/'runs/blow-fast-branch-20261015';OUT.mkdir(exist_ok=True)
BASE=json.loads((f.OUT/'3414e055b2cdf35c17a3.json').read_text())
OMEGA=BASE['rpm']*2*np.pi/60
MASS=4*np.pi/3*f.BALL_R**3*f.DENSITY
SEED_ROWS=np.load(f.OUT/(BASE['key']+'.npz'))['rows']

def config():
    v=BASE['parameters'];alpha=f.KSPACE*f.BALL_R
    gamma=3*4*np.pi*1e-7*f.MS**2*f.KSPACE**2/(f.DENSITY*OMEGA**2)
    p=f.mod.SolverParams(alpha,3*1.225*.47/(8*f.DENSITY*f.BALL_R*f.KSPACE),
                         f.KSPACE*9.81/OMEGA**2,gamma,gamma/(.4*alpha**2))
    p.reverse_push_acceleration=0
    f.mod.configure_material(*[v[k] for k in f.KEYS[:4]])
    f.mod.configure_magnet_moment_scale(1.)
    f.mod.configure_laser_tilt(0,0)
    f.mod.configure_rolling_time_scale(1.)
    f.mod.configure_rolling_resistance(*[v[k] for k in f.KEYS[4:]])
    f.mod.configure_normal_spin_damping(1/(OMEGA*BASE['spin_tau_s']))
    return p

def seed(index):
    desired=SEED_ROWS[-1,0]-BASE['period_s']*index/4
    row=SEED_ROWS[np.argmin(abs(SEED_ROWS[:,0]-desired))]
    return row[1:16].copy(),float(row[0])

def metrics(rows,error,completed):
    tail=rows[rows[:,0]>=max(0,completed-180)]
    wins=[]
    for u in np.array_split(tail,3):
        if len(u)<5:return dict(stationary=False,outcome=('outside_field_domain' if error and 'kStepLUTOutOfDomain' in error else 'insufficient_data'))
        angle=np.unwrap(np.arctan2(u[:,3],u[:,2]));t=u[:,0]
        rate=float(np.polyfit(t-t.mean(),angle,1)[0]);r=np.linalg.norm(u[:,2:4],axis=1)
        vt=(u[:,2]*u[:,5]-u[:,3]*u[:,4])/r
        wins.append(dict(rate_rad_s=rate,period_s=-2*np.pi/rate if rate<-.01 else None,
                         radius_mean_mm=float(r.mean()/f.KSPACE*1000),
                         reverse_fraction=float(np.mean(vt<0)),
                         spin_rms_rad_s=float(np.sqrt(np.mean(u[:,12]**2))*OMEGA)))
    rates=np.array([w['rate_rad_s'] for w in wins]);radii=[w['radius_mean_mm'] for w in wins]
    spin=np.array([w['spin_rms_rad_s'] for w in wins])
    spread=float(np.ptp(rates)/max(abs(rates.mean()),1e-9))
    steady=bool(not error and rates.mean()<-.1 and spread<.005 and np.ptp(radii)<.5 and min(w['reverse_fraction'] for w in wins)>.99)
    u=tail;rate=float(np.polyfit(u[:,0]-u[:,0].mean(),np.unwrap(np.arctan2(u[:,3],u[:,2])),1)[0])
    period=-2*np.pi/rate if rate<-.01 else None
    kind='fast_candidate' if steady and period<.99*BASE['period_s'] else ('same_branch' if steady and abs(period/BASE['period_s']-1)<.01 else ('other_rotating' if steady else 'unsettled'))
    if error:kind='outside_field_domain' if 'kStepLUTOutOfDomain' in error else 'solver_error'
    return dict(stationary=steady,outcome=kind,period_s=period,windows=wins,
                rate_relative_spread=spread,radius_mean_mm=float(np.linalg.norm(u[:,2:4],axis=1).mean()/f.KSPACE*1000),
                radius_range_mm=(np.linalg.norm(u[:,2:4],axis=1)/f.KSPACE*1000)[[np.argmin(np.linalg.norm(u[:,2:4],axis=1)),np.argmax(np.linalg.norm(u[:,2:4],axis=1))]].tolist(),
                normal_spin_rms_rad_s=float(np.sqrt(np.mean(u[:,12]**2))*OMEGA),
                spin_stationary=bool(np.ptp(spin)<max(.02,.01*spin.mean())))

def run(task):
    acceleration,duration,phase,free_seconds,dt=task
    state,seed_time=seed(phase)
    inputs=dict(acceleration_m_s2=acceleration,push_duration_s=duration,seed_index=phase,
                seed_time_s=seed_time,seed_state=state.tolist(),free_seconds=free_seconds,dt=dt,
                force_N=MASS*acceleration,ideal_delta_v_m_s=acceleration*duration,
                baseline_key=BASE['key'],binary_sha256=f.BINARY_HASH,lut_sha256=f.LUT_HASH,
                direction='reverse tangent through centre; no direct torque',metric_version=1)
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:20]
    path=OUT/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    p=config();mode=direction=None;global_step=0;clock_step=0;error=None;failed_stage=None
    start=time.perf_counter();rows=[np.r_[0.,state,0.,0.]];free=[];push_actual=0.
    try:
        for stage,seconds,acc in [('push',duration,acceleration),('free',free_seconds,0.)]:
            p.reverse_push_acceleration=acc*f.KSPACE/OMEGA**2
            limit=round(seconds*OMEGA/dt);done=0;chunk=max(1,round(.02*OMEGA/dt))
            if stage=='free':free=[np.r_[0.,state,float(mode or False)]]
            while done<limit:
                n=min(1 if mode is None else chunk,limit-done)
                state,mode,direction=f.mod.evolve_trajectory(state,dt,n,p,mode,direction,global_step)
                done+=n;global_step+=n;clock_step+=n
                rows.append(np.r_[clock_step*dt/OMEGA,state,float(mode),acc])
                if stage=='free':free.append(np.r_[done*dt/OMEGA,state,float(mode)])
            if stage=='push':push_actual=done*dt/OMEGA
    except RuntimeError as exc:
        error=str(exc);failed_stage=stage
    finally:p.reverse_push_acceleration=0
    rec=dict(**inputs,key=key,error=error,failed_stage=failed_stage,push_actual_s=push_actual,
             completed_total_s=clock_step*dt/OMEGA,final_state=state.tolist(),wall_s=time.perf_counter()-start)
    if free:
        free=np.array(free);completed=float(free[-1,0]);rec['completed_free_s']=completed
        rec.update(metrics(free,error,completed))
        if completed<free_seconds-.05:rec['stationary']=False
    else:rec.update(stationary=False,outcome='outside_field_domain' if error and 'kStepLUTOutOfDomain' in error else 'solver_error',completed_free_s=0.)
    # During-force speed is recorded separately and never used as a branch period.
    rows=np.array(rows);r=np.linalg.norm(rows[:,2:4],axis=1)
    rec['maximum_radius_mm']=float(r.max()/f.KSPACE*1000)
    rec['maximum_center_speed_m_s']=float(np.linalg.norm(rows[:,4:6],axis=1).max()*OMEGA/f.KSPACE)
    path.write_text(json.dumps(rec,indent=2));np.savez_compressed(path.with_suffix('.npz'),rows=rows)
    return rec

def batch(tasks,name):
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run,tasks):
            records.append(r);(OUT/(name+'.json')).write_text(json.dumps(records,indent=2))
            print(json.dumps({k:r.get(k) for k in ['key','acceleration_m_s2','push_duration_s','seed_index','period_s','outcome','completed_free_s']}),flush=True)
    return records

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['scan','verify','refine']);args=ap.parse_args()
    if args.action=='scan':
        manifest=dict(baseline=BASE,build=json.loads((f.SOURCE/'build_manifest.json').read_text()),
                      force='Finite reverse tangent force through ball centre; direction follows current centre position.',
                      seeds='Four full states sampled from final orbit; magnetic memory, attitude and spin all preserved.',
                      classification='Only post-withdrawal motion is classified; <0.99 baseline period required for fast candidate.')
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        tasks=[(0.,0.,i,600.,.0005) for i in range(4)]
        tasks += [(a,d,i,600.,.0005) for a in [.001,.003,.01,.03,.1,.3] for d in [.1,.3,1.,3.,10.] for i in range(4)]
        batch(tasks,'scan')
    elif args.action=='refine':
        # Denser kicks around the transition from return to loss of confinement.
        pulses=[(a,d) for a in [.003,.01,.03,.1] for d in [.5,1.5,5.]]
        # About +0.02 m/s would reach the measured angular rate at fixed radius.
        pulses += [(.001,20.),(.003,7.),(.01,2.),(.03,.7),(.1,.2),(.3,.07)]
        pulses += [(a,d) for a in [.0003,.001,.003] for d in [30.,60.,120.]]
        tasks=[(a,d,i,600.,.0005) for a,d in pulses for i in range(4)]
        batch(tasks,'refined')
    else:
        records=[]
        for name in ['scan','refined']:
            p=OUT/(name+'.json')
            if p.exists():records.extend(json.loads(p.read_text()))
        good=[r for r in records if r.get('stationary')]
        candidates=sorted(good,key=lambda r:r.get('period_s',1e6))[:4]
        # Include zero-force control, strongest surviving impulse and longest push.
        candidates += [next(r for r in good if r['acceleration_m_s2']==0),
                       max(good,key=lambda r:r['ideal_delta_v_m_s']),max(good,key=lambda r:r['push_duration_s'])]
        short=[r for r in good if r['acceleration_m_s2']>0 and r['push_duration_s']<=10]
        candidates += [max(short,key=lambda r:r['ideal_delta_v_m_s']),max(short,key=lambda r:r['acceleration_m_s2'])]
        unsettled=[r for r in records if not r.get('error') and not r.get('stationary') and r.get('period_s')]
        candidates += sorted(unsettled,key=lambda r:r['period_s'])[:3]
        lost=[r for r in records if r.get('error') and 'kStepLUTOutOfDomain' in r['error']]
        if lost:candidates.append(min(lost,key=lambda r:r['ideal_delta_v_m_s']))
        unique={(r['acceleration_m_s2'],r['push_duration_s'],r['seed_index']):r for r in candidates}
        batch([(a,d,i,1200.,.00025) for a,d,i in unique],'validated')
