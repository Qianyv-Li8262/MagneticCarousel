"""Continuous reverse COM force after local capture; then truly force-free motion.

Force follows the instantaneous reverse tangent, as clarified in the experiment.
It is evaluated inside every RK/contact evaluation by the isolated CPU build.
No speed/spin/q projection or repeated velocity impulses are used.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from search_prepared_release import load_hysteresis,ROOT

OUT=ROOT/'runs'/'sustained-push'
CASE=ROOT/'results'/'small-tilt-release-fit'/'release_case.json'
OMEGA=100*2*np.pi/60
SCALE=4/.07
MASS=4*np.pi*.005**3/3*7850


def advance(mod,params,state,mode,direction,seconds,dt,stride,force_mn=0.,first_step=False):
    acceleration=force_mn*.001/MASS*SCALE/OMEGA**2
    mod.configure_air_push(acceleration,True,0.,0.)
    state=np.asarray(state,dtype=mod.real_dtype).copy()
    direction=np.asarray(direction,dtype=mod.real_dtype).copy()
    steps=round(seconds*OMEGA/dt);step=0
    rows=[np.r_[0.,state,float(mode),force_mn]]
    while step<steps:
        chunk=min(1 if first_step and step==0 else stride,steps-step)
        state,mode,direction=mod.evolve_trajectory(state,dt,chunk,params,mode,direction,step)
        step+=chunk
        rows.append(np.r_[step*dt/OMEGA,state,float(mode),force_mn])
    return np.asarray(rows),state,bool(mode),np.asarray(direction)


def motion_metrics(table):
    t=table[:,0];s=table[:,1:16];pos=s[:,1:3]*1000/SCALE
    v=s[:,3:5];radius=np.linalg.norm(s[:,1:3],axis=1)
    speed=np.linalg.norm(v,axis=1)*OMEGA*1000/SCALE
    slip=v+(2/7)*np.column_stack([-s[:,10],s[:,9]])
    slip_speed=np.linalg.norm(slip,axis=1)*OMEGA*1000/SCALE
    ratio=slip_speed/np.maximum(speed,1e-9)
    rate=(s[:,1]*v[:,1]-s[:,2]*v[:,0])/radius**2*OMEGA
    angle=np.unwrap(np.arctan2(pos[:,1],pos[:,0]));angle-=angle[0]
    tail=t>=t[-1]-min(10.,t[-1]/3)
    # Window means avoid demanding a constant speed on a tilted disk.
    half=t>=t[-1]-min(20.,t[-1]*2/3)
    reverse=float(np.mean(rate[tail]<0))
    rolling=float(np.mean(table[tail,16]==0))
    net_cycles=float(angle[-1]/(2*np.pi))
    result=dict(net_ball_revolutions=net_cycles,
       tail_mean_angular_speed_rad_s=float(np.mean(rate[tail])),
       tail_reverse_time_fraction=reverse,tail_pure_rolling_fraction=rolling,
       tail_slip_ratio_median=float(np.median(ratio[tail])),
       tail_slip_ratio_p95=float(np.quantile(ratio[tail],.95)),
       tail_mean_center_speed_mm_s=float(np.mean(speed[tail])),
       tail_position_box_mm=float(np.linalg.norm(np.ptp(pos[tail],axis=0))),
       tail_radius_mean_mm=float(np.mean(radius[tail]*1000/SCALE)),
       radius_range_mm=[float(np.min(radius)*1000/SCALE),float(np.max(radius)*1000/SCALE)],
       sustained_reverse=bool(net_cycles<-3 and np.mean(rate[tail])<-.2 and reverse>.95),
       near_pure_reverse=bool(net_cycles<-3 and np.mean(rate[tail])<-.2 and reverse>.95 and
                             np.quantile(ratio[tail],.95)<.05),
       strict_pure_reverse=bool(net_cycles<-3 and np.mean(rate[tail])<-.2 and reverse>.95 and rolling>.95))
    # Compare complete late revolutions, rather than clock windows of different phase.
    cycles=[]
    if net_cycles<-3:
        for n in range(max(1,int(-net_cycles)-5),int(-net_cycles)+1):
            ids=np.flatnonzero(angle<=-n*2*np.pi)
            prev=np.flatnonzero(angle<=-(n-1)*2*np.pi)
            if len(ids) and len(prev):
                a,b=prev[0],ids[0]
                cycles.append(dict(revolution=n,period_s=float(t[b]-t[a]),
                                   radius_mean_mm=float(np.mean(radius[a:b+1])*1000/SCALE)))
    result['late_revolutions']=cycles
    if len(cycles)>=3:
        periods=np.array([c['period_s'] for c in cycles[-3:]])
        result['last3_revolution_period_relative_span']=float(np.ptp(periods)/np.mean(periods))
    return result


def save_trace(path,table):
    s=table[:,1:16];pos=s[:,1:3]*1000/SCALE
    angle=np.unwrap(np.arctan2(pos[:,1],pos[:,0]));angle-=angle[0]
    np.savez_compressed(path,state_table=table,time_s=table[:,0],position_mm=pos,angle_rad=angle,
                        external_force_mN=table[:,17])


def baseline(mod,params,case,dt=.0005,seconds=60.):
    mod.configure_air_push(0.,True,0.,0.)
    return advance(mod,params,case['initial_state'],case['initial_sliding'],case['initial_sliding_direction'],
                   seconds,dt,20,first_step=True)


def adaptive_push(mod,params,captured,force_mn,dt=.001,max_seconds=30.):
    """Withdraw only after three completed reverse orbits repeat their period/radius."""
    _,state,mode,direction=captured
    state=state.copy();direction=direction.copy()
    mod.configure_air_push(force_mn*.001/MASS*SCALE/OMEGA**2,True,0.,0.)
    stride=80;steps=round(max_seconds*OMEGA/dt);step=0
    table=[np.r_[0.,state,float(mode),force_mn]]
    last_phi=float(np.arctan2(state[2],state[1]));angle=0.
    last_cycle_step=0;last_cycle_time=0.;cycles=[];stable=False
    while step<steps:
        chunk=min(stride,steps-step)
        state,mode,direction=mod.evolve_trajectory(state,dt,chunk,params,mode,direction,step)
        step+=chunk;t=step*dt/OMEGA
        table.append(np.r_[t,state,float(mode),force_mn])
        previous_angle=angle;previous_t=table[-2][0]
        phi=float(np.arctan2(state[2],state[1]));angle+=float(np.arctan2(np.sin(phi-last_phi),np.cos(phi-last_phi)));last_phi=phi
        if angle<=-(len(cycles)+1)*2*np.pi:
            target=-(len(cycles)+1)*2*np.pi
            crossing_t=previous_t+(target-previous_angle)/(angle-previous_angle)*(t-previous_t)
            segment=np.asarray(table[last_cycle_step:])
            radius=np.linalg.norm(segment[:,2:4],axis=1)*1000/SCALE
            rate=(segment[:,2]*segment[:,5]-segment[:,3]*segment[:,4])/np.sum(segment[:,2:4]**2,axis=1)*OMEGA
            cycles.append(dict(revolution=len(cycles)+1,period_s=crossing_t-last_cycle_time,
                               radius_mean_mm=float(np.mean(radius)),reverse_time_fraction=float(np.mean(rate<0))))
            last_cycle_step=len(table)-1;last_cycle_time=crossing_t
            if len(cycles)>=4: # discard the first acceleration lap
                recent=cycles[-3:]
                periods=np.array([c['period_s'] for c in recent]);radii=np.array([c['radius_mean_mm'] for c in recent])
                stable=bool(np.ptp(periods)/np.mean(periods)<.01 and np.ptp(radii)<.5 and
                            min(c['reverse_time_fraction'] for c in recent)>.99)
                if stable:break
    return np.asarray(table),state,bool(mode),direction,dict(stable_under_push=stable,
                actual_push_seconds=float(t),completed_pushed_revolutions=len(cycles),pushed_revolutions=cycles)


def trial(mod,params,captured,force_mn,push_seconds,free_seconds=50.,dt=.001,save=False,tag=''):
    _,state,mode,direction=captured
    result=dict(force_mN=float(force_mn),push_seconds=float(push_seconds),
                force_free_seconds=float(free_seconds),dt=dt,status='ok')
    try:
        push,after,mode,direction=advance(mod,params,state,mode,direction,push_seconds,dt,40,force_mn)
        result['under_push']=motion_metrics(push)
        result['state_at_force_removal']=after.tolist()
        result['mode_at_force_removal']=bool(mode)
        result['direction_at_force_removal']=direction.tolist()
        # This call explicitly sets the external force to ZERO for every free step.
        free,final,mode,direction=advance(mod,params,after,mode,direction,free_seconds,dt,40,0.)
        result['after_force_removal']=motion_metrics(free)
        allrows=np.vstack([push,free[1:]+np.r_[push[-1,0],np.zeros(17)]])
        if save:
            path=OUT/(tag+'.npz');save_trace(path,allrows);result['trace']=str(path)
            result['pushed_initial_state']=state.tolist()
            result['pushed_initial_sliding']=bool(captured[2])
            result['pushed_initial_sliding_direction']=captured[3].tolist()
        return result,allrows
    except RuntimeError as exc:
        result.update(status='failed',error=str(exc));return result,None
    finally:mod.configure_air_push(0.,True,0.,0.)


def score(r):
    if r['status']!='ok':return (5,1e6)
    free=r['after_force_removal']
    return (0 if free['strict_pure_reverse'] else 1 if free['near_pure_reverse'] else
            2 if free['sustained_reverse'] else 3,
            free['tail_slip_ratio_p95'] if free['sustained_reverse'] else -abs(free['net_ball_revolutions']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['sweep','verify','adaptive','replay'],default='sweep')
    p.add_argument('--force-mn',type=float,default=1.)
    p.add_argument('--push-seconds',type=float,default=5.)
    p.add_argument('--free-seconds',type=float,default=180.)
    p.add_argument('--dt',type=float,default=.0005)
    p.add_argument('--capture-after',type=float,default=45.)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    mod=load_hysteresis();case=json.loads(CASE.read_text(encoding='utf-8'))
    params=mod.SolverParams(**case['solver_parameters'])
    captured=baseline(mod,params,case)
    save_trace(OUT/'unforced_baseline.npz',captured[0])
    reference=np.load(ROOT/'results'/'small-tilt-release-fit'/'trajectory.npz')
    # Prior table has one fewer column (the external-force column is new).
    difference=float(np.max(np.abs(captured[0][:,1:17]-reference['state_table'][:,1:17])))
    time_difference=float(np.max(np.abs(captured[0][:,0]-reference['time_s'])))
    print(json.dumps(dict(zero_push_max_state_difference=difference,time_difference_s=time_difference,mass_kg=MASS)),flush=True)
    (OUT/'baseline_validation.json').write_text(json.dumps(dict(max_state_difference=difference,time_difference_s=time_difference),indent=2),encoding='utf-8')
    rows=[]
    if args.stage=='sweep':
        for force in [.05,.1,.2,.4,.6,.8,1.,1.5,2.,3.,5.]:
            for duration in [.5,1.,2.,4.,8.,15.,30.]:
                r,_=trial(mod,params,captured,force,duration)
                rows.append(r)
            print(json.dumps(dict(force_mN=force,results=[dict(seconds=r['push_seconds'],status=r['status'],
                metrics=r.get('after_force_removal')) for r in rows[-7:]])),flush=True)
            (OUT/'sweep.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    elif args.stage=='adaptive':
        for force in [.6,.7,.8,1.,1.5]:
            for dt in [.001,.0005]:
                start=baseline(mod,params,case,dt=.0005,seconds=args.capture_after)
                record=dict(force_mN=force,dt=dt,capture_after_s=args.capture_after,status='ok')
                try:
                    push,after,mode,direction,stop=adaptive_push(mod,params,start,force,dt,max_seconds=30.)
                    record.update(stop);record['under_push']=motion_metrics(push)
                    record['state_at_force_removal']=after.tolist();record['mode_at_force_removal']=bool(mode)
                    record['direction_at_force_removal']=direction.tolist()
                    free,_,_,_=advance(mod,params,after,mode,direction,180.,dt,40,0.)
                    record['after_force_removal']=motion_metrics(free)
                    allrows=np.vstack([push,free[1:]+np.r_[push[-1,0],np.zeros(17)]])
                    tag=f'adaptive_{force:g}_{dt}'
                    path=OUT/(tag+'.npz');save_trace(path,allrows);record['trace']=str(path)
                    record['pushed_initial_state']=start[1].tolist();record['pushed_initial_sliding']=start[2]
                    record['pushed_initial_sliding_direction']=start[3].tolist()
                except RuntimeError as exc:record.update(status='failed',error=str(exc))
                finally:mod.configure_air_push(0.,True,0.,0.)
                rows.append(record)
                print(json.dumps(record),flush=True)
                (OUT/'adaptive.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    elif args.stage=='verify':
        pool=json.loads((OUT/'sweep.json').read_text(encoding='utf-8'))
        running=sorted([r for r in pool if r['status']=='ok' and r['after_force_removal']['sustained_reverse']],key=score)
        selected=[]
        for r in running:
            if all(r['force_mN']!=s['force_mN'] for s in selected):selected.append(r)
            if len(selected)==3:break
        for i,r in enumerate(selected):
            for dt in [.001,.0005]:
                v,_=trial(mod,params,captured,r['force_mN'],r['push_seconds'],180.,dt,save=True,
                          tag=f'verify_{i}_{dt}')
                v['candidate']=i;rows.append(v);print(json.dumps(v),flush=True)
    else:
        r,_=trial(mod,params,captured,args.force_mn,args.push_seconds,args.free_seconds,args.dt,
                  save=True,tag='replay')
        rows.append(r);print(json.dumps(r),flush=True)
    (OUT/(args.stage+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage=args.stage,completed=len(rows),best=sorted(rows,key=score)[:2])),flush=True)


if __name__=='__main__':main()
