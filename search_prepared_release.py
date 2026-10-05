"""Parameter-only search for immediate reverse motion and local arrest.

No geometry, sliding friction, drag or mechanical coefficients are varied.
Every release has zero speed/spin and 1--3 disk revolutions of held magnetization.
Raw Cartesian motion is used to judge arrest, not only a smoothed angle.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch'/'legacy-release-source'
OUT=ROOT/'runs'/'prepared-release-fit'
sys.path.insert(0,str(SOURCE))
from animate_one_trajectory import read_build_config,load_hysteresis,make_initial_state,scaled_model_parameters


def initial(module,params,phase,turns,dt,radius_mm=70.):
    state=make_initial_state(module,alpha=params.alpha,phi0=0)
    state[1]=radius_mm/17.5
    steps=round(2*np.pi*turns/dt)
    state[0]=phase-steps*dt
    return module.magnetize_held_ball(state,dt,steps,params)


def simulate(module,params,settings,phase,turns=2,seconds=12,dt=.001,stride=80,save=False,tag='',
             kick_mm_s=0.,roll_fraction=0.,state_override=None,radius_mm=70.):
    mur,aja,cja,khys,tilt,axis=settings
    if hasattr(module,'configure_release_model'):
        module.configure_release_model(*settings)
    try:
        state=(initial(module,params,phase,turns,dt,radius_mm) if state_override is None
               else np.asarray(state_override,dtype=module.real_dtype).copy())
        prepared_q=state[12:15].copy()
        diskomega=100*2*np.pi/60
        if state_override is None:
            state[4]=-kick_mm_s/1000*(4/.07)/diskomega
            state[9]=-state[4]/params.alpha*roll_fraction
        slip=state[3:5]+params.alpha*np.array([-state[10],state[9]])
        if np.linalg.norm(slip)>1e-10:
            mode,direction=True,slip/np.linalg.norm(slip)
        else:
            mode,direction=module.release_contact_mode(state,params)
        release_state=state.copy()
        release_mode=bool(mode)
        release_direction=np.asarray(direction).copy()
        data=[np.r_[0.,state,float(mode)]]
        state,mode,direction=module.evolve_trajectory(state,dt,1,params,mode,direction,0)
        data.append(np.r_[dt,state,float(mode)])
        steps=round(seconds*diskomega/dt)
        start=1
        while start<steps:
            chunk=min(stride,steps-start)
            state,mode,direction=module.evolve_trajectory(state,dt,chunk,params,mode,direction,start)
            start+=chunk
            data.append(np.r_[start*dt,state,float(mode)])
    except (RuntimeError,ValueError) as exc:
        return dict(settings=list(settings),phase=phase,turns=turns,status='failed',error=str(exc)),None
    data=np.asarray(data)
    times=data[:,0]/diskomega
    pos=data[:,2:4]*1000/(4/.07)
    radius=np.linalg.norm(pos,axis=1)
    angle=np.unwrap(np.arctan2(pos[:,1],pos[:,0]))
    angle-=angle[0]
    angular_speed=np.sum(data[:,4:6]*np.column_stack([-data[:,3],data[:,2]]),axis=1)/np.sum(data[:,2:4]**2,axis=1)*diskomega
    first_accel=(angular_speed[1]-angular_speed[0])/times[1]
    starts_reverse=angular_speed[0]<0 or first_accel<0
    sampledt=stride*dt/diskomega
    smooth=uniform_filter1d(angle,max(1,round(.15/sampledt)),mode='nearest')
    tail=times>seconds-min(3,seconds/3)
    center=np.mean(pos[tail],axis=0)
    tail_box=np.linalg.norm(np.ptp(pos[tail],axis=0))
    tail_maxdist=float(np.max(np.linalg.norm(pos[tail]-center,axis=1)))
    end=float(np.mean(angle[tail]))
    running_min=np.minimum.accumulate(smooth)
    moved=running_min<-.2
    rebound=float(np.max((smooth-running_min)[moved])) if np.any(moved) else 10.
    firstdepart=np.flatnonzero(angle<-.05)
    earlyend=firstdepart[0] if len(firstdepart) else len(angle)
    forward=float(np.max(angle[:max(2,earlyend)]))
    target_miss=max(0,end+1,-2-end)
    distance=np.linalg.norm(pos-center,axis=1)
    bad=np.flatnonzero(distance>1)
    settle=float(times[bad[-1]+1]) if len(bad) and bad[-1]+1<len(times) else None
    if not len(bad): settle=0.
    score=(forward/.01+rebound/.03+tail_box/1+target_miss/.1+
           (10 if first_accel>=0 else 0))
    record=dict(settings=list(settings),phase=float(phase),turns=turns,dt=dt,stride=stride,
        seconds=float(times[-1]),status='ok',score=float(score),
        initial_angular_acceleration_rad_s2=float(first_accel),
        initial_forward_rad=forward,final_displacement_rad=end,
        minimum_displacement_rad=float(np.min(angle)),max_slow_rebound_rad=rebound,
        tail_raw_angle_span_rad=float(np.ptp(angle[tail])),tail_position_box_mm=float(tail_box),
        tail_max_distance_from_mean_mm=tail_maxdist,settle_within_1mm_s=settle,
        radial_range_mm=[float(radius.min()),float(radius.max())],
        tail_mean_speed_mm_s=float(np.mean(np.linalg.norm(data[tail,4:6],axis=1)*diskomega*1000/(4/.07))),
        prepared_q=prepared_q.tolist(),
        kick_mm_s=float(kick_mm_s),roll_fraction=float(roll_fraction),
        prepared_at_release=state_override is None,
        release_radius_mm=float(np.linalg.norm(release_state[1:3])*17.5),
        initial_state=release_state.tolist(),initial_sliding=release_mode,
        initial_sliding_direction=release_direction.tolist(),
        initial_angular_speed_rad_s=float(angular_speed[0]),
        passes=bool(starts_reverse and forward<=.01 and -2<=end<=-1 and
                    rebound<=.03 and tail_box<=1 and settle is not None and settle<seconds-3))
    if save:
        OUT.mkdir(parents=True,exist_ok=True)
        name=tag or f'trace_{int(time.time()*1e6)}'
        path=OUT/(name+'.npz')
        np.savez_compressed(path,state_table=data,time_s=times,position_mm=pos,
                            angle_rad=angle,smoothed_angle_rad=smooth,angular_speed_rad_s=angular_speed)
        record['trace']=str(path)
    return record,(data,times,pos,angle,smooth)


def magnets():
    old=[2292.6667,.0032,.38,.0010909091]
    out=[('legacy',old),('current',[4138.55,.00070921,.0198,.00104963636364])]
    for x in [.03,.1,.3,1.7,3.,10.]:
        out.append((f'Kx{x}',old[:3]+[old[3]*x]))
    for x in [.08,.18,.6,.8,.95]:
        out.append((f'C{x}',[old[0],old[1],x,old[3]]))
    for x in [.0001,.0007,.0015,.0064,.02]:
        out.append((f'A{x}',[old[0],x,old[2],old[3]]))
    for x in [2.,5.,20.,100.,300.,8000.]:
        out.append((f'Mu{x}',[x]+old[1:]))
    for c,k,a in [(.1,.0003,.0032),(.2,.003,.0064),(.8,.0001,.0032),(.1,.003,.0007),(.6,.003,.02)]:
        out.append((f'mix{c}_{k}_{a}',[old[0],a,c,k]))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['coarse','refine','random','optimize','verify'],default='coarse')
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--wide-axis',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=read_build_config(kahan=False)
    module=load_hysteresis()
    params=module.SolverParams(*scaled_model_parameters(8,sr=5/6,somega=1))
    phases=np.arange(8)*np.pi/32
    records=[]
    if args.stage=='coarse':
        for name,mag in magnets():
            best=None
            for tilt in [.0035,.007,.02]:
                for phase in phases:
                    row,_=simulate(module,params,mag+[tilt,.1],phase)
                    row['magnetic_group']=name
                    records.append(row)
                    if row['status']=='ok' and (best is None or row['score']<best['score']):best=row
            print(json.dumps(dict(group=name,best=best)),flush=True)
            (OUT/'coarse.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    elif args.stage=='optimize':
        from scipy.optimize import differential_evolution
        prior=[]
        for name in ['coarse','refine','optimize']:
            if not (OUT/(name+'.json')).exists():continue
            prior+=json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
        seed=min([r for r in prior if r['status']=='ok'],key=lambda r:r['score'])
        s=seed['settings']
        x0=[np.log10(s[0]),np.log10(s[1]),s[2],np.log10(s[3]),np.log10(s[4]),s[5],seed['phase']]
        best=[None]
        def objective(x):
            settings=[10**x[0],10**x[1],x[2],10**x[3],10**x[4],x[5]]
            row,_=simulate(module,params,settings,x[6],seconds=20)
            records.append(row)
            value=row['score'] if row['status']=='ok' else 10000.
            if row['status']=='ok' and (best[0] is None or value<best[0]['score']):best[0]=row
            if len(records)%40==0:
                print(json.dumps(dict(evaluations=len(records),best=best[0])),flush=True)
                (OUT/('wide_optimize.json' if args.wide_axis else 'optimize.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
            return value
        differential_evolution(objective,[(np.log10(1.1),4),(-5,np.log10(.03)),(.025,.98),
             (-5,np.log10(.03)),(-3,np.log10(.08)),(-np.pi,np.pi) if args.wide_axis else (-.4,.6),(0,np.pi/4)],
             x0=x0,seed=18262 if args.wide_axis else 8262,popsize=8,maxiter=16 if args.wide_axis else 24,polish=False,tol=.001,
             mutation=(.5,1),recombination=.8,workers=1)
    elif args.stage in ['refine','random']:
        base=json.loads((OUT/'coarse.json').read_text(encoding='utf-8'))
        bests=sorted([r for r in base if r['status']=='ok'],key=lambda r:r['score'])
        unique=[]
        for r in bests:
            if r['settings'][:4] not in [u[:4] for u in unique]:unique.append(r['settings'])
            if len(unique)==6:break
        if args.stage=='refine':
            candidates=[u[:4]+[tilt,axis] for u in unique for tilt in [.002,.005,.01,.015,.03]
                        for axis in [-.2,.1,.4]]
        else:
            rng=np.random.default_rng(8262)
            candidates=[]
            for i in range(160):
                u=unique[i%len(unique)]
                mag=[max(1.05,u[0]*np.exp(rng.normal(0,.7))),
                     u[1]*np.exp(rng.normal(0,.6)),
                     float(np.clip(u[2]+rng.normal(0,.12),.025,.98)),
                     u[3]*np.exp(rng.normal(0,.6))]
                candidates.append(mag+[float(np.exp(rng.uniform(np.log(.002),np.log(.03)))),
                                       float(rng.uniform(-.35,.5))])
        for i,settings in enumerate(candidates):
            best=None
            for phase in phases:
                row,_=simulate(module,params,settings,phase,seconds=15)
                records.append(row)
                if row['status']=='ok' and (best is None or row['score']<best['score']):best=row
            if i%10==0:
                valid=sorted([r for r in records if r['status']=='ok'],key=lambda r:r['score'])
                print(json.dumps(dict(candidate=i,total=len(candidates),best_so_far=valid[:1])),flush=True)
                (OUT/(args.stage+'.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
    else:
        if args.fixed:
            selected=json.loads((OUT/'fixed_candidate.json').read_text(encoding='utf-8'))
            bests=[selected]
        else:
            allrows=[]
            for name in ['coarse','refine','random','optimize','wide_optimize']:
                path=OUT/(name+'.json')
                if path.exists():allrows+=json.loads(path.read_text(encoding='utf-8'))
            ordered=sorted([r for r in allrows if r['status']=='ok'],key=lambda r:r['score'])
            bests=[]
            for r in ordered:
                if all(np.linalg.norm(np.log(np.array(r['settings'][:2])/np.array(b['settings'][:2])))>.05
                       or abs(r['settings'][2]-b['settings'][2])>.03
                       or abs(r['settings'][4]-b['settings'][4])>.001 for b in bests):bests.append(r)
                if len(bests)==4:break
        for i,best in enumerate(bests):
            for turns in [1,2,3]:
                for dt in [.001,.0005]:
                    row,_=simulate(module,params,best['settings'],best['phase'],turns,
                                   seconds=60,dt=dt,stride=20,save=True,
                                   tag=f"{'fixed' if args.fixed else 'verify'}_{i}_turns{turns}_dt{dt}")
                    records.append(row)
                    print(json.dumps(row),flush=True)
    name=('fixed_verify' if args.fixed else 'wide_optimize' if args.wide_axis else args.stage)+'.json'
    (OUT/name).write_text(json.dumps(records,indent=2),encoding='utf-8')
    valid=sorted([r for r in records if r['status']=='ok'],key=lambda r:r['score'])
    print(json.dumps(dict(stage=args.stage,completed=len(valid),failed=len(records)-len(valid),
                         passes=sum(r.get('passes',False) for r in valid),best=valid[:3])),flush=True)


if __name__=='__main__':main()
