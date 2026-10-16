"""Current core: settle in capture, follow with a reverse push, then withdraw it."""
from pathlib import Path
import argparse
import json
import numpy as np
import animate_one_trajectory as default
from replay_prepared_release import plot_result, animate_result

ROOT=Path(__file__).resolve().parent
OMEGA=100*2*np.pi/60
SCALE=4/.07

def advance(module,params,state,mode,direction,seconds,dt,force_mn,mass):
    params.reverse_push_acceleration=force_mn*.001/mass*SCALE/OMEGA**2
    count=round(seconds*OMEGA/dt);step=0
    rows=[np.r_[0.,state,float(mode) if mode is not None else 0.,force_mn]]
    while step<count:
        n=min(1 if step==0 and mode is None else 80,count-step)
        state,mode,direction=module.evolve_trajectory(state,dt,n,params,mode,direction,step)
        step+=n;rows.append(np.r_[step*dt/OMEGA,state,float(mode),force_mn])
    return np.asarray(rows),state,mode,direction

def lap_metrics(table):
    states=table[:,1:16];r=np.linalg.norm(states[:,1:3],axis=1)
    angle=np.unwrap(np.arctan2(states[:,2],states[:,1]));angle-=angle[0]
    rate=(states[:,1]*states[:,4]-states[:,2]*states[:,3])/r**2*OMEGA
    cycles=[];start=0;last_t=0.
    for i in range(1,len(table)):
        if angle[i]<=-(len(cycles)+1)*2*np.pi:
            target=-(len(cycles)+1)*2*np.pi
            crossing=table[i-1,0]+(target-angle[i-1])/(angle[i]-angle[i-1])*(table[i,0]-table[i-1,0])
            cycles.append(dict(period_s=float(crossing-last_t),radius_mean_mm=float(np.mean(r[start:i+1])*1000/SCALE),
                               reverse_fraction=float(np.mean(rate[start:i+1]<0))))
            start=i;last_t=crossing
    return angle,rate,cycles

def push_until_stable(module,params,captured,force_mn,mass,dt,max_seconds,min_seconds):
    _,state,mode,direction=captured
    parts=[];elapsed=0.;stable=False
    while elapsed<max_seconds:
        part,state,mode,direction=advance(module,params,state,mode,direction,min(.5,max_seconds-elapsed),dt,force_mn,mass)
        part[:,0]+=elapsed;elapsed=part[-1,0]
        parts.append(part if not parts else part[1:])
        table=np.vstack(parts)
        _,_,cycles=lap_metrics(table)
        if len(cycles)>=4:
            recent=cycles[-3:];periods=np.array([c['period_s'] for c in recent]);radii=np.array([c['radius_mean_mm'] for c in recent])
            stable=bool(np.ptp(periods)/np.mean(periods)<.01 and np.ptp(radii)<.5 and min(c['reverse_fraction'] for c in recent)>.99)
            if stable and elapsed>=min_seconds:break
    return table,state,mode,direction,dict(stable_under_push=stable,push_seconds=float(elapsed),pushed_laps=len(cycles),last3_pushed_laps=cycles[-3:])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force-mn',type=float,default=2.)
    ap.add_argument('--capture-seconds',type=float,default=30.)
    ap.add_argument('--max-push-seconds',type=float,default=90.)
    ap.add_argument('--min-push-seconds',type=float,default=30.,help='Continue pushing at least this long, even if orbit periods have settled.')
    ap.add_argument('--free-seconds',type=float,default=180.)
    ap.add_argument('--dt',type=float,default=default.DT_AT_EIGHT_MAGNETS)
    ap.add_argument('--animate',action='store_true')
    ap.add_argument('--output',type=Path,default=ROOT/'runs/gaussian-after-settling')
    args=ap.parse_args()
    if not all(np.isfinite(v) and v>0 for v in [args.force_mn,args.capture_seconds,args.max_push_seconds,args.free_seconds,args.dt]) or args.capture_seconds<20:
        ap.error('Positive finite values required; capture observation must be at least 20 s.')
    if not np.isfinite(args.min_push_seconds) or not 0<=args.min_push_seconds<=args.max_push_seconds:
        ap.error('--min-push-seconds must lie between zero and --max-push-seconds.')
    args.output.mkdir(parents=True,exist_ok=True)
    config=default.read_build_config(kahan=False);module=default.load_hysteresis()
    params=module.SolverParams(sr=1.0, somega=1.0)
    # Derive mass from the unchanged reference radius, never from the historical
    # 10 mm experimental scripts.
    radius=params.alpha/SCALE;mass=4*np.pi/3*radius**3*7850
    state=default.prepare_initial_state(module,params,args.dt)
    captured=advance(module,params,state,None,None,args.capture_seconds,args.dt,0.,mass)
    table=captured[0];t=table[:,0];xy=table[:,2:4]*1000/SCALE
    first=(t>=t[-1]-10)&(t<t[-1]-5);second=t>=t[-1]-5
    drift=float(np.linalg.norm(xy[first].mean(0)-xy[second].mean(0)))
    envelope=float(np.linalg.norm(np.ptp(xy[t>=t[-1]-10],axis=0)))
    if drift>.02 or envelope>1.1:
        raise RuntimeError(f'Capture not sufficiently settled: mean drift={drift} mm; envelope={envelope} mm')
    print(json.dumps(dict(capture_confirmed=True,wait_seconds=float(t[-1]),last_two_5s_mean_position_difference_mm=drift,envelope_mm=envelope)),flush=True)
    push,state,mode,direction,push_summary=push_until_stable(module,params,captured,args.force_mn,mass,args.dt,args.max_push_seconds,args.min_push_seconds)
    try:
        free,_,_,_=advance(module,params,state,mode,direction,args.free_seconds,args.dt,0.,mass)
    finally:
        params.reverse_push_acceleration=0.
    start=table[-1,0];off=start+push[-1,0]
    allrows=np.vstack([table,push+np.r_[start,np.zeros(17)],free+np.r_[off,np.zeros(17)]])
    angle,rate,cycles=lap_metrics(free);states=free[:,1:16];tail=free[:,0]>=free[-1,0]-10
    slip=states[:,3:5]+params.alpha*np.column_stack([-states[:,10],states[:,9]])
    ratio=np.linalg.norm(slip,axis=1)/np.maximum(np.linalg.norm(states[:,3:5],axis=1),1e-12)
    periods=np.array([c['period_s'] for c in cycles[-3:]])
    after=dict(reverse_laps=float(-angle[-1]/(2*np.pi)),last3_laps=cycles[-3:],
               last3_period_relative_span=float(np.ptp(periods)/np.mean(periods)) if len(periods)>=3 else None,
               tail_mean_angular_speed_rad_s=float(np.mean(rate[tail])),tail_reverse_fraction=float(np.mean(rate[tail]<0)),
               tail_slip_ratio_median=float(np.median(ratio[tail])),tail_slip_ratio_p95=float(np.quantile(ratio[tail],.95)),
               tail_pure_rolling_fraction=float(np.mean(free[tail,16]==0)),external_force_mN=float(np.max(np.abs(free[:,17]))))
    switch=max(float(np.max(np.abs(table[-1,1:17]-push[0,1:17]))),float(np.max(np.abs(push[-1,1:17]-free[0,1:17]))))
    summary=dict(ball_diameter_mm=2*radius*1000,ball_mass_g=mass*1000,force_mN=args.force_mn,
                 capture_wait_s=float(start),capture_last_two_5s_mean_difference_mm=drift,capture_envelope_mm=envelope,
                 **push_summary,force_off_s=float(off),after_force_removal=after,force_switch_state_difference=switch,
                 initial_state=table[0,1:16].tolist(),state_at_force_off=state.tolist(),compiled_config=config)
    np.savez_compressed(args.output/'trajectory.npz',state_table=allrows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    plot_result(allrows[:,0],allrows[:,1:16],args.output/'trajectory.png',title='12 mm ball: settled capture > following push > force off',
                capture_candidate=False,force_interval=(start,off))
    print(json.dumps({k:v for k,v in summary.items() if k!='compiled_config'},indent=2),flush=True)
    if args.animate:
        # Show five seconds of settled capture before the force begins. The
        # complete release/preparation trace remains in the saved result.
        displayed=allrows[allrows[:,0]>=start-5]
        print('Starting animation: settled capture, following reverse push, then force off.',flush=True)
        animate_result(displayed[:,0],displayed[:,1:16],max(1,round(.03*OMEGA/(80*args.dt))),
                       title='Settled capture > reverse air push > force off',ball_radius_mm=radius*1000,force_interval=(start,off))

if __name__=='__main__':main()
