"""Measured laser tilt: blow into a repeatable orbit, remove force, compare flat.

Uses the current fixed Q235 constants in an isolated FP64 build. The requested
no-added-rolling-resistance limit retains contact friction, drag and hysteresis.
"""
from pathlib import Path
import argparse,importlib.util,json,hashlib,time
import numpy as np
import animate_one_trajectory as defaults

ROOT=Path(__file__).resolve().parent
SCALE=4/.07
LASER_ANGLE=.5*np.arctan(5/900)

def laps(table,omega):
    s=table[:,1:16];angle=np.unwrap(np.arctan2(s[:,2],s[:,1]));angle-=angle[0]
    levels=-2*np.pi*np.arange(0,int(max(0,-angle.min()/(2*np.pi)))+1)
    if np.any(np.diff(angle)>1e-4):
        crossings=[]
        for level in levels:
            ids=np.flatnonzero(angle<=level)
            if len(ids):
                i=ids[0]
                crossing=table[0,0] if i==0 else np.interp(level,angle[i-1:i+1][::-1],table[i-1:i+1,0][::-1])
                crossings.append(crossing)
    else:crossings=np.interp(levels,angle[::-1],table[::-1,0])
    crossings=np.array(crossings);return angle,crossings,np.diff(crossings)

def summarize(table,omega,mod,params):
    s=table[:,1:16];t=table[:,0];angle,crossings,periods=laps(table,omega)
    if len(periods)<4:return {'complete_reverse_laps':len(periods),'steady_free_orbit':False},None
    start=crossings[max(0,len(crossings)-9)];end=crossings[-1]
    used=(t>=start)&(t<=end);state=s[used];tt=t[used]
    radius=np.linalg.norm(state[:,1:3],axis=1);speed=np.linalg.norm(state[:,3:5],axis=1)*omega*1000/SCALE
    tangent=abs((state[:,1]*state[:,4]-state[:,2]*state[:,3])/radius)*omega*1000/SCALE
    phi=np.arctan2(state[:,2],state[:,1])%(2*np.pi)
    matrix=np.c_[np.ones(len(phi)),np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi)]
    coef=np.linalg.lstsq(matrix,tangent,rcond=None)[0]
    grid=np.linspace(0,2*np.pi,1000);pred=np.c_[np.ones(len(grid)),np.cos(grid),np.sin(grid),np.cos(2*grid),np.sin(2*grid)]@coef
    recent=periods[-5:];period_span=float(np.ptp(recent)/np.mean(recent))
    last10=t>=t[-1]-10;last_radius=np.linalg.norm(s[last10,1:3],axis=1)
    last_rate=(s[last10,1]*s[last10,4]-s[last10,2]*s[last10,3])/last_radius**2*omega
    slip=state[:,3:5]+params.alpha*np.c_[-state[:,10],state[:,9]]
    mag=np.array([mod.laser_magnetic_force(row,params) for row in state[::max(1,len(state)//800)]]) if hasattr(mod,'laser_magnetic_force') else None
    metric={'complete_reverse_laps':len(periods),'analysis_time_range_s':[float(start),float(end)],'last5_periods_s':recent.tolist(),'last5_period_relative_span':period_span,'steady_free_orbit':bool(period_span<.01 and np.mean(last_rate<0)>.99),'mean_orbit_period_s':float(np.mean(periods[-8:])),'radius_mean_mm':float(radius.mean()*1000/SCALE),'mean_speed_mm_s':float(speed.mean()),'mean_tangential_speed_mm_s':float(tangent.mean()),'first_harmonic_relative_half_amplitude_percent':float(np.hypot(*coef[1:3])/coef[0]*100),'second_harmonic_relative_half_amplitude_percent':float(np.hypot(*coef[3:5])/coef[0]*100),'phase_model_half_range_over_midpoint_percent':float((pred.max()-pred.min())/(pred.max()+pred.min())*100),'raw_extrema_half_range_over_midpoint_percent':float((tangent.max()-tangent.min())/(tangent.max()+tangent.min())*100),'min_tangential_speed_mm_s':float(tangent.min()),'max_tangential_speed_mm_s':float(tangent.max()),'tail_mean_angular_speed_rad_s':float(last_rate.mean()),'tail_reverse_fraction':float(np.mean(last_rate<0)),'rolling_fraction':float(np.mean(table[used,16]==0)),'slip_ratio_p95':float(np.percentile(np.linalg.norm(slip,axis=1)/np.maximum(np.linalg.norm(state[:,3:5],axis=1),1e-12),95)),'mean_normal_load_over_weight':float(np.mean(1-mag[:,2]/params.gamma_g)) if mag is not None else None,'external_force_max_mN':float(np.max(abs(table[used,17]))),'gravity_only_rolling_estimate_percent':None}
    all_radius=np.linalg.norm(s[:,1:3],axis=1)*1000/SCALE
    lap_radii=[float(all_radius[(t>=a)&(t<=b)].mean()) for a,b in zip(crossings[-6:-1],crossings[-5:])]
    metric['last5_lap_mean_radii_mm']=lap_radii
    metric['last5_lap_mean_radius_span_mm']=float(np.ptp(lap_radii))
    metric['steady_free_orbit']=bool(metric['steady_free_orbit'] and np.ptp(lap_radii)<.5)
    return metric,(tt,phi,tangent,grid,pred)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rpm',type=float,default=100);ap.add_argument('--diameter-mm',type=float,default=12);ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--free-seconds',type=float,default=180);ap.add_argument('--force-mn',type=float,default=2);ap.add_argument('--push-seconds',type=float);ap.add_argument('--capture-seconds',type=float,default=0);ap.add_argument('--angles',type=float,nargs='+',default=[0,LASER_ANGLE]);ap.add_argument('--rolling',action='store_true');ap.add_argument('--initial-speed-mm-s',type=float,default=0);ap.add_argument('--v0',type=float);ap.add_argument('--mu-low',type=float,default=.01);ap.add_argument('--mu-high',type=float,default=.0001);ap.add_argument('--rolling-lambda',type=float,default=.003);ap.add_argument('--tag',default='');ap.add_argument('--moment-scale',type=float,default=1);args=ap.parse_args()
    build=ROOT/'scratch/laser-tilt-20261009-source'/('build' if args.rolling else 'build-no-rolling');binary=next(build.glob('hysteresis*.pyd'))
    spec=importlib.util.spec_from_file_location('hysteresis',binary);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if not np.isfinite(args.moment_scale) or args.moment_scale<=0: raise ValueError('moment scale must be finite and positive')
    if hasattr(mod,'configure_magnet_moment_scale'): mod.configure_magnet_moment_scale(args.moment_scale)
    elif args.moment_scale!=1: raise RuntimeError('This audit build does not support magnetic moment scaling')
    if args.rolling:mod.configure_rolling_resistance(args.mu_low,args.mu_high,args.rolling_lambda)
    omega=args.rpm*2*np.pi/60;mass=4*np.pi/3*(args.diameter_mm/2000)**3*7850
    out=ROOT/'runs/laser-tilt-20261009'/f'rpm{args.rpm:g}_diam{args.diameter_mm:g}_dt{args.dt:g}{args.tag}';out.mkdir(parents=True,exist_ok=True)
    records=[];plots=[]
    for tilt in args.angles:
        started=time.perf_counter();tag=f'tilt_{tilt:.9f}';folder=out/tag;folder.mkdir(exist_ok=True)
        mod.configure_laser_tilt(tilt,np.pi/2)
        params=mod.SolverParams(sr=args.diameter_mm/12, somega=args.rpm/100)
        release_v0=args.v0 if args.v0 is not None else -args.initial_speed_mm_s/1000/.07/omega
        state=defaults.prepare_initial_state(mod,params,args.dt,v0=release_v0);mode=None;direction=None
        def advance(seconds,force):
            nonlocal state,mode,direction
            params.reverse_push_acceleration=force*.001/mass*SCALE/omega**2
            nsteps=round(seconds*omega/args.dt);step=0;rows=[np.r_[0,state,float(mode) if mode is not None else 0,force]]
            while step<nsteps:
                chunk=min(1 if mode is None else 80,nsteps-step)
                try:
                    state,mode,direction=mod.evolve_trajectory(state,args.dt,chunk,params,mode,direction,step)
                except RuntimeError as exc:
                    np.savez_compressed(folder/'failed_stage_trace.npz',state_table=np.asarray(rows))
                    raise RuntimeError(f'{exc}; segment_time_s={step*args.dt/omega}; external_force_mN={force}; prior_radius_mm={np.linalg.norm(state[1:3])*1000/SCALE}; prior_state={state.tolist()}') from exc
                step+=chunk;rows.append(np.r_[step*args.dt/omega,state,float(mode),force])
            return np.array(rows)
        print(json.dumps({'start':tag,'rpm':args.rpm,'diameter':args.diameter_mm,'rolling_enabled':args.rolling}),flush=True)
        try:
            capture=advance(args.capture_seconds,0);parts=[];elapsed=0;push_stable=False;per=[]
            maximum_push=args.push_seconds if args.push_seconds is not None else 60
            while elapsed<maximum_push-1e-4:
                part=advance(min(.5,maximum_push-elapsed),args.force_mn);part[:,0]+=elapsed;elapsed=part[-1,0];parts.append(part if not parts else part[1:])
                push=np.vstack(parts);_,cr,per=laps(push,omega)
                if len(per)>=5 and elapsed>=20 and np.ptp(per[-3:])/np.mean(per[-3:])<.01:
                    push_stable=True;break
            push=np.vstack(parts) if parts else capture[-1:].copy()
            force_off_state=state.copy();force_off_mode=bool(mode);force_off_direction=np.asarray(direction).copy()
            np.savez_compressed(folder/'startup.npz',capture=capture,push=push,force_off_state=force_off_state)
            print(json.dumps({'pushed':tag,'elapsed_s':elapsed,'stable':push_stable,'laps':len(per)}),flush=True)
            free=advance(args.free_seconds,0)
            # Extend a drifting rotating solution; do not force or reset it.
            metric,curve=summarize(free,omega,mod,params)
            if not metric.get('steady_free_orbit',False) and metric.get('tail_reverse_fraction',0)>.99:
                extension=advance(180,0);extension[:,0]+=free[-1,0];free=np.vstack([free,extension[1:]]);metric,curve=summarize(free,omega,mod,params)
            if curve is not None:
                metric['gravity_only_rolling_estimate_percent']=float(9.81*np.sin(tilt)*(metric['radius_mean_mm']/1000)/((1+.4)*(metric['mean_tangential_speed_mm_s']/1000)**2)*100)
                plots.append((tilt,metric,curve))
            np.savez_compressed(folder/'trajectory.npz',capture=capture,push=push,free=free,force_off_state=force_off_state)
            np.savetxt(folder/'free_trajectory.csv',free,delimiter=',',header='time_s,disk_angle,x,y,vx,vy,qw,qx,qy,qz,wx,wy,wz,hx,hy,hz,sliding,force_mN',comments='')
            record={'status':'ok','tilt_rad':tilt,'tilt_deg':float(np.degrees(tilt)),'rpm':args.rpm,'ball_diameter_mm':args.diameter_mm,'ball_mass_g':mass*1000,'dt':args.dt,'added_rolling_resistance_enabled':args.rolling,'push_force_mN':args.force_mn,'push_duration_s':elapsed,'stable_under_push':push_stable,'free_duration_s':float(free[-1,0]),'force_off_state':force_off_state.tolist(),'force_off_sliding':force_off_mode,'force_off_sliding_direction':force_off_direction.tolist(),'force_removal_state_discontinuity':float(np.max(abs(force_off_state-free[0,1:16]))),'metrics':metric,'wall_time_s':time.perf_counter()-started,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'tilt_force_model':'theta*(gamma_g-Fmag_z), constant downhill azimuth pi/2; fixed magnetic gap, not a geometric tilted-field recalculation'}
        except RuntimeError as exc:record={'status':'failed','tilt_rad':tilt,'rpm':args.rpm,'error':str(exc),'wall_time_s':time.perf_counter()-started}
        record.update({'rpm':args.rpm,'ball_diameter_mm':args.diameter_mm,'magnet_center_to_ball_center_mm':39.62,'field_height_normalized':2.264,'dt':args.dt,'added_rolling_resistance_enabled':args.rolling,'requested_push_duration_s':args.push_seconds,'push_force_mN':args.force_mn,'initial_speed_mm_s':args.initial_speed_mm_s,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'field_lut_sha256':hashlib.sha256((build.parent/'field-lut/field_lut_fp64.bin').read_bytes()).hexdigest()})
        record.update({'v0':release_v0,'rolling_mu_low':args.mu_low,'rolling_mu_high':args.mu_high,'rolling_lambda':args.rolling_lambda})
        record.update({'magnet_moment_scale':args.moment_scale,'magnet_beta':.056999326368103*args.moment_scale,'magnet_moment_A_m2':.056999326368103*args.moment_scale*4*np.pi*1.1e6/SCALE**3})
        records.append(record);(folder/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(record,indent=2),flush=True)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,1,figsize=(11,9),constrained_layout=True)
        for tilt,metric,(tt,phi,speed,grid,pred) in plots:
            label=f'{np.degrees(tilt):.3f} deg; A1/mean={metric["first_harmonic_relative_half_amplitude_percent"]:.2f}%'
            take=tt>=tt[-1]-10;ax[0].plot(tt[take]-tt[-1],speed[take],label=label,lw=.9)
            ax[1].scatter(np.degrees(phi[::5]),speed[::5],s=2,alpha=.15)
            ax[1].plot(np.degrees(grid),pred,label=label,lw=2)
        ax[0].set(xlabel='Time relative to last completed lap (s)',ylabel='Tangential speed (mm/s)')
        ax[1].set(xlabel='phi (degrees)',ylabel='Tangential speed (mm/s)')
        for a in ax:a.legend();a.grid(alpha=.25)
        fig.suptitle(f'Fixed Q235 | disk {args.rpm:g} rpm | ball {args.diameter_mm:g} mm | added rolling resistance '+('ON' if args.rolling else 'OFF'))
        fig.savefig(out/'speed_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
