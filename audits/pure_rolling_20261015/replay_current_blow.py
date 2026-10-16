"""Replay production animation solver/schedule, without display timing overhead."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import animate_one_trajectory as a
ap=argparse.ArgumentParser();ap.add_argument('--half-step',action='store_true');args=ap.parse_args()
OUT=Path(__file__).parent/('current_blow_halfstep' if args.half_step else 'current_blow')
OUT.mkdir(exist_ok=True)
config=a.read_build_config(kahan=False)
mod=a.load_hysteresis()
p=mod.SolverParams(*a.scaled_model_parameters(mod.magnet_number,somega=2))
dt=.00025 if args.half_step else .0005
s=a.prepare_initial_state(mod,p,dt,v0=a.ANIMATION_V0,s0=0,phi0=0,disk_phase=a.DISK_PHASE,magnetize_turns=a.MAGNETIZE_TURNS)
schedule=a.BlowSchedule(mod,p,dt,sr=1,somega=2,start=75,duration=200,force_mn=.7)
rows=[np.r_[0,s,0,0]]; mode=None; direction=None
tic=time.perf_counter(); failure=None
while schedule.elapsed<650-1e-9:
    try:
        s,mode,direction=schedule.evolve(s,round(.05/dt),mode,direction)
    except RuntimeError as exc:
        failure=str(exc); break
    rows.append(np.r_[schedule.elapsed,s,float(mode),.7 if schedule.active(schedule.elapsed) else 0])
trace=np.array(rows)
np.savez_compressed(OUT/'trace.npz',table=trace)
np.savetxt(OUT/'trace.csv',trace,delimiter=',',fmt='%.17g',header='time_2d,disk_angle,x,y,vx,vy,qw,qx,qy,qz,wx,wy,wz,hqx,hqy,hqz,sliding,blow_mN',comments='')
r=np.linalg.norm(trace[:,2:4],axis=1)*.0175
v=np.linalg.norm(trace[:,4:6],axis=1)*schedule.omega*.0175
vt=(trace[:,2]*trace[:,5]-trace[:,3]*trace[:,4])/np.linalg.norm(trace[:,2:4],axis=1)*schedule.omega*.0175
rec=dict(config=config,dt=dt,blow_start_2d=75,blow_duration_2d=200,blow_force_mN=.7,
         duration_s=float(trace[-1,0]/schedule.omega),max_speed_m_s=float(max(v)),
         speed_at_blow_off_m_s=float(v[np.argmin(abs(trace[:,0]-275))]),
         radius_at_blow_off_mm=float(r[np.argmin(abs(trace[:,0]-275))]*1000),
         speed_tail_mean_m_s=float(np.mean(v[trace[:,0]>600])),
         rolling_fraction=float(np.mean(trace[:,16]==0)),failure=failure,wall_seconds=time.perf_counter()-tic)
(OUT/'summary.json').write_text(json.dumps(rec,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in rec.items() if k!='config'},indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
t=trace[:,0]/schedule.omega
axs[0].plot(t,v*1000,label='Speed');axs[0].plot(t,-vt*1000,label='Reverse tangential speed',alpha=.7);axs[0].legend();axs[0].set_ylabel('Speed (mm/s)')
axs[1].plot(t,r*1000);axs[1].set_ylabel('Radius (mm)')
axs[2].plot(t,trace[:,17]);axs[2].set_ylabel('Applied force (mN)');axs[2].set_xlabel('Time after release (s)')
for ax in axs:
    ax.axvspan(75/schedule.omega,275/schedule.omega,color='orange',alpha=.15);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'replay.png',dpi=150)
