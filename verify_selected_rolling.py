"""Validate the actual production animation build over 600 physical seconds."""
from pathlib import Path
import argparse,json,time
import numpy as np
import animate_one_trajectory as animation
from laser_tilt_experiment import summarize
from assess_bounded_orbit import assess
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=root/'runs/laser-tilt-20261009/rolling_candidates');ap.add_argument('--free-seconds',type=float,default=600);args=ap.parse_args()
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
config=animation.read_build_config(kahan=False)
mod=animation.load_hysteresis();omega=200*2*np.pi/60;dt=.0005
params=mod.SolverParams(sr=1.0, somega=2)
state=animation.prepare_initial_state(mod,params,dt,v0=-.075)
initial=state.copy();mode=None;direction=None;step=0;count=round(args.free_seconds*omega/dt)
rows=[np.r_[0,state,0,0]];started=time.perf_counter()
while step<count:
    chunk=min(80,count-step)
    state,mode,direction=mod.evolve_trajectory(state,dt,chunk,params,mode,direction,step)
    step+=chunk;rows.append(np.r_[step*dt/omega,state,float(mode),0])
table=np.array(rows);metrics,_=summarize(table,omega,mod,params)
compiled=config['model_parameters']
def literal(name): return float(str(compiled[name]).removeprefix('REAL_LITERAL(').removesuffix(')'))
record={'rpm':200,'v0':-.075,'initial_position_mm':[70,0],'initial_velocity_mm_s':(initial[3:5]*omega/(4/.07)*1000).tolist(),'initial_spin_rad_s':(initial[9:12]*omega).tolist(),'initial_state':initial.tolist(),'held_magnetization_turns':2,'held_magnetization_seconds':.6,'free_duration_s':float(table[-1,0]),'blowing_force_mN':0,'rolling_mu_low':literal('kRollingMuLow'),'rolling_mu_high':literal('kRollingMuHigh'),'rolling_lambda':literal('kRollingLambda'),'metrics':metrics,'bounded_assessment':assess(table,omega),'build_config':config,'wall_time_s':time.perf_counter()-started}
np.savez_compressed(out/'production_600s_trajectory.npz',free=table,initial_state=initial)
(out/'production_600s_summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
assert record['bounded_assessment']['bounded_stationary'], 'Production build did not reach bounded stationary reverse motion'
print(json.dumps(metrics,indent=2))
