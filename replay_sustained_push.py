"""Replay release, capture, a following reverse push, and complete force removal.

Loads the archived force-capable build, preserving the complete magnetic,
attitude and contact state between phases. --animate displays the saved run.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from replay_prepared_release import load_case, plot_result, animate_result
from sustained_push_experiment import baseline, adaptive_push, advance, motion_metrics, save_trace

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'results/sustained-push/protocol.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/sustained-push/full-protocol-replay')
    parser.add_argument('--animate',action='store_true')
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    case,mod=load_case(args.protocol.parent/protocol['force_capable_case'])
    params=mod.SolverParams(*case['solver_parameters'])
    dt=protocol['dt']
    if not np.isclose(case['disk_angular_speed_rad_s'],100*2*np.pi/60):
        raise ValueError('This replay protocol uses the archived 100 rpm conversion.')
    try:
        captured=baseline(mod,params,case,dt,protocol['capture_after_release_s'])
        push,state,mode,direction,stop=adaptive_push(
            mod,params,captured,protocol['force_mN'],dt,protocol['max_push_seconds'])
        if not stop['stable_under_push']:
            raise RuntimeError('The push did not reach the saved per-lap stopping criterion.')
        free,_,_,_=advance(mod,params,state,mode,direction,
            protocol['force_free_observation_seconds'],dt,40,force_mn=0.)
    finally:
        mod.configure_air_push(0.,True,0.,0.)
    start=captured[0][-1,0]
    off=start+push[-1,0]
    # Duplicate boundary times record the external force switch with unchanged state.
    table=np.vstack([captured[0],push+np.r_[start,np.zeros(17)],
                     free+np.r_[off,np.zeros(17)]])
    args.output.mkdir(parents=True,exist_ok=True)
    save_trace(args.output/'trajectory.npz',table)
    times=table[:,0];states=table[:,1:16]
    pos,angle,speed=plot_result(times,states,args.output/'trajectory.png',
        kick_at=off,title='Release, capture, continuous reverse push, then FORCE OFF (contact slip remains)',
        capture_candidate=False,force_interval=(start,off))
    np.savetxt(args.output/'trajectory.csv',np.column_stack([
        times,pos,angle,speed,table[:,17],table[:,16]]),delimiter=',',comments='',
        header='time_after_release_s,x_mm,y_mm,angle_rad,speed_mm_s,external_force_mN,sliding')
    summary=dict(full_experimental_fit=False,push_start_after_release_s=start,
        force_off_after_release_s=off,**stop,after_force_removal=motion_metrics(free),
        state_at_force_removal=state.tolist(),mode_at_force_removal=bool(mode),
        direction_at_force_removal=np.asarray(direction).tolist(),
        max_free_external_force_mN=float(np.max(np.abs(free[:,17]))),
        max_force_switch_state_difference=float(max(
            np.max(np.abs(captured[0][-1,1:17]-push[0,1:17])),
            np.max(np.abs(push[-1,1:17]-free[0,1:17])))))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),**summary),indent=2),flush=True)
    if args.animate:
        animate_result(times,states,max(1,round(.03/(40*dt/case['disk_angular_speed_rad_s']))),
            title='Release > capture > following push > force off | contact slip remains')


if __name__=='__main__':main()
