"""Verify the archived protocol and its independent ordinary force-free replay."""
import json
from pathlib import Path
import numpy as np
from replay_prepared_release import load_case

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/sustained-push'
RUN=ROOT/'runs/sustained-push/full-protocol-replay'


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    table=np.load(RUN/'trajectory.npz')['state_table']
    # At withdrawal there are two records: the last forced and first unforced.
    off=summary['force_off_after_release_s']
    indices=np.flatnonzero((table[:,0]>=off)&(table[:,17]==0))
    free=table[indices]
    case,ordinary=load_case(DEST/'reverse_running_initial_state.json')
    assert not hasattr(ordinary,'configure_air_push'), 'Expected the ordinary archived build.'
    params=ordinary.SolverParams(**case['solver_parameters'])
    state=np.asarray(case['initial_state'],dtype=ordinary.real_dtype)
    mode=case['initial_sliding'];direction=np.asarray(case['initial_sliding_direction'],dtype=ordinary.real_dtype)
    differences=[float(np.max(np.abs(np.r_[state,float(mode)]-free[0,1:17])))]
    step=0
    for i in range(1,len(free)):
        count=round((free[i,0]-free[i-1,0])*case['disk_angular_speed_rad_s']/case['dt'])
        state,mode,direction=ordinary.evolve_trajectory(state,case['dt'],count,params,mode,direction,step)
        step+=count
        differences.append(float(np.max(np.abs(np.r_[state,float(mode)]-free[i,1:17]))))
    max_difference=max(differences)
    start=summary['push_start_after_release_s']
    assert np.all(table[table[:,0]<start,17]==0)
    assert np.all(free[:,17]==0)
    assert abs(summary['actual_push_seconds']-protocol['observed_push_seconds'])<1e-12
    assert summary['max_force_switch_state_difference']==0
    assert max_difference<1e-10, max_difference
    result=dict(ordinary_build_has_external_force_interface=False,
        ordinary_vs_force_capable_free_max_state_difference=max_difference,
        force_switch_state_difference=summary['max_force_switch_state_difference'],
        force_free_external_force_mN=float(np.max(np.abs(free[:,17]))),
        force_free_observation_seconds=float(free[-1,0]-free[0,0]),
        archived_protocol_push_time_match=True,checked_free_samples=len(free),
        zero_force_baseline=json.loads((DEST/'baseline_validation.json').read_text(encoding='utf-8')))
    (DEST/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
