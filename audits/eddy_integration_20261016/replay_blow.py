"""Replay the animation initial state and production blowing schedule headlessly."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import animate_one_trajectory as animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--no-blow', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'scratch/eddy_integration_20261016' / args.label
    output.mkdir(parents=True, exist_ok=True)
    config = animation.read_build_config(kahan=False)
    module = animation.load_hysteresis()
    params = module.SolverParams(sr=animation.SR, somega=animation.ANIMATION_SOMEGA)
    dt = animation.timestep_for_magnets(module.magnet_number)
    state = animation.prepare_initial_state(
        module, params, dt, v0=animation.ANIMATION_V0, s0=0, phi0=0,
        magnetize_turns=animation.MAGNETIZE_TURNS, disk_phase=animation.DISK_PHASE)
    initial = state.copy()
    schedule = animation.BlowSchedule(
        module, params, dt, sr=animation.SR, somega=animation.ANIMATION_SOMEGA,
        start=None if args.no_blow else 75, duration=None if args.no_blow else 200,
        force_mn=animation.DEFAULT_BLOW_FORCE_MN)
    times, states, modes, forces = [0.], [state.copy()], [False], [0.]
    mode = direction = None
    failure = None
    started = time.perf_counter()
    total = round(650/dt)
    sample_steps = round(.05/dt)
    for begin in range(0, total, sample_steps):
        try:
            state, mode, direction = schedule.evolve(state, min(sample_steps,total-begin),mode,direction)
        except RuntimeError as error:
            failure = str(error)
            break
        times.append(schedule.elapsed)
        states.append(state.copy())
        modes.append(mode)
        forces.append(animation.DEFAULT_BLOW_FORCE_MN if schedule.active(schedule.elapsed) else 0.)
        if len(times)%2000 == 0:
            print(f'{args.label}: t_2d={schedule.elapsed:.2f}',flush=True)
    times, states = np.asarray(times), np.asarray(states)
    units = module.physical_units(sr=animation.SR,somega=animation.ANIMATION_SOMEGA)
    length = units.get('length_m', 0.070/(module.magnet_number/2))
    omega = units['disk_omega_rad_s']
    radius = np.linalg.norm(states[:,1:3],axis=1)
    speed = np.linalg.norm(states[:,3:5],axis=1)*length*omega
    tangential = (states[:,1]*states[:,4]-states[:,2]*states[:,3])/radius*length*omega
    np.savez_compressed(output/'trace.npz',time_2d=times,state=states,sliding=modes,blow_mN=forces,initial=initial)
    np.savetxt(output/'observables.csv',np.column_stack((times,times/omega,radius*length*1000,
        speed*1000,tangential*1000,modes,forces)),delimiter=',',
        header='time_2d,time_s,radius_mm,speed_mm_s,tangential_mm_s,sliding,blow_mN',comments='')
    tail = times>=600
    off = int(np.argmin(abs(times-275)))
    record = dict(label=args.label,config=config,state_size=module.state_size,
        eddy_currents_enabled=bool(getattr(module,'eddy_currents_enabled',False)),
        initial=initial.tolist(),sr=animation.SR,somega=animation.ANIMATION_SOMEGA,
        v0=animation.ANIMATION_V0,s0=0,phi0=0,disk_phase=animation.DISK_PHASE,
        magnetize_turns=animation.MAGNETIZE_TURNS,dt=dt,
        blow_start_2d=None if args.no_blow else 75,blow_duration_2d=None if args.no_blow else 200,
        blow_force_mN=animation.DEFAULT_BLOW_FORCE_MN,
        completed_time_2d=float(times[-1]),completed_time_s=float(times[-1]/omega),
        max_speed_mm_s=float(speed.max()*1000),speed_at_blow_off_mm_s=float(speed[off]*1000),
        radius_at_blow_off_mm=float(radius[off]*length*1000),
        tail_mean_speed_mm_s=float(np.mean(speed[tail])*1000) if tail.any() else None,
        tail_mean_radius_mm=float(np.mean(radius[tail])*length*1000) if tail.any() else None,
        rolling_fraction=float(1-np.mean(modes[1:])),failure=failure,
        wall_seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in ('config','initial')},indent=2),flush=True)
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
