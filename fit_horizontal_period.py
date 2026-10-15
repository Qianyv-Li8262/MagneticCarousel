"""Horizontal disk: fixed cylinder moment/height, JA + planar rolling fit.

Uses the existing runtime-configurable, zero-added-Z-damping material module.
Each worker owns its module globals. Results are cached with all inputs/hashes.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc
import animate_one_trajectory as animation

ROOT = Path(__file__).resolve().parent
MEASURED = os.environ.get('IYPT16_PERIOD_MODEL') == 'spin-measured'
MU02 = os.environ.get('IYPT16_PERIOD_MODEL') in ('spin-mu02','spin-measured')
SPIN = os.environ.get('IYPT16_PERIOD_MODEL') in ('spin','spin-mu02','spin-measured')
OUT = ROOT / ('runs/measured-rolling-period-20261015' if MEASURED else ('runs/friction02-period-20261015' if MU02 else ('runs/horizontal-spin-period-20261015' if SPIN else 'runs/horizontal-period-20261015')))
OUT.mkdir(parents=True, exist_ok=True)
NATIVE = (ROOT/'scratch/horizontal-period-20261015-source').exists()
SOURCE = ROOT / ('scratch/measured-rolling-period-20261015-source' if MEASURED else ('scratch/friction02-period-20261015-source' if MU02 else ('scratch/horizontal-spin-period-20261015-source' if SPIN else ('scratch/horizontal-period-20261015-source' if NATIVE else 'scratch/material-response-20261013-source'))))
MODULE_NAME = 'measured_rolling_audit' if MEASURED else ('friction02_audit' if MU02 else ('horizontal_spin_audit' if SPIN else ('horizontal_period_audit' if NATIVE else 'material_audit')))
BINARY = next(SOURCE.glob(MODULE_NAME+'*.pyd'))
LUT = ROOT / 'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
BINARY_HASH, LUT_HASH = sha(BINARY), sha(LUT)
spec = importlib.util.spec_from_file_location(MODULE_NAME, BINARY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
BASE = [605.8974337495617, .0021323083849413885, .22415154978863677, .005375927782155882]
TARGETS = json.loads((ROOT.parent / 'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())
KSPACE, BALL_R, DENSITY, MS = 4/.07, .006, 7850., 1.1e6
KEYS = ['mu_r', 'alpha_JA', 'C_JA', 'K', 'rolling_low', 'rolling_high', 'rolling_lambda']

def run(values, rpm=TARGETS[1]['disk_rpm'], seconds=180., dt=.0005,
        v0=-.075, save=False, seed=None, spin_tau=None, slip0=0.,
        push_duration_s=0.,push_acceleration_m_s2=0.,push_stop_speed_m_s=None):
    values = [float(x) for x in values]
    if MEASURED:
        if values[4:] != [.00490,.00490,0.]:
            raise ValueError('Measured rolling parameters are fixed, not fit variables')
        assert tuple(mod.measured_rolling_parameters)==(.00490,.00490,1.5,.002,0.)
        assert not mod.inclined_disk_enabled
    omega = rpm * 2*np.pi/60
    alpha = KSPACE * BALL_R
    gamma = 3*(4*np.pi*1e-7)*MS**2*KSPACE**2/(DENSITY*omega**2)
    p = mod.SolverParams(alpha, 3*1.225*.47/(8*DENSITY*BALL_R*KSPACE),
                         KSPACE*9.81/omega**2, gamma, gamma/(.4*alpha**2))
    p.reverse_push_acceleration = 0
    mod.configure_material(*values[:4])
    mod.configure_magnet_moment_scale(1.)
    mod.configure_laser_tilt(0., 0.)
    # Lambda and rate constants are specified at the long video's RPM.
    # Rescale them when checking another RPM, preserving the SI torque law.
    ratio = TARGETS[1]['disk_rpm']/rpm
    if hasattr(mod,'configure_rolling_time_scale'):
        mod.configure_rolling_time_scale(ratio)
    elif abs(ratio-1)>1e-12:
        raise ValueError('Multi-RPM validation requires the horizontal native build')
    mod.configure_rolling_resistance(values[4], values[5], values[6]*ratio)
    if spin_tau is not None and (not np.isfinite(spin_tau) or spin_tau<=0):
        raise ValueError('spin_tau must be positive seconds, or None for zero damping')
    spin_lambda=0. if spin_tau is None else 1/(omega*spin_tau)
    if hasattr(mod,'configure_normal_spin_damping'):
        mod.configure_normal_spin_damping(spin_lambda)
    elif spin_tau is not None:
        raise ValueError('normal-spin damping requires the spin build')
    inputs = dict(parameters=dict(zip(KEYS, values)), rpm=rpm, seconds=seconds,
                  dt=dt, v0=v0, tilt_rad=0., normal_spin_damping=0.,
                  height_mm=39.62, moment_Am2=4.222663,
                  binary_sha256=BINARY_HASH, lut_sha256=LUT_HASH,
                  seed=seed, metric_version=1)
    if SPIN:
        inputs.update(spin_tau_s=spin_tau,normal_spin_damping=spin_lambda)
    if MU02:
        assert mod.contact_friction==.2
        inputs['contact_friction']=.2
    if slip0!=0.:
        inputs['initial_slip_normalized']=float(slip0)
    if push_duration_s>0:
        if push_acceleration_m_s2<=0:raise ValueError('Pulse acceleration must be positive')
        inputs['temporary_reverse_push']=dict(max_duration_s=push_duration_s,acceleration_m_s2=push_acceleration_m_s2,
                                               stop_speed_m_s=push_stop_speed_m_s,protocol_version=2)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    path = OUT / (key+'.json')
    if path.exists() and (not save or path.with_suffix('.npz').exists()):
        return json.loads(path.read_text())
    began = time.perf_counter()
    state = (animation.prepare_initial_state(mod,p,dt,v0=v0,s0=slip0) if seed is None else np.array(seed,dtype=float))
    mode = direction = None
    step = 0
    maximum = round(seconds*omega/dt)
    chunk = max(1, round(.04*omega/dt))
    rows = [np.r_[0., state, 0.]]
    error = None
    push_stop_step=round(push_duration_s*omega/dt)
    push_active=push_stop_step>0;push_ended_s=None;push_peak_speed=0.;push_ended_speed=None
    try:
        while step < maximum:
            n = min(1 if mode is None else chunk, maximum-step)
            speed_si=float(np.linalg.norm(state[3:5])*omega/KSPACE)
            if push_active and (step>=push_stop_step or (push_stop_speed_m_s is not None and speed_si>=push_stop_speed_m_s)):
                push_active=False;push_ended_s=step*dt/omega;push_ended_speed=speed_si
            p.reverse_push_acceleration=KSPACE*push_acceleration_m_s2/omega**2 if push_active else 0.
            if push_active:
                n=min(n,push_stop_step-step,max(1,round(.005*omega/dt)));push_peak_speed=max(push_peak_speed,speed_si)
            state,mode,direction = mod.evolve_trajectory(state,dt,n,p,mode,direction,step)
            if push_active:push_peak_speed=max(push_peak_speed,float(np.linalg.norm(state[3:5])*omega/KSPACE))
            step += n
            rows.append(np.r_[step*dt/omega,state,float(mode)])
    except RuntimeError as exc:
        error = str(exc)
    rows = np.array(rows)
    rec = dict(**inputs, key=key, error=error, completed_s=float(rows[-1,0]),
               wall_s=time.perf_counter()-began, final_state=state.tolist())
    if push_stop_step>0:
        rec.update(push_ended_s=push_ended_s,push_peak_sampled_speed_m_s=push_peak_speed,
                   push_ended_speed_m_s=push_ended_speed,
                   final_push_acceleration_m_s2=float(p.reverse_push_acceleration)*omega**2/KSPACE)
    if len(rows)>20:
        used=rows[rows[:,0]>=max(0., rows[-1,0]-min(90.,seconds/2))]
        t=used[:,0]; s=used[:,1:16]
        phi=np.unwrap(np.arctan2(s[:,2],s[:,1]))
        rate=np.polyfit(t-t.mean(),phi,1)[0]
        radius=np.linalg.norm(s[:,1:3],axis=1)/KSPACE*1000
        tang=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/np.linalg.norm(s[:,1:3],axis=1)
        windows=[]
        for u in np.array_split(used,3):
            ang=np.unwrap(np.arctan2(u[:,3],u[:,2]))
            slope=np.polyfit(u[:,0]-u[:,0].mean(),ang,1)[0]
            windows.append(dict(rate_rad_s=float(slope),period_s=float(-2*np.pi/slope) if slope<0 else None,
                                radius_mm=float(np.linalg.norm(u[:,2:4],axis=1).mean()/KSPACE*1000),
                                spin_rms_rad_s=float(np.sqrt(np.mean(u[:,12]**2))*omega)))
        rates=np.array([w['rate_rad_s'] for w in windows])
        spread=float(np.ptp(rates)/max(abs(rates.mean()),1e-8))
        spin_windows=np.array([w['spin_rms_rad_s'] for w in windows])
        rec.update(period_s=float(-2*np.pi/rate) if rate<-.01 else None,
                   radius_mean_mm=float(radius.mean()),radius_range_mm=[float(radius.min()),float(radius.max())],
                   reverse_fraction=float(np.mean(tang<0)),rolling_fraction=float(np.mean(used[:,16]==0)),
                   normal_spin_mean_rad_s=float(s[:,11].mean()*omega),windows=windows,
                   normal_spin_rms_rad_s=float(np.sqrt(np.mean(s[:,11]**2))*omega),
                   normal_spin_abs_mean_rad_s=float(np.mean(abs(s[:,11]))*omega),
                   normal_spin_max_abs_rad_s=float(abs(s[:,11]).max()*omega),
                   spin_stationary=bool(np.ptp(spin_windows)<max(.02,.01*spin_windows.mean())),
                   rate_relative_spread=spread,
                   stationary=bool(error is None and step==maximum and spread<.01 and
                                   np.ptp([w['radius_mm'] for w in windows])<.5 and np.mean(tang<0)>.99))
    path.write_text(json.dumps(rec,indent=2))
    if save: np.savez_compressed(path.with_suffix('.npz'),rows=rows)
    return rec

def worker(task):
    values, options = task
    try: return run(values, **options)
    except Exception as exc: return dict(parameters=dict(zip(KEYS,values)),error=repr(exc))

def batch(tasks, name):
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(worker,tasks):
            records.append(r)
            (OUT/(name+'.json')).write_text(json.dumps(records,indent=2))
            print(json.dumps(dict({k:r.get(k) for k in ['key','period_s','radius_mean_mm','stationary','wall_s']},failed=bool(r.get('error')))),flush=True)
    return records

def score(r):
    if r.get('error') or not r.get('period_s') or r.get('reverse_fraction',0)<.99: return 1e6
    return abs(np.log(r['period_s']/TARGETS[1]['ball_period_target_s'])) + min(2.,r.get('rate_relative_spread',2.))

def scan():
    values=[]
    for c in [.16,.22415155,.30,.40]:
        for high,lam in [(0.,0.),(1e-5,.0004),(1e-4,.004),(3e-4,.012)]:
            values.append([BASE[0],BASE[1],c,BASE[3],high/10,high,lam])
    # Search positive, dissipative coefficients; alpha is kept independent of mu_r.
    for x in qmc.LatinHypercube(7,seed=20261015).random(96):
        high=10**(-6+3.5*x[5])
        values.append([10**(2+2*x[0]),10**(-4+2.3*x[1]),.02+.78*x[2],
                       10**(-4+2.7*x[3]),high*10**(-2+2*x[4]),high,10**(-5+3.5*x[6])])
    return batch([(v,dict(seconds=180.)) for v in values],'coarse')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['scan','confirm','local','refine','weak-k','verify','final-check'])
    args=parser.parse_args()
    if args.action=='scan':
        manifest=dict(targets=TARGETS,fit_observables=['mean reverse orbital period'],
                      fixed=dict(tilt_rad=0,normal_spin_damping=0,magnet_moment_scale=1,height_mm=39.62,
                                 magnet_diameter_mm=28,magnet_thickness_mm=15,ring_radius_mm=70,
                                 contact_mu=.02,rolling_transition=1.5,rolling_epsilon=.002),
                      binary=str(BINARY),binary_sha256=BINARY_HASH,lut=str(LUT),lut_sha256=LUT_HASH,
                      source_manifest=json.loads((SOURCE/('build_manifest.json' if NATIVE else 'audit_build.json')).read_text()),
                      note='Constant measured disk RPM per video; no speed-modulation target. No added Z torque.')
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));scan()
    elif args.action=='confirm':
        records=json.loads((OUT/'coarse.json').read_text())
        best=sorted(records,key=score)[:10]
        batch([(list(r['parameters'].values()),dict(seconds=600.,dt=.00025,save=True)) for r in best],'confirmed')
    elif args.action=='local':
        values=[]
        for c in [.24,.28,.32,.36]:
            for kfactor in [.3,1.,3.]:
                for high,lam in [(0.,0.),(1e-5,.0004),(1e-4,.004)]:
                    values.append([BASE[0],BASE[1],c,BASE[3]*kfactor,high/10,high,lam])
        batch([(v,dict(seconds=240.,v0=-.12)) for v in values],'local')
    elif args.action=='refine':
        values=[]
        # Refine the high-spin, bounded material family discovered by the LHS.
        for c in [.35,.5,.65,.8,.95]:
            for k in [3e-5,.00013,.0005]:
                for high in [.0003,.0006,.001,.0016,.0025]:
                    values.append([6147.45620724178,.005580649834954433,c,k,high*.083572,high,5.619861426864048e-5])
        # Cover nearly reversible materials and very weak planar resistance.
        for c in [.8,.9,.95,.98,.995]:
            for high,lam in [(0.,0.),(1e-6,1e-5),(1e-5,.0004)]:
                values.append([BASE[0],BASE[1],c,BASE[3],high/10,high,lam])
        batch([(v,dict(seconds=240.)) for v in values],'refined')
    elif args.action=='weak-k':
        values=[]
        for c in [.1,.35,.65]:
            for k in [1e-7,1e-6,3e-6,1e-5,3e-5]:
                for high in [.0001,.0003,.0006,.001,.0015,.002]:
                    values.append([6147.45620724178,.005580649834954433,c,k,high*.083572,high,5.619861426864048e-5])
        batch([(v,dict(seconds=300.)) for v in values],'weak_k')
    elif args.action=='verify':
        records=[]
        for name in ['coarse','refined','weak_k']:
            records.extend(json.loads((OUT/(name+'.json')).read_text()))
        good=sorted([r for r in records if r.get('stationary')],key=score)
        selected=good[:3]
        transient=min([r for r in records if not r.get('error') and r.get('period_s')],key=score)
        if transient['key'] not in [r['key'] for r in selected]:selected.append(transient)
        tasks=[(list(r['parameters'].values()),dict(seconds=1200.,dt=.00025,save=True)) for r in selected]
        best=list(good[0]['parameters'].values())
        tasks += [(best,dict(seconds=1200.,dt=.0005,save=True)),
                  (best,dict(seconds=1200.,dt=.00025,rpm=TARGETS[0]['disk_rpm'],save=True))]
        unique={json.dumps(task,sort_keys=True):task for task in tasks}
        batch(list(unique.values()),'validated')
    else:
        records=json.loads((OUT/'validated.json').read_text())
        best=min([r for r in records if r.get('stationary') and r['rpm']==TARGETS[1]['disk_rpm']],key=score)
        values=list(best['parameters'].values())
        batch([(values,dict(seconds=1200.,dt=.0005,save=True)),
               (values,dict(seconds=1200.,dt=.00025,rpm=TARGETS[0]['disk_rpm'],save=True)),
               (values,dict(seconds=1200.,dt=.00025,v0=-.05,save=True))],'final_check')
