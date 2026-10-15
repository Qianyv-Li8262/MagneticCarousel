"""Joint local JA/planar-rolling/Z-damping search around the 2.78579 s orbit."""
import os
os.environ['IYPT16_PERIOD_MODEL']='spin'
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import qmc
import fit_horizontal_period as f

OLD=f.ROOT/'runs/horizontal-period-20261015'
CENTER=json.loads((OLD/'31c9d917fd098144de73.json').read_text())
VALUES=[CENTER['parameters'][k] for k in f.KEYS]
SPIN_LIMIT=2*np.pi # Earlier observed order of magnitude: 1--2 seconds per spin.

def score(r):
    if r.get('error') or not r.get('period_s') or r.get('reverse_fraction',0)<.99:return 1e6
    return abs(np.log(r['period_s']/f.TARGETS[1]['ball_period_target_s']))+min(2,r.get('rate_relative_spread',2))+.15*max(0,np.log(max(r.get('normal_spin_rms_rad_s',1e6),1e-9)/SPIN_LIMIT))

def check():
    omega=f.TARGETS[1]['disk_rpm']*2*np.pi/60
    f.mod.configure_material(*VALUES[:4]);f.mod.configure_magnet_moment_scale(1e-12)
    f.mod.configure_rolling_resistance(0,0,0);f.mod.configure_normal_spin_damping(1/omega)
    p=f.mod.SolverParams(f.KSPACE*f.BALL_R,0,f.KSPACE*9.81/omega**2,1e-20,1e-20)
    state=np.zeros(15);state[1]=4;state[5]=1;state[11]=10/omega
    dt=.0005;n=round(3*omega/dt)
    state,_,_=f.mod.evolve_trajectory(state,dt,n,p,None,None,0)
    expected=10*np.exp(-n*dt/omega);relative=abs(state[11]*omega/expected-1)
    assert relative<1e-6,relative
    for bad in [-1,np.nan,np.inf]:
        try:f.mod.configure_normal_spin_damping(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid damping accepted')
    r=f.run(VALUES,seconds=1200,dt=.00025,spin_tau=None)
    regression=abs(r['period_s']/CENTER['period_s']-1)
    assert regression<1e-10,regression
    (f.OUT/'verification.json').write_text(json.dumps(dict(free_decay_relative_error=relative,zero_damping_period_relative_error=regression,zero_damping_record=r['key']),indent=2))
    print('CHECKS',relative,regression,flush=True)

def load(names):
    records=[]
    for name in names:
        p=f.OUT/(name+'.json')
        if p.exists():records.extend(json.loads(p.read_text()))
    return records

def spin_settled(r):
    windows=[w['spin_rms_rad_s'] for w in r.get('windows',[])]
    return bool(windows and np.ptp(windows)<max(.02,.01*np.mean(windows)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','baseline','joint','refine','verify','final-check']);args=ap.parse_args()
    if args.action=='check':check()
    elif args.action=='baseline':
        manifest=dict(center=CENTER,targets=f.TARGETS,spin_reference_rad_s=[np.pi,2*np.pi],
                      fixed=dict(height_mm=39.62,magnet_moment_scale=1,tilt_rad=0,constant_rpm=True),
                      build=json.loads((f.SOURCE/'build_manifest.json').read_text()),
                      note='The historical 1--2 s spin period is only a reference, not a new measured target. Both signs use RMS/absolute spin.')
        (f.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        tasks=[(VALUES,dict(seconds=600.,spin_tau=tau,seed=seed)) for tau in [None,.1,.3,1,3,10,15,30,100,300] for seed in [None,CENTER['final_state']]]
        f.batch(tasks,'baseline')
    elif args.action=='joint':
        tasks=[]
        for x in qmc.LatinHypercube(8,seed=2026101502).random(160):
            values=[VALUES[0]*2**(2*x[0]-1),VALUES[1]*2**(2*x[1]-1),.18+.42*x[2],
                    VALUES[3]*4**(2*x[3]-1),VALUES[4]*4**(2*x[4]-1),
                    VALUES[5]*4**(2*x[5]-1),VALUES[6]*4**(2*x[6]-1)]
            tau=10**(-1+3.5*x[7])
            tasks.append((values,dict(seconds=360.,spin_tau=tau)))
        f.batch(tasks,'joint')
    elif args.action=='refine':
        records=load(['baseline','joint'])
        eligible=[r for r in records if not r.get('error') and r.get('period_s')]
        low=sorted([r for r in eligible if r.get('normal_spin_rms_rad_s',1e6)<=SPIN_LIMIT],key=score)
        roots=(low[:2]+sorted(eligible,key=score)[:2])
        unique={tuple(list(r['parameters'].values())+[r.get('spin_tau_s')]):r for r in roots}
        tasks=[]
        for j,r in enumerate(unique.values()):
            base=np.array([r['parameters'][k] for k in f.KEYS]);tau=r.get('spin_tau_s') or 100.
            for x in qmc.LatinHypercube(8,seed=2026101510+j).random(48):
                v=base*1.6**(2*x[:7]-1);v[2]=np.clip(base[2]+.12*(2*x[2]-1),.02,.98)
                tasks.append((v.tolist(),dict(seconds=480.,spin_tau=float(tau*2**(2*x[7]-1)))))
        f.batch(tasks,'refined')
    elif args.action=='verify':
        records=load(['baseline','joint','refined'])
        eligible=[r for r in records if r.get('stationary') and spin_settled(r) and r.get('spin_tau_s') is not None]
        low=[r for r in eligible if r.get('normal_spin_rms_rad_s',1e6)<=SPIN_LIMIT]
        selected=sorted(low,key=score)[:3]+sorted(eligible,key=score)[:2]
        selected+=sorted(eligible,key=lambda r:r.get('period_s',1e6))[:2]
        selected.append(next(r for r in records if r.get('spin_tau_s')==15 and r.get('seed') is None))
        unique={tuple(list(r['parameters'].values())+[r['spin_tau_s']]):r for r in selected}
        f.batch([(list(r['parameters'].values()),dict(seconds=1200.,dt=.00025,spin_tau=r['spin_tau_s'],save=True)) for r in unique.values()],'validated')
    else:
        records=load(['validated']);eligible=[r for r in records if r.get('stationary') and spin_settled(r)]
        low=[r for r in eligible if r.get('normal_spin_rms_rad_s',1e6)<=SPIN_LIMIT]
        selected=sorted(low,key=score)[:1]+sorted(eligible,key=score)[:1]
        unique={r['key']:r for r in selected};tasks=[]
        for r in unique.values():
            v=[r['parameters'][k] for k in f.KEYS];tau=r['spin_tau_s']
            tasks += [(v,dict(seconds=1200.,dt=.0005,spin_tau=tau,save=True)),
                      (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,rpm=f.TARGETS[0]['disk_rpm'],save=True)),
                      (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,v0=-.05,save=True))]
        f.batch(tasks,'final_check')
