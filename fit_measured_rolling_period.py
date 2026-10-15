"""Fit horizontal orbital periods with measured friction frozen throughout."""
import os
os.environ['IYPT16_PERIOD_MODEL']='spin-measured'
import argparse,json
import numpy as np
from scipy.stats import qmc
import fit_horizontal_period as f
OLD=json.loads((f.ROOT/'runs/friction02-period-20261015/summary.json').read_text())['best']
FIXED=[.00490,.00490,0.]
CENTER=[OLD['parameters'][k] for k in f.KEYS[:4]]
TAU=OLD['spin_tau_s']

def records(names):
    out=[]
    for name in names:
        p=f.OUT/(name+'.json')
        if p.exists():out.extend(json.loads(p.read_text()))
    return out

def score(r):
    if r.get('error') or not r.get('period_s') or r.get('reverse_fraction',0)<.99:return 1e6
    return abs(np.log(r['period_s']/f.TARGETS[1]['ball_period_target_s']))+min(2,r.get('rate_relative_spread',2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['baseline','scan','startup','blow','blow-scan','drift-check','refine','verify','crosscheck']);args=ap.parse_args()
    if args.action=='baseline':
        (f.OUT/'manifest.json').write_text(json.dumps(dict(measured_rolling=dict(low=.0049,high=.0049,lambda_=0.,transition=1.5,epsilon=.002),
            contact_friction=.2,fixed=dict(tilt_rad=0.,height_mm=39.62,magnet_moment_scale=1.,constant_rpm=True),
            tunable=['mu_r','alpha_JA','C_JA','K','existing_spin_tau_s'],targets=f.TARGETS,
            build=json.loads((f.SOURCE/'build_manifest.json').read_text()),
            note='Measured friction is never sampled or changed. Existing phenomenological normal-spin law retained; it is not a measured spin-torque calibration.'),indent=2))
        f.batch([(CENTER+FIXED,dict(seconds=360.,spin_tau=TAU,save=True)),
                 (CENTER+FIXED,dict(seconds=360.,spin_tau=TAU,slip0=-.075)),
                 (f.BASE+FIXED,dict(seconds=360.,spin_tau=TAU)),
                 (CENTER+FIXED,dict(seconds=360.,spin_tau=15.))],'baseline')
    elif args.action=='scan':
        tasks=[]
        for x in qmc.LatinHypercube(5,seed=2026101516).random(24):
            v=np.array(CENTER)*5**(2*x[:4]-1);v[2]=np.clip(CENTER[2]+.3*(2*x[2]-1),.01,.95)
            tasks.append((v.tolist()+FIXED,dict(seconds=360.,spin_tau=float(TAU*10**(2*x[4]-1)))))
        for x in qmc.LatinHypercube(5,seed=2026101517).random(40):
            v=[10**(2+3*x[0]),10**(-4+2.5*x[1]),.02+.93*x[2],10**(-7+5.5*x[3])]
            tasks.append((v+FIXED,dict(seconds=360.,spin_tau=float(10**(-1+3*x[4])))))
        f.batch(tasks,'scan')
    elif args.action=='startup':
        tasks=[]
        scan=records(['scan']);best_drift=max(scan,key=lambda r:r.get('reverse_fraction',0))
        for v,tau in [(CENTER+FIXED,TAU),(f.BASE+FIXED,TAU),
                      ([best_drift['parameters'][k] for k in f.KEYS],best_drift['spin_tau_s'])]:
            for speed in [-.10,-.125,-.15]:tasks.append((v,dict(seconds=360.,v0=speed,spin_tau=tau,save=True)))
            tasks.append((v,dict(seconds=360.,seed=OLD['final_state'],spin_tau=tau,save=True)))
        f.batch(tasks,'startup')
    elif args.action=='blow':
        tasks=[]
        # Start from the already observed slow-drift state, then remove all drive.
        locked=json.loads((f.OUT/'2465b192b2994b233819.json').read_text())['final_state']
        for accel in [.05,.15,.5,1.5]:
            for duration in [.5,1.,2.]:
                tasks.append((CENTER+FIXED,dict(seconds=360.,spin_tau=TAU,seed=locked,
                    push_duration_s=duration,push_acceleration_m_s2=accel,push_stop_speed_m_s=.19,save=True)))
        f.batch(tasks,'blow')
    elif args.action=='blow-scan':
        tasks=[]
        for r in records(['scan']):
            v=[r['parameters'][k] for k in f.KEYS]
            tasks.append((v,dict(seconds=360.,spin_tau=r['spin_tau_s'],seed=r['final_state'],
                push_duration_s=1.,push_acceleration_m_s2=.5,push_stop_speed_m_s=.17,save=True)))
        f.batch(tasks,'blow_scan')
    elif args.action=='drift-check':
        locked=json.loads((f.OUT/'2465b192b2994b233819.json').read_text())['final_state'];tasks=[]
        for step in [.0005,.00025]:
            tasks.append((CENTER+FIXED,dict(seconds=1200.,dt=step,spin_tau=TAU,save=True)))
            tasks.append((CENTER+FIXED,dict(seconds=1200.,dt=step,spin_tau=TAU,seed=locked,
                push_duration_s=1.,push_acceleration_m_s2=.5,push_stop_speed_m_s=.17,save=True)))
        f.batch(tasks,'drift_validation')
    elif args.action=='refine':
        rr=[r for r in records(['baseline','scan']) if score(r)<1e5]
        roots=sorted(rr,key=score)[:2];tasks=[]
        for j,r in enumerate(roots):
            v=np.array([r['parameters'][k] for k in f.KEYS[:4]])
            for x in qmc.LatinHypercube(5,seed=2026101530+j).random(16):
                vv=v*1.8**(2*x[:4]-1);vv[2]=np.clip(v[2]+.12*(2*x[2]-1),.01,.99)
                tasks.append((vv.tolist()+FIXED,dict(seconds=480.,spin_tau=float(r['spin_tau_s']*2**(2*x[4]-1)))))
        f.batch(tasks,'refined')
    elif args.action=='verify':
        rr=sorted([r for r in records(['baseline','scan','refined']) if score(r)<1e5],key=score)
        chosen=[]
        for r in rr:
            if all(r['parameters']!=v['parameters'] for v in chosen):chosen.append(r)
            if len(chosen)==3:break
        f.batch([([r['parameters'][k] for k in f.KEYS],dict(seconds=1200.,dt=.00025,spin_tau=r['spin_tau_s'],save=True)) for r in chosen],'validated')
    else:
        rr=[r for r in records(['validated']) if r.get('stationary') and r.get('spin_stationary')]
        if not rr:raise RuntimeError('No converged candidate; do not label a transient period a fit')
        r=min(rr,key=score);v=[r['parameters'][k] for k in f.KEYS];tau=r['spin_tau_s']
        f.batch([(v,dict(seconds=1200.,dt=.0005,spin_tau=tau,save=True)),
                 (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,rpm=f.TARGETS[0]['disk_rpm'],save=True)),
                 (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,slip0=-.075,save=True))],'crosscheck')
