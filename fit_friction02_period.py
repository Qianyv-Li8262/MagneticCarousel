"""Freeze contact Mu=0.2; test and tune JA, planar rolling and Z damping."""
import os
os.environ['IYPT16_PERIOD_MODEL']='spin-mu02'
import argparse,json
import numpy as np
from scipy.stats import qmc
import fit_horizontal_period as f

OLD=f.ROOT/'runs/horizontal-spin-period-20261015'
CENTER=json.loads((OLD/'3414e055b2cdf35c17a3.json').read_text())
VALUES=[CENTER['parameters'][k] for k in f.KEYS]
TAU=CENTER['spin_tau_s'];LIMIT=2*np.pi

def records(names):
    out=[]
    for name in names:
        p=f.OUT/(name+'.json')
        if p.exists():out.extend(json.loads(p.read_text()))
    return out

def settled(r):
    w=[v['spin_rms_rad_s'] for v in r.get('windows',[])]
    return bool(r.get('stationary') and w and np.ptp(w)<max(.02,.01*np.mean(w)))

def score(r):
    if r.get('error') or not r.get('period_s') or r.get('reverse_fraction',0)<.99:return 1e6
    return abs(np.log(r['period_s']/f.TARGETS[1]['ball_period_target_s']))+min(2,r.get('rate_relative_spread',2))+.15*max(0,np.log(max(r.get('normal_spin_rms_rad_s',1e6),1e-9)/LIMIT))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['baseline','joint','refine','verify','final-check']);args=ap.parse_args()
    assert f.mod.contact_friction==.2
    if args.action=='baseline':
        (f.OUT/'manifest.json').write_text(json.dumps(dict(center=CENTER,contact_friction=.2,
            fixed=dict(height_mm=39.62,magnet_moment_scale=1,tilt_rad=0,constant_rpm=True),targets=f.TARGETS,
            build=json.loads((f.SOURCE/'build_manifest.json').read_text()),
            note='One fixed Mu governs sliding Coulomb friction and rolling feasibility; Mu is never sampled. Independent rolling and normal-spin resistance remain tunable.'),indent=2))
        tasks=[(VALUES,dict(seconds=1200.,dt=.00025,spin_tau=TAU,save=True)),
               (VALUES,dict(seconds=1200.,dt=.00025,spin_tau=TAU,seed=CENTER['final_state'],save=True)),
               (VALUES,dict(seconds=1200.,dt=.00025,spin_tau=TAU,slip0=-.075,save=True)),
               (VALUES,dict(seconds=600.,spin_tau=15.)),
               (f.BASE+[1e-5,1e-4,.004],dict(seconds=600.,spin_tau=15.))]
        f.batch(tasks,'baseline')
    elif args.action=='joint':
        tasks=[]
        for x in qmc.LatinHypercube(8,seed=2026101503).random(128):
            v=np.array(VALUES)*4**(2*x[:7]-1);v[2]=.08+.65*x[2]
            tasks.append((v.tolist(),dict(seconds=360.,spin_tau=float(TAU*10**(2*x[7]-1)))))
        for x in qmc.LatinHypercube(8,seed=2026101504).random(64):
            high=10**(-6+3.5*x[5])
            v=[10**(2+3*x[0]),10**(-4+2.5*x[1]),.02+.93*x[2],10**(-7+5.5*x[3]),
               high*10**(-2+2*x[4]),high,10**(-6+4.5*x[6])]
            tasks.append((v,dict(seconds=360.,spin_tau=10**(-1+3.5*x[7]))))
        f.batch(tasks,'joint')
    elif args.action=='refine':
        allr=records(['baseline','joint']);eligible=[r for r in allr if not r.get('error') and r.get('period_s')]
        eligible=list({tuple([r['parameters'][k] for k in f.KEYS]+[r['spin_tau_s']]):r for r in eligible}.values())
        low=sorted([r for r in eligible if r.get('normal_spin_rms_rad_s',1e6)<=LIMIT],key=score)
        roots=low[:2]+sorted(eligible,key=score)[:2]
        unique={tuple(list(r['parameters'].values())+[r['spin_tau_s']]):r for r in roots}
        tasks=[]
        for j,r in enumerate(unique.values()):
            base=np.array([r['parameters'][k] for k in f.KEYS]);tau=r['spin_tau_s']
            for x in qmc.LatinHypercube(8,seed=2026101520+j).random(48):
                v=base*1.7**(2*x[:7]-1);v[2]=np.clip(base[2]+.12*(2*x[2]-1),.01,.99)
                tasks.append((v.tolist(),dict(seconds=480.,spin_tau=float(tau*2**(2*x[7]-1)))))
        f.batch(tasks,'refined')
    elif args.action=='verify':
        allr=records(['baseline','joint','refined']);eligible=[r for r in allr if settled(r)]
        low=[r for r in eligible if r['normal_spin_rms_rad_s']<=LIMIT]
        selected=sorted(low,key=score)[:3]+sorted(eligible,key=lambda r:r['period_s'])[:2]
        # Follow up any faster surviving but not-yet-stationary runs as well.
        unsettled=[r for r in allr if not r.get('error') and not settled(r) and r.get('period_s') and r['period_s']<2.8]
        selected+=sorted(unsettled,key=score)[:2]
        unique={tuple([r['parameters'][k] for k in f.KEYS]+[r['spin_tau_s']]):r for r in selected}
        f.batch([([r['parameters'][k] for k in f.KEYS],dict(seconds=1200.,dt=.00025,spin_tau=r['spin_tau_s'],save=True)) for r in unique.values()],'validated')
    else:
        allr=records(['validated']);eligible=[r for r in allr if settled(r)]
        low=[r for r in eligible if r['normal_spin_rms_rad_s']<=LIMIT]
        selected=sorted(low,key=score)[:1]+sorted(eligible,key=lambda r:r['period_s'])[:1]
        tasks=[]
        for r in {r['key']:r for r in selected}.values():
            v=[r['parameters'][k] for k in f.KEYS];tau=r['spin_tau_s']
            tasks += [(v,dict(seconds=1200.,dt=.0005,spin_tau=tau,save=True)),
                      (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,rpm=f.TARGETS[0]['disk_rpm'],save=True)),
                      (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,slip0=-.075,save=True)),
                      (v,dict(seconds=1200.,dt=.00025,spin_tau=tau,v0=-.05,save=True))]
        f.batch(tasks,'final_check')
