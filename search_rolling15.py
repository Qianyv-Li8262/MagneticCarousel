"""Current cylinder model, 1.5x moment: finite blows and passive rolling trials.

All times and accelerations in this driver are SI. Only low/high rolling
coefficients and the spin transition rate vary; material is the formal baseline.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import animate_one_trajectory as animation

ROOT=Path(__file__).resolve().parent
SHARP=os.environ.get('IYPT16_ROLLING15_SHARP')=='1'
SOURCE=ROOT/('scratch/rolling15-sharp-source' if SHARP else 'scratch/rolling15-search-source')
OUT=ROOT/'runs/rolling15-search-20261015'
OUT.mkdir(parents=True,exist_ok=True)
MANIFEST=json.loads((SOURCE/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('rolling15_sharp' if SHARP else 'rolling15_trial',MANIFEST['binary'])
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
KS=4/.07; R=.006; DENSITY=7850.; MASS=4*np.pi/3*R**3*DENSITY

def metrics(rows,seconds,error):
    tail=rows[rows[:,0]>=max(0,seconds-min(180,seconds/2))]
    windows=[]
    for u in np.array_split(tail,3):
        if len(u)<5:return dict(stationary=False)
        t=u[:,0];r=np.linalg.norm(u[:,2:4],axis=1)
        rate=float(np.polyfit(t-t.mean(),np.unwrap(np.arctan2(u[:,3],u[:,2])),1)[0])
        vt=(u[:,2]*u[:,5]-u[:,3]*u[:,4])/r
        windows.append(dict(rate_rad_s=rate,radius_mm=float(r.mean()/KS*1000),
                            reverse_fraction=float(np.mean(vt<0))))
    rates=np.array([w['rate_rad_s'] for w in windows]);spread=float(np.ptp(rates)/max(abs(rates.mean()),1e-9))
    rate=float(np.polyfit(tail[:,0]-tail[:,0].mean(),np.unwrap(np.arctan2(tail[:,3],tail[:,2])),1)[0])
    radius=np.linalg.norm(tail[:,2:4],axis=1)/KS*1000
    return dict(period_s=float(-2*np.pi/rate) if rate<-.01 else None,
                stationary=bool(not error and rate<-.1 and spread<.005 and np.ptp([w['radius_mm'] for w in windows])<.5 and min(w['reverse_fraction'] for w in windows)>.99),
                windows=windows,rate_relative_spread=spread,radius_mean_mm=float(radius.mean()),
                radius_range_mm=[float(radius.min()),float(radius.max())])

def run(task):
    cfg=dict(low=.0049,high=.0049,transition=1.5,accel=0.,duration=0.,start=10.,
             seconds=300.,dt=.0005,rpm=200.,v0=-.05,phi0=0.,seed=None)
    cfg.update(task)
    key=hashlib.sha256(json.dumps(dict(cfg=cfg,binary=MANIFEST['binary_sha256'],version=1),sort_keys=True).encode()).hexdigest()[:20]
    path=OUT/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    omega=cfg['rpm']*2*np.pi/60
    p=mod.SolverParams(sr=R/mod.reference_ball_radius_m, somega=omega/mod.reference_disk_omega_rad_s)
    if SHARP:mod.configure_trial_rolling(cfg['low'],cfg['high'],cfg['transition'],cfg.get('exponent',2.))
    else:mod.configure_trial_rolling(cfg['low'],cfg['high'],cfg['transition'])
    state=animation.prepare_initial_state(mod,p,cfg['dt'],v0=cfg['v0'],phi0=cfg['phi0']) if cfg['seed'] is None else np.array(cfg['seed'],dtype=float)
    mode=direction=None;step=0;dt=cfg['dt'];limit=round(cfg['seconds']*omega/dt)
    first=round(cfg['start']*omega/dt);last=round((cfg['start']+cfg['duration'])*omega/dt)
    rows=[np.r_[0.,state,0.]];error=None;began=time.perf_counter()
    while step<limit:
        active=first<=step<last
        p.reverse_push_acceleration=cfg['accel']*KS/omega**2 if active else 0.
        chunk=max(1,round((.005 if active else .04)*omega/dt))
        n=min(1 if mode is None else chunk,limit-step)
        for edge in [first,last]:
            if step<edge:n=min(n,edge-step)
        try:state,mode,direction=mod.evolve_trajectory(state,dt,n,p,mode,direction,step)
        except RuntimeError as exc:error=str(exc);break
        step+=n;rows.append(np.r_[step*dt/omega,state,float(mode)])
    p.reverse_push_acceleration=0.
    rows=np.array(rows);completed=float(rows[-1,0]);speed=np.linalg.norm(rows[:,4:6],axis=1)*omega/KS
    rec=dict(key=key,config=cfg,error=error,completed_s=completed,wall_s=time.perf_counter()-began,
             final_state=state.tolist(),force_mN=cfg['accel']*MASS*1000,
             completed_free_s=max(0,completed-cfg['start']-cfg['duration']),
             max_speed_m_s=float(speed.max()),binary_sha256=MANIFEST['binary_sha256'],
             lut_sha256=MANIFEST['lut_sha256'],moment_beta=.0854985)
    rec.update(metrics(rows,completed,error))
    if completed<cfg['seconds']-.05 or rec['completed_free_s']<min(100,cfg['seconds']/2):rec['stationary']=False
    tail=rows[rows[:,0]>max(0,completed-90)]
    w=np.linalg.norm(tail[:,10:12],axis=1)
    mu=cfg['high']+(cfg['low']-cfg['high'])*np.exp(-(w/cfg['transition'])**cfg.get('exponent',2.))
    rec['tail_effective_mu_range']=[float(mu.min()),float(mu.max())]
    rec['tail_speed_range_m_s']=[float(np.linalg.norm(tail[:,4:6],axis=1).min()*omega/KS),float(np.linalg.norm(tail[:,4:6],axis=1).max()*omega/KS)]
    rec['tail_normal_spin_rms_rad_s']=float(np.sqrt(np.mean(tail[:,12]**2))*omega)
    path.write_text(json.dumps(rec,indent=2));np.savez_compressed(path.with_suffix('.npz'),rows=rows)
    return rec

def batch(tasks,name):
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run,tasks):
            records.append(r);(OUT/(name+'.json')).write_text(json.dumps(records,indent=2))
            print(json.dumps(dict(key=r['key'],low=r['config']['low'],high=r['config']['high'],wc=r['config']['transition'],a=r['config']['accel'],duration=r['config']['duration'],T=r.get('period_s'),stable=r['stationary'],completed=r['completed_s'],error=r['error'])),flush=True)
    return records

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['scan','kick','boundary','fine','sharp','basin','crosscheck','verify']);args=ap.parse_args()
    if args.action=='scan':
        laws=[dict(low=mu,high=mu) for mu in [.0049,.003,.002,.001,.0005,.0002,0.]]
        laws += [dict(low=.0049,high=hi,transition=wc) for hi in [.0002,.0005,.001] for wc in [.3,.6,1.,1.5]]
        tasks=[dict(**law,accel=a,duration=d) for law in laws for a,d in [(0.,0.),(.03,1.),(.1,1.),(.03,5.)]]
        batch(tasks,'screen')
    elif args.action=='kick':
        laws=[dict(low=mu,high=mu) for mu in [.0011,.0012,.0013,.0014,.0015,.0016,.0018]]
        laws += [dict(low=.0049,high=hi,transition=wc) for hi in [.0005,.001,.0013,.0016] for wc in [.6,1.,1.5]]
        batch([dict(**law,accel=a,duration=d) for law in laws for a,d in [(.1,3.),(.3,1.),(.1,5.)]],'kick')
    elif args.action=='boundary':
        laws=[dict(low=mu,high=mu) for mu in np.linspace(.001,.002,21)]
        batch([dict(**law,accel=a,duration=d,seconds=600.) for law in laws for a,d in [(0.,0.),(.1,3.)]],'boundary')
    elif args.action=='fine':
        laws=[dict(low=mu,high=mu) for mu in np.linspace(.00128,.0014,25)]
        batch([dict(**law,accel=a,duration=d,seconds=600.) for law in laws for a,d in [(0.,0.),(.1,3.),(.15,2.5)]],'fine')
    elif args.action=='sharp':
        assert SHARP
        laws=[dict(low=.0049,high=hi,transition=wc,exponent=power) for hi in [.0005,.001,.0013] for wc in [.6,1.,1.4] for power in [4.,8.]]
        batch([dict(**law,accel=a,duration=d,seconds=400.) for law in laws for a,d in [(.1,3.),(.3,1.),(.1,4.)]],'sharp')
    elif args.action=='basin':
        laws=[dict(low=mu,high=mu) for mu in [.00130,.00132,.00134,.00136,.0014,.0016]]
        laws += [dict(low=.0049,high=hi,transition=wc) for hi,wc in [(.001,.6),(.0013,.6),(.0005,1.),(.001,1.)]]
        batch([dict(**law,accel=a,duration=d,start=start,seconds=500.) for law in laws for a,d,start in [(.1,2.,0.),(.1,3.,0.),(.1,4.,0.),(.03,10.,10.),(.03,15.,10.),(.1,3.,30.),(.1,4.,30.),(.3,1.5,30.)]],'basin')
    elif args.action=='crosscheck':
        keys=(['011c8e68fe83bf67c5b2','3b155469db777e071537'] if SHARP else
              ['164c63c17e0d043bd3eb','7afb06699447f5dbe26f','fb3e93e526df8ebe0ccb','4944fff6f5bb18a036d8'])
        tasks=[dict(json.loads((OUT/(key+'.json')).read_text())['config'],dt=.00025,seconds=1200.) for key in keys]
        if not SHARP:
            tasks += [dict(low=.0049,high=.001,transition=.6,accel=.3,duration=1.,dt=.00025,seconds=1200.),
                      dict(low=.0049,high=.001,transition=1.,accel=.1,duration=3.,dt=.00025,seconds=1200.)]
        batch(tasks,'crosscheck_sharp' if SHARP else 'crosscheck')
    else:
        screen=json.loads((OUT/'screen.json').read_text())
        good=[r for r in screen if r.get('stationary')]
        chosen=[]
        for kind in ['constant','decay']:
            group=[r for r in good if (r['config']['low']==r['config']['high'])==(kind=='constant')]
            group.sort(key=lambda r:abs(np.log(r['period_s']/2.416257)))
            chosen+=group[:3]
        batch([dict(r['config'],seconds=1200.,dt=.00025) for r in chosen],'validated')
