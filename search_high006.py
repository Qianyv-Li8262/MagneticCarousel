"""Current cylinder model, 1.5x moment: finite blows and passive rolling trials.

All times and accelerations in this driver are SI. Only low/high rolling
coefficients, spin transition rate, and hysteresis material parameters vary.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import animate_one_trajectory as animation

ROOT=Path(__file__).resolve().parent
SHARP=True
SOURCE=ROOT/'scratch/hysteresis15-search-source'
OUT=ROOT/'runs/high006-hysteresis15-20261015'
OUT.mkdir(parents=True,exist_ok=True)
MANIFEST=json.loads((SOURCE/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('hysteresis15_trial',MANIFEST['binary'])
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
        windows.append(dict(normal_spin_rms_dimensionless=float(np.sqrt(np.mean(u[:,12]**2))),rate_rad_s=rate,radius_mm=float(r.mean()/KS*1000),
                            reverse_fraction=float(np.mean(vt<0))))
    rates=np.array([w['rate_rad_s'] for w in windows]);spread=float(np.ptp(rates)/max(abs(rates.mean()),1e-9))
    rate=float(np.polyfit(tail[:,0]-tail[:,0].mean(),np.unwrap(np.arctan2(tail[:,3],tail[:,2])),1)[0])
    radius=np.linalg.norm(tail[:,2:4],axis=1)/KS*1000
    return dict(period_s=float(-2*np.pi/rate) if rate<-.01 else None,
                stationary=bool(not error and rate<-.1 and spread<.005 and np.ptp([w['radius_mm'] for w in windows])<.5 and min(w['reverse_fraction'] for w in windows)>.99),
                windows=windows,rate_relative_spread=spread,radius_mean_mm=float(radius.mean()),
                radius_range_mm=[float(radius.min()),float(radius.max())])

def run(task):
    cfg=dict(low=.0049,high=.006,transition=.1,exponent=2.,accel=0.,duration=0.,start=10.,
             seconds=300.,dt=.0005,rpm=200.,v0=-.05,phi0=0.,seed=None)
    cfg.update(task)
    if (cfg['low'],cfg['high'],cfg['transition'],cfg.get('exponent',2.)) != (.0049,.006,.1,2.):
        raise ValueError('User fixed rolling law must not be tuned')
    cfg.setdefault('material',[605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882])
    key=hashlib.sha256(json.dumps(dict(cfg=cfg,binary=MANIFEST['binary_sha256'],version=1),sort_keys=True).encode()).hexdigest()[:20]
    path=OUT/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    omega=cfg['rpm']*2*np.pi/60
    p=mod.SolverParams(sr=R/mod.reference_ball_radius_m, somega=omega/mod.reference_disk_omega_rad_s)
    mod.configure_material(*cfg['material'])
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
    rec['tail_normalized_rolling_spin_range']=[float(w.min()),float(w.max())]
    rec['high_plateau_reached']=bool(np.min(mu)>.005999)
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

def score(r):
    if r.get('error') or not r.get('period_s'):return 1e6
    return abs(np.log(r['period_s']/2.416257))+2*min(5,r.get('rate_relative_spread',5))+abs(r.get('radius_mean_mm',70)-70)/70

def all_records(names):
    out={}
    for name in names:
        p=OUT/(name+'.json')
        if p.exists():out.update({r['key']:r for r in json.loads(p.read_text())})
    return list(out.values())

if __name__=='__main__':
    from scipy.stats import qmc
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['scan','access','refine','verify','perturb']);args=ap.parse_args()
    if args.action=='scan':
        prior=ROOT/'runs/hysteresis15-search-20261015'
        keys=['571d98971d29fb765919','d627be1daf646b367264','ea82771dc91b34c90b27','1816308f00771308f922','0c5b7079b745f080fbe2']
        mats=[json.loads((prior/(key+'.json')).read_text())['config']['material'] for key in keys]
        mats += [[605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882]]
        tasks=[dict(material=mat,accel=a,duration=d,seconds=450.) for mat in mats for a,d in [(0.,0.),(.1,3.),(.3,1.5)]]
        # Search stronger irreversible magnetic response around the preserved material.
        base=np.array(mats[0])
        for x in qmc.LatinHypercube(4,seed=2026101570).random(96):
            mat=base*10**(1.4*(2*x-1));mat[2]=.01+.89*x[2]
            tasks.append(dict(material=mat.tolist(),accel=.15,duration=3.,seconds=450.))
        for x in qmc.LatinHypercube(4,seed=2026101571).random(96):
            mat=[10**(2+3*x[0]),10**(-5+4*x[1]),.01+.97*x[2],10**(-6+4.5*x[3])]
            tasks.append(dict(material=mat,accel=.15,duration=3.,seconds=450.))
        batch(tasks,'scan')
    elif args.action=='access':
        rr=sorted([r for r in all_records(['scan']) if not r.get('error') and r.get('period_s')],key=score)
        mats=[]
        for r in rr:
            if r['config']['material'] not in mats:mats.append(r['config']['material'])
            if len(mats)==16:break
        prior=ROOT/'runs/hysteresis15-search-20261015'
        for name in ['refined','target','validated']:
            for r in json.loads((prior/(name+'.json')).read_text()):
                if r.get('stationary') and r['config']['material'] not in mats:mats.append(r['config']['material'])
        baseline=json.loads((ROOT/'baselines/hysteresis15-T1p866883-20261015/validation/long_1500s_half_dt.json').read_text())
        tasks=[]
        for mat in mats:
            tasks += [dict(material=mat,accel=.3,duration=1.5,start=start,seconds=600.) for start in [0.,30.]]
            tasks += [dict(material=mat,seed=baseline['final_state'],start=0.,accel=0.,duration=0.,seconds=600.,test_label='continuation_from_preserved_orbit')]
        batch(tasks,'access')
    elif args.action=='refine':
        rr=sorted([r for r in all_records(['scan','access']) if not r.get('error') and r.get('period_s')],key=score)
        roots=[]
        for r in rr:
            if all(r['config']['material']!=v['config']['material'] for v in roots):roots.append(r)
            if len(roots)==4:break
        tasks=[]
        for j,r in enumerate(roots):
            v=np.array(r['config']['material'])
            for x in qmc.LatinHypercube(4,seed=2026101580+j).random(32):
                mat=v*3**(2*x-1);mat[2]=np.clip(v[2]+.15*(2*x[2]-1),.001,.99)
                tasks.append(dict(material=mat.tolist(),accel=.15,duration=3.,seconds=600.))
        batch(tasks,'refined')
    elif args.action=='verify':
        rr=sorted([r for r in all_records(['scan','access','refined']) if not r.get('error') and r.get('period_s')],key=score)
        roots=[]
        for r in rr:
            if all(r['config']['material']!=v['config']['material'] for v in roots):roots.append(r)
            if len(roots)==4:break
        batch([dict(r['config'],seconds=1800.,dt=.00025) for r in roots],'validated')
    else:
        rr=sorted([r for r in all_records(['validated']) if r.get('stationary') and r.get('high_plateau_reached')],key=score)
        tasks=[]
        for r in rr[:2]:
            for factor in [.98,1.02]:
                seed=np.array(r['final_state']);seed[3:5]*=factor;seed[9:11]*=factor
                tasks.append(dict(r['config'],seed=seed.tolist(),start=0.,accel=0.,duration=0.,seconds=600.,test_label='speed_'+str(factor)))
            tasks += [dict(r['config'],accel=0.,duration=0.,seconds=900.,test_label='no_blow'),
                      dict(r['config'],accel=.1,duration=3.,seconds=900.,test_label='reference_blow')]
        batch(tasks,'perturbed')

