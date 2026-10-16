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
OUT=ROOT/'runs/hysteresis15-search-20261015'
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
    cfg.setdefault('material',[605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882])
    key=hashlib.sha256(json.dumps(dict(cfg=cfg,binary=MANIFEST['binary_sha256'],version=1),sort_keys=True).encode()).hexdigest()[:20]
    path=OUT/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    omega=cfg['rpm']*2*np.pi/60;alpha=KS*R
    gamma=3*4*np.pi*1e-7*1.1e6**2*KS**2/(DENSITY*omega**2)
    p=mod.SolverParams(alpha,3*1.225*.47/(8*DENSITY*R*KS),KS*9.81/omega**2,gamma,gamma/(.4*alpha**2))
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
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['coarse','wide','focus','refine','target','verify','perturb']);args=ap.parse_args()
    if args.action=='coarse':
        mats=[[605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882],
              [6147.45620724178,.005580649834954433,.35,1e-5],
              [4245.43876763801,.01089492205460393,.17438865816095875,2.2407394677774326e-5]]
        mats += [[mur,.0021323083849413885,c,k] for mur in [1000.,3000.,10000.] for c in [.1,.35,.65] for k in [.00001,.0001,.001]]
        mats += [[10**(2.3+2.2*x[0]),10**(-4+2.7*x[1]),.03+.92*x[2],10**(-6+4*x[3])] for x in qmc.LatinHypercube(4,seed=2026101520).random(96)]
        laws=[dict(low=.0013,high=.0013,transition=1.5,exponent=2.),
              dict(low=.0049,high=.001,transition=1.,exponent=8.)]
        tasks=[dict(**law,material=mat,accel=.1,duration=3.,seconds=300.) for mat in mats for law in laws]
        # Reference and previously promising material regions, multiple losses and no-blow controls.
        tasks += [dict(low=mu,high=mu,material=mat,accel=a,duration=d,seconds=300.) for mat in mats[:3] for mu in [.0003,.0006,.002,.003,.0049] for a,d in [(0.,0.),(.1,3.)]]
        batch(tasks,'coarse')
    elif args.action=='wide':
        tasks=[]
        for x in qmc.LatinHypercube(7,seed=2026101550).random(160):
            mat=[10**(2.3+2.2*x[0]),10**(-4+2.7*x[1]),.02+.95*x[2],10**(-6+4*x[3])]
            hi=10**(-6+3.5*x[4]);wc=.3+1.5*x[5]
            for decay in [False,True]:
                tasks.append(dict(material=mat,low=.0049 if decay else hi,high=hi,transition=wc,exponent=8. if decay else 2.,accel=.1,duration=1.+3*x[6],seconds=360.))
        centers=[[6147.45620724178,.005580649834954433,.35,1e-5],[4245.43876763801,.01089492205460393,.17438865816095875,2.2407394677774326e-5]]
        tasks += [dict(material=mat,low=mu,high=mu,accel=a,duration=d,seconds=360.) for mat in centers for mu in [0.,.00001,.00003,.0001,.0003,.0006] for a,d in [(0.,0.),(.1,3.)]]
        batch(tasks,'wide')
    elif args.action=='focus':
        mat=json.loads((OUT/'571d98971d29fb765919.json').read_text())['config']['material']
        laws=[dict(low=mu,high=mu,transition=1.5,exponent=2.) for mu in [.0003,.0006,.001,.0013,.0016,.002,.0025,.003,.004,.0049]]
        laws += [dict(low=.0049,high=hi,transition=wc,exponent=8.) for hi in [.001,.0016,.0025] for wc in [.5,.9,1.3]]
        batch([dict(**law,material=mat,accel=a,duration=d,seconds=500.) for law in laws for a,d in [(0.,0.),(.1,3.),(.3,1.)]],'focus')
    elif args.action=='refine':
        rr=all_records(['coarse','wide']);good=sorted([r for r in rr if not r.get('error') and (r.get('period_s') or 1e6)<15],key=score)
        stable=[r for r in good if r.get('stationary')]
        if stable:good=stable
        roots=[]
        for r in good:
            if all(r['config']['material']!=v['config']['material'] for v in roots):roots.append(r)
            if len(roots)==3:break
        tasks=[]
        for j,r in enumerate(roots):
            c=r['config'];v=np.array(c['material'])
            for x in qmc.LatinHypercube(6,seed=2026101530+j).random(60):
                mat=v*3**(2*x[:4]-1);mat[2]=np.clip(v[2]+.2*(2*x[2]-1),.01,.98)
                high=c['high']*3**(2*x[4]-1);wc=c['transition']*1.5**(2*x[5]-1)
                tasks.append(dict(c,material=mat.tolist(),high=high,low=high if c['low']==c['high'] else .0049,transition=wc,seconds=450.))
        batch(tasks,'refined')
    elif args.action=='target':
        tasks=[]
        for j,key in enumerate(['1816308f00771308f922','0c5b7079b745f080fbe2']):
            c=json.loads((OUT/(key+'.json')).read_text())['config'];v=np.array(c['material'])
            for x in qmc.LatinHypercube(6,seed=2026101560+j).random(48):
                mat=v*1.8**(2*x[:4]-1);mat[2]=np.clip(v[2]+.08*(2*x[2]-1),.01,.98)
                high=c['high']*3**(2*x[4]-1)
                tasks.append(dict(c,material=mat.tolist(),high=high,low=high if c['low']==c['high'] else .0049,transition=c['transition']*1.4**(2*x[5]-1),seconds=600.,dt=.0005))
        batch(tasks,'target')
    elif args.action=='verify':
        rr=sorted(all_records(['coarse','wide','focus','refined','target']),key=score)
        chosen=[]
        for kind in ['constant','decay']:
            group=[r for r in rr if (r['config']['low']==r['config']['high'])==(kind=='constant') and not r.get('error') and (r.get('period_s') or 1e6)<15]
            chosen+=group[:3]
        batch([dict(r['config'],seconds=1500.,dt=.00025) for r in chosen],'validated')
    else:
        rr=sorted([r for r in all_records(['validated']) if r.get('stationary')],key=score)
        tasks=[]
        for r in rr[:2]:
            state=np.array(r['final_state'])
            for speed in [.98,1.02]:
                seed=state.copy();seed[3:5]*=speed;seed[9:11]*=speed
                tasks.append(dict(r['config'],seed=seed.tolist(),start=0.,duration=0.,accel=0.,seconds=900.,dt=.00025))
            tasks += [dict(r['config'],accel=0.,duration=0.,seconds=1200.),dict(r['config'],accel=.1,duration=2.8,seconds=1200.),dict(r['config'],accel=.1,duration=3.2,seconds=1200.)]
        batch(tasks,'perturbed')

