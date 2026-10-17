"""Isolated velocity-weakening rolling fit; never changes production configuration."""
from pathlib import Path
import os, json, shutil, subprocess, hashlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import fit_video56c6 as f

OUT=f.ROOT/'runs/video56c6-rolling-fit-20261017'
OUT.mkdir(exist_ok=True)
SOURCE=f.ROOT/'scratch/video56c6-rolling-fit-20261017-source'
OLD=f.ROOT/'runs/video56c6-scale-fit-20261017'
O=2*f.np.pi*200/60
R=.006
RAW=pd.read_csv(f.VIDEO/'kinematics.csv')
TARGET=[]
for lo in np.arange(7.,72.):
    w=(RAW.source_pts_s>=lo)&(RAW.source_pts_s<lo+1)
    v=w&(RAW.derivative_valid==1);p=w&(RAW.position_valid==1)
    if v.sum()>=5 and p.sum()>=5:
        TARGET.append([lo+.5,lo+.5-f.RELEASE,RAW.com_speed_mm_s[v].mean(),RAW.r_mm[p].mean()])
TARGET=np.array(TARGET)
QUIET=(TARGET[:,0]>=9)&(TARGET[:,0]<18)
LAUNCH=(TARGET[:,0]>=18)&(TARGET[:,0]<35)
LATE=TARGET[:,0]>=35

def prepare():
    shutil.copytree(f.SOURCE/'kernels',SOURCE/'kernels',dirs_exist_ok=True)
    p=SOURCE/'kernels/headers/hysteresis_config.hpp';s=p.read_text()
    for key in ['kRollingMuLow','kRollingMuHigh','kRollingTransitionOmega']:
        s=s.replace('CONFIG_CONST real_t '+key,'inline real_t '+key)
    p.write_text(s)
    preset=json.loads((f.ROOT/'CMakeUserPresets.json').read_text())['configurePresets'][0]
    env=os.environ.copy()
    for k,v in preset['environment'].items():env[k]=v.replace('$penv{PATH}',os.environ['PATH'])
    cmd=[preset['cacheVariables']['CMAKE_CXX_COMPILER'],'-std=c++17','-O3','-DUSE_FP64',
         '-I'+str(SOURCE/'kernels/headers'),'-ID:/cpplibs/vcpkg/installed/x64-windows/include',
         '-mavx2','-ffp-contract=off','-DFIELD_EVAL_MODE=1','-DIYPT16_FIT_ROLLING',
         str(f.ROOT/'tests/native/fit_video56c6.cpp'),'-o',str(SOURCE/'fit_rolling.exe')]
    subprocess.run(cmd,env=env,check=True)
    manifest=dict(source_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'kernels').rglob('*') if p.is_file()},
        executable_sha256=hashlib.sha256((SOURCE/'fit_rolling.exe').read_bytes()).hexdigest(),
        lut=str(f.SOURCE/'field_lut_fp64.bin'),lut_sha256=hashlib.sha256((f.SOURCE/'field_lut_fp64.bin').read_bytes()).hexdigest(),
        h_mm=32,production_untouched=True,law='mu(v)=mu_high+(mu_low-mu_high)*exp(-(v/vc)^2) under pure rolling',
        transition_speed_bounds_m_s=[.218,.300],omega_conversion='omega_c_code=vc/(R*Omega_disk); only horizontal rolling angular velocity enters')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

def seed(name):
    rec=json.loads((OLD/name).read_text());c=rec['config']
    return np.array([c['sigma']/1e7,c['mu']/f.BASE['mu'],c['c'],c['K']/f.BASE['K'],rec['scale'],c['alpha']/f.BASE['alpha'],.0049, .0025,.26])

def evaluate(x,seconds=72.,dt=.0005):
    # Different optimizer processes can request the same finite-difference point.
    # Hold a per-input file lock until both the trajectory and metadata are complete.
    token=hashlib.sha256(json.dumps(dict(x=list(map(float,x)),seconds=seconds,dt=dt),sort_keys=True).encode()).hexdigest()[:16]
    lock=OUT/(token+'.lock');deadline=time.monotonic()+120
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);break
        except FileExistsError:
            if time.monotonic()>deadline:raise TimeoutError(str(lock))
            time.sleep(.1)
    try:return _evaluate(x,seconds,dt)
    finally:lock.unlink()

def _evaluate(x,seconds=72.,dt=.0005):
    sig,mu,c,k,scale,alpha,low,high,vc=map(float,x)
    cfg=dict(f.BASE,sigma=sig*1e7,mu=mu*f.BASE['mu'],c=c,K=k*f.BASE['K'],alpha=alpha*f.BASE['alpha'],r=scale*63.092177723551295,seconds=seconds,dt=dt)
    key=hashlib.sha256(json.dumps(dict(version=2,x=list(map(float,x)),seconds=seconds,dt=dt),sort_keys=True).encode()).hexdigest()[:16]
    meta=OUT/(key+'.json');path=OUT/(key+'.csv')
    if meta.exists():return json.loads(meta.read_text())
    cmd=[str(SOURCE/'fit_rolling.exe'),str(f.SOURCE/'field_lut_fp64.bin'),str(path)]+[str(cfg[k]) for k in f.KEYS]+[str(f.eddy_table(cfg['mu'])),str(low),str(high),str(vc/(R*O))]
    started=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,check=True)
    a=pd.read_csv(path);t=a.time_s.to_numpy();phi=np.unwrap(a.phi_rad.to_numpy())
    vm=[];rm=[]
    for _,tt,_,_ in TARGET:
        w=(t>=tt-.5)&(t<tt+.5)
        vm.append(float(a.speed_mm_s[w].mean())/scale if w.any() else 1e4)
        rm.append(float(a.radius_mm[w].mean())/scale if w.any() else 1e4)
    vm=np.array(vm);rm=np.array(rm)
    w=(t>=35-f.RELEASE)&(t<=72-f.RELEASE)
    period=2*np.pi/abs(np.polyfit(t[w],phi[w],1)[0]) if w.sum()>10 else 1000.
    n=round(1/np.median(np.diff(t)));sr=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid')/scale
    late=(t>=55-f.RELEASE)&(t<=72-f.RELEASE)
    over=float(sr.max()-a.radius_mm[late].mean()/scale) if late.any() else 1000.
    # Soft measurement constraints: do not silently erase prior rolling measurements.
    measured_v=np.array([.06,.10,.13,.20])
    measured_mu=high+(low-high)*np.exp(-(measured_v/vc)**2)
    residual=np.r_[(vm-TARGET[:,2])[QUIET]/5/np.sqrt(QUIET.sum()),
        (vm-TARGET[:,2])[LAUNCH]/20/np.sqrt(LAUNCH.sum()),
        (vm-TARGET[:,2])[LATE]/20/np.sqrt(LATE.sum()),
        (rm-TARGET[:,3])/1.2/np.sqrt(len(rm)),
        (period-f.TARGET_PERIOD)/.025,max(0,over-1.5)/1.5,
        (measured_mu-.0049)/np.array([.0006,.0006,.0007,.0015])/2,
        max(0,high-low)/.0005]
    ok=t[-1]>=seconds-.01 and 'status=0' in r.stdout
    if not ok:residual[:]=1e3
    residual=np.nan_to_num(residual,nan=1e3,posinf=1e3,neginf=-1e3)
    rec=dict(key=key,x=list(map(float,x)),config=cfg,scale=scale,rolling=dict(mu_low=low,mu_high=high,vc_m_s=vc,omega_c_rad_s=vc/R,omega_c_code=vc/(R*O),measured_speeds_m_s=measured_v.tolist(),effective_mu=measured_mu.tolist()),
        residual=residual.tolist(),score=float(residual@residual),period_s=float(period),quiet_speed=float(vm[QUIET].mean()),startup_rmse=float(np.sqrt(np.mean((vm[~LATE]-TARGET[~LATE,2])**2))),radius_rmse=float(np.sqrt(np.mean((rm-TARGET[:,3])**2))),overshoot=over,model_v=vm.tolist(),model_r=rm.tolist(),status=r.stdout.strip(),wall_s=time.perf_counter()-started)
    meta.write_text(json.dumps(rec,indent=2));return rec

BOUNDS=([.3,120/f.BASE['mu'],.10,.05,.90,.3,.0040,0.,.218],
        [3.,2000/f.BASE['mu'],.70,2.5,1.15,1.8,.0065,.0065,.300])

def optimize(start,name,indices,max_nfev=24,measurement_prior=True):
    start=np.array(start);indices=np.array(indices);history=[];best=1e99
    def fun(z):
        nonlocal best
        x=start.copy();x[indices]=z;r=evaluate(x)
        if not measurement_prior:
            residual=np.r_[r['residual'][:-5],r['residual'][-1]]
            r['measurement_prior_score']=r['score'];r['score']=float(residual@residual)
            r['residual']=residual.tolist();r['measurement_prior']=False
        history.append(dict(key=r['key'],score=r['score']))
        if r['score']<best:
            best=r['score'];(OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
            print(name,len(history),'score',round(best,3),'quiet',round(r['quiet_speed'],2),'T',round(r['period_s'],4),'over',round(r['overshoot'],2),'rolling',r['rolling'],flush=True)
        return np.array(r['residual'])
    result=least_squares(fun,start[indices],bounds=(np.array(BOUNDS[0])[indices],np.array(BOUNDS[1])[indices]),diff_step=.004,x_scale='jac',max_nfev=max_nfev,ftol=.002,xtol=.0002,gtol=.01)
    (OUT/('optimization_'+name+'.json')).write_text(json.dumps(dict(history=history,nfev=result.nfev,message=result.message,
        initial=start.tolist(),indices=indices.tolist(),bounds=BOUNDS,measurement_prior=measurement_prior),indent=2))
    return json.loads((OUT/('best_'+name+'.json')).read_text())

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--measured' in sys.argv:
        x=json.loads((OUT/'best_wait_weak.json').read_text())['x']
        BOUNDS[0][6]=.0045;BOUNDS[1][6]=.0056;x[6]=np.clip(x[6],.0045,.0056)
        optimize(x,'measured_band',list(range(9)),40)
    elif '--free' in sys.argv:
        x=json.loads((OUT/'best_wait_stable.json').read_text())['x']
        BOUNDS[1][6]=.012
        optimize(x,'free_rolling',list(range(9)),45,measurement_prior=False)
    elif '--refine' in sys.argv:
        x=json.loads((OUT/'best_free_rolling.json').read_text())['x']
        BOUNDS[0][1]=60/f.BASE['mu'];BOUNDS[0][5]=.05;BOUNDS[1][6]=.012
        optimize(x,'free_material_refine',list(range(9)),25,measurement_prior=False)
    else:
        wait=seed('selected_wait_candidate.json');small=seed('selected_smallover_candidate.json')
        cases=[]
        for name,x in [('wait',wait),('small',small)]:
            for high in [.0049,.003,.001,0.]:
                y=x.copy();y[7]=high;cases.append((name+str(high),y))
        with ThreadPoolExecutor(max_workers=4) as pool:
            records=list(pool.map(lambda z:dict(tag=z[0],record=evaluate(z[1])),cases))
        (OUT/'profiles.json').write_text(json.dumps(records,indent=2))
        weak=wait.copy();weak[7]=.003;wait[7]=.0049
        jobs=[(wait,'wait_stable',list(range(9))),(weak,'wait_weak',list(range(9))),(small,'small',list(range(9)))]
        with ThreadPoolExecutor(max_workers=2) as pool:
            res=list(pool.map(lambda z:optimize(*z),jobs))
        (OUT/'selected.json').write_text(json.dumps(min(res,key=lambda r:r['score']),indent=2))
