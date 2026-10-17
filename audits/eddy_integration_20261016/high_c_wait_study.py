"""High-C behavior study: quiet waiting, abrupt acceleration, small radial overshoot.

No experimental trajectory objective, time shifting or premagnetization is used.
"""
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import overshoot_gallery as gallery

OUT=gallery.f.ROOT/'runs/high-C-wait-study-20261017';OUT.mkdir(exist_ok=True)
gallery.OUT=OUT
BASE=gallery.f.BASE
_guard=threading.Lock();_locks={}

def analyze(native):
    dest=OUT/(native['key']+'.features.json')
    if dest.exists():return json.loads(dest.read_text())
    a=pd.read_csv(OUT/(native['key']+'.csv'));t=a.time_s.to_numpy()
    n=round(1/np.median(np.diff(t)));kernel=np.ones(n)/n
    sm=lambda col:np.convolve(a[col],kernel,mode='valid')
    ts=np.convolve(t,kernel,mode='valid');v=sm('speed_mm_s');radius=sm('radius_mm');wz=sm('wz_rad_s')
    final=native['tail_speed_mm_s'] or 0
    def crossing(level):
        # Ignore the initial magnetic release impulse; require two seconds above threshold.
        ix=np.flatnonzero((ts>=3)&(v>=level))
        m=max(1,round(2/np.median(np.diff(ts))))
        for j in ix:
            if j+m<len(v) and np.min(v[j:j+m])>=level:return float(ts[j])
        return None
    t20=crossing(.2*final);t80=crossing(.8*final)
    quiet=(ts>=4)&(ts<=13);qmean=float(v[quiet].mean()) if quiet.any() else 1e3
    q95=float(np.percentile(v[quiet],95)) if quiet.any() else 1e3
    rise=t80-t20 if t20 is not None and t80 is not None else 1000
    before=(ts>=3)&(ts<=t20) if t20 is not None else ts<0
    gain=float(wz[before][-1]-wz[before][0]) if before.sum()>2 else None
    residual=np.array([max(0,qmean-8)/5,max(0,q95-18)/10,
        (max(0,6-(t20 or 0))+max(0,(t20 or 1000)-25))/5,
        max(0,rise-7)/3,max(0,native['overshoot_mm']-.8)/.5,
        (final-300)/40,(native['tail_radius_std_mm'] or 0)/.1])
    ok=native['end_s']>239.9 and 'status=0' in native['status'] and final>100
    if not ok:residual[:]=1000
    residual=np.nan_to_num(residual,nan=1000,posinf=1000,neginf=-1000)
    r=dict(native,quiet_mean_mm_s=qmean,quiet_p95_mm_s=q95,t20_s=t20,t80_s=t80,rise20_80_s=rise,
        spin_gain_3s_to_t20_rad_s=gain,pre_onset_spin_peak_rad_s=float(np.max(wz[before])) if before.any() else None,
        residual=residual.tolist(),score=float(residual@residual),valid_behavior_run=ok)
    dest.write_text(json.dumps(r,indent=2));return r

def evaluate(x,tag=''):
    # sigma/1e7, K/Kref, alpha/alpharef, mu/muref, C_JA.
    sigma,k,alpha,mu,c=map(float,x)
    cfg=dict(sigma=sigma*1e7,K=k*BASE['K'],alpha=alpha*BASE['alpha'],mu=mu*BASE['mu'],c=c)
    token=json.dumps(cfg,sort_keys=True)
    with _guard:lock=_locks.setdefault(token,threading.Lock())
    with lock:r=analyze(gallery.run((tag,cfg)))
    return r

def profile():
    cases=[]
    for c in [.44,.46]:
        for sigma in [.3,.45,.6,.75,.9,1.05,1.2,1.5,2.]:
            cases.append((f'sigma_C{c}_{sigma}',[sigma,.2,1,1,c]))
    for sigma in [.6,1.2]:
        for k in [.05,.1,.4,.8]:cases.append((f'K_s{sigma}_{k}',[sigma,k,1,1,.44]))
        for alpha in [.25,.5,2.]:cases.append((f'alpha_s{sigma}_{alpha}',[sigma,.2,alpha,1,.44]))
        for mu in [.5,2.]:cases.append((f'mu_s{sigma}_{mu}',[sigma,.2,1,mu,.44]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda z:dict(tag=z[0],x=z[1],record=evaluate(z[1],z[0])),cases))
    (OUT/'profiles.json').write_text(json.dumps(rows,indent=2))
    for z in sorted(rows,key=lambda z:z['record']['score'])[:10]:
        r=z['record'];print('RANK',z['tag'],r['score'],r['quiet_mean_mm_s'],r['rise20_80_s'],r['overshoot_mm'],r['tail_speed_mm_s'],flush=True)
    return rows

BOUNDS=([.2,.03,.08,80/BASE['mu'],.42],[3.,2.,2.5,1800/BASE['mu'],.65])

def coordinate_search():
    saved=[json.loads(p.read_text()) for p in OUT.glob('best_branch*.json')]
    best=min(saved,key=lambda r:r['score']);history=[];scale=1.
    for iteration in range(4):
        c=best['config'];x=np.array([c['sigma']/1e7,c['K']/BASE['K'],c['alpha']/BASE['alpha'],c['mu']/BASE['mu'],c['c']])
        trials=[]
        for j,step in enumerate([.02,.04,.03,.02,.003]):
            for sign in [-1,1]:
                y=x.copy();y[j]=y[j]+sign*step*scale if j==4 else y[j]*(1+sign*step*scale)
                trials.append(y)
        for sign in [-1,1]:
            # Nearly preserve eddy relaxation scale sigma*mu while changing mu.
            y=x.copy();factor=1+sign*.03*scale;y[0]*=factor;y[3]/=factor;trials.append(y)
            # Preserve alpha/K while changing magnetic coupling strength.
            y=x.copy();factor=1+sign*.04*scale;y[1]*=factor;y[2]*=factor;trials.append(y)
        trials=[np.clip(y,BOUNDS[0],BOUNDS[1]) for y in trials]
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda y:evaluate(y,'coordinate'),trials))
        candidate=min(rows,key=lambda r:r['score'])
        history.append(dict(iteration=iteration,step_scale=scale,candidates=[dict(key=r['key'],score=r['score']) for r in rows]))
        if candidate['score']<best['score']:best=candidate
        else:scale*=.5
        (OUT/'best_coordinate.json').write_text(json.dumps(best,indent=2))
        print('COORD',iteration,best['score'],best['quiet_mean_mm_s'],best['rise20_80_s'],best['overshoot_mm'],flush=True)
    (OUT/'coordinate_history.json').write_text(json.dumps(history,indent=2))
    return best

def optimize(seed,name,max_nfev=20):
    best=1e99;history=[]
    def fun(x):
        nonlocal best
        r=evaluate(x,name);history.append(dict(x=list(map(float,x)),key=r['key'],score=r['score']))
        if r['score']<best:
            best=r['score'];(OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
            print('BEST',name,len(history),round(best,3),'quiet',round(r['quiet_mean_mm_s'],2),'rise',round(r['rise20_80_s'],2),'over',round(r['overshoot_mm'],3),'v',round(r['tail_speed_mm_s'] or 0,1),'x',x,flush=True)
        return r['residual']
    try:
        result=least_squares(fun,seed,bounds=BOUNDS,diff_step=.008,x_scale='jac',max_nfev=max_nfev,ftol=.003,xtol=.0005,gtol=.01)
        message=result.message
    except KeyboardInterrupt:message='Stopped after inspecting diminishing improvements; best completed candidate retained.'
    (OUT/('optimization_'+name+'.json')).write_text(json.dumps(dict(history=history,bounds=BOUNDS,message=message),indent=2))
    return json.loads((OUT/('best_'+name+'.json')).read_text())

if __name__=='__main__':
    if '--coordinate' in sys.argv:coordinate_search()
    elif '--optimize' in sys.argv:
        rows=json.loads((OUT/'profiles.json').read_text());rows=sorted(rows,key=lambda z:z['record']['score'])
        # Choose seeds with distinct mechanisms rather than a random parameter cloud.
        first=rows[0]
        quiet=[z for z in rows if z['record']['valid_behavior_run'] and z['record']['quiet_mean_mm_s']<15 and z['x'][0]!=first['x'][0]]
        second=min(quiet or [z for z in rows if z['x'][0]!=first['x'][0]],key=lambda z:z['record']['score'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda z:optimize(z[1]['x'],z[0]),[('branch1',first),('branch2',second)]))
    else:profile()
