"""Keep the high-C magnetic branch; test previously authorized rolling parameters."""
import json,sys,threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
from scipy.optimize import least_squares
import high_c_wait_study as h
import fit_video56c6_rolling as d

MATERIAL_OUT=h.OUT
OUT=h.OUT/'rolling';OUT.mkdir(exist_ok=True);h.OUT=OUT;d.OUT=OUT
guard=threading.Lock();locks={}

def evaluate(x,tag=''):
    sigma,k,alpha,mu,c,low,high,vc=map(float,x)
    y=[sigma,mu,c,k,67.4/63.092177723551295,alpha,low,high,vc]
    rec=d.evaluate(y,seconds=240.)
    with guard:lock=locks.setdefault(rec['key'],threading.Lock())
    with lock:
        a=pd.read_csv(OUT/(rec['key']+'.csv'));t=a.time_s.to_numpy();tail=t>=220
        n=round(1/np.median(np.diff(t)));sm=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid');ts=np.convolve(t,np.ones(n)/n,mode='valid')
        tr=float(a.radius_mm[tail].mean()) if tail.any() else float('nan')
        native=dict(name=tag,key=rec['key'],config=rec['config'],rolling=rec['rolling'],status=rec['status'],end_s=float(t[-1]),
            tail_radius_mm=tr,overshoot_mm=float(max(0,sm.max()-tr)),peak_radius_mm=float(sm.max()),peak_time_s=float(ts[np.argmax(sm)]),
            tail_speed_mm_s=float(a.speed_mm_s[tail].mean()) if tail.any() else None,
            tail_radius_std_mm=float(a.radius_mm[tail].std()) if tail.any() else None,
            tail_wz_rad_s=float(a.wz_rad_s[tail].mean()) if tail.any() else None,
            tail_period_s=float(2*np.pi/abs(np.polyfit(t[tail],np.unwrap(a.phi_rad)[tail],1)[0])) if tail.sum()>10 else None,
            rolling_fraction=float(1-a.sliding.mean()))
        r=h.analyze(native)
    return r

def profiles():
    seed=min([json.loads(p.read_text()) for p in MATERIAL_OUT.glob('best_*.json')],key=lambda r:r['score'])
    c=seed['config'];base=[c['sigma']/1e7,c['K']/h.BASE['K'],c['alpha']/h.BASE['alpha'],c['mu']/h.BASE['mu'],c['c'],.0049,.0049,.26]
    cases=[]
    for low in [.0049,.0056,.0065]:
        for high in [.0049,.0035,.002]:
            x=base.copy();x[5:7]=[low,high];cases.append(x)
    for factor in [1.1,1.25]:
        x=base.copy();x[0]*=factor;x[5:7]=[.0056,.0035];cases.append(x)
    for vc in [.218,.3]:
        x=base.copy();x[5:]=[.0056,.0035,vc];cases.append(x)
    # Having restored a quiet interval by increasing low-speed rolling loss,
    # restore magnetic drive with sigma and counter the new overshoot with C.
    for c in [.46,.48,.50]:
        for sigma in [.7,.8,.9]:
            x=base.copy();x[0]=sigma;x[4]=c;x[5:7]=[.0056,.0035];cases.append(x)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda x:dict(x=x,record=evaluate(x,'rolling_profile')),cases))
    (OUT/'profiles.json').write_text(json.dumps(rows,indent=2))
    best=min(rows,key=lambda z:z['record']['score']);(OUT/'best_profile.json').write_text(json.dumps(best['record'],indent=2))
    print('BEST_PROFILE',best['record']['score'],best['x'],flush=True)

def optimize():
    rows=json.loads((OUT/'profiles.json').read_text());seed=min(rows,key=lambda z:z['record']['score'])['x'];seed=np.array(seed)
    ids=np.array([0,1,4,5,6,7]);bounds=([.2,.03,.42,.0045,.0005,.218],[3.,2.,.65,.008,.0065,.3]);best=1e99;history=[]
    def fun(z):
        nonlocal best
        x=seed.copy();x[ids]=z;r=evaluate(x,'rolling_joint');rr=np.r_[r['residual'],max(0,x[6]-x[5])/.001]
        score=float(rr@rr);history.append(dict(key=r['key'],x=x.tolist(),score=score))
        if score<best:
            best=score;(OUT/'best_joint.json').write_text(json.dumps(r,indent=2));print('BEST',len(history),best,'quiet',r['quiet_mean_mm_s'],'rise',r['rise20_80_s'],'over',r['overshoot_mm'],'v',r['tail_speed_mm_s'],flush=True)
        return rr
    try:
        result=least_squares(fun,seed[ids],bounds=bounds,diff_step=.008,x_scale='jac',max_nfev=14,ftol=.003,xtol=.0005,gtol=.01);message=result.message
    except KeyboardInterrupt:message='Stopped after inspecting diminishing improvements; best complete point retained.'
    (OUT/'optimization.json').write_text(json.dumps(dict(history=history,seed=seed.tolist(),indices=ids.tolist(),bounds=bounds,message=message),indent=2))

if __name__=='__main__':
    if '--optimize' in sys.argv:optimize()
    else:profiles()
