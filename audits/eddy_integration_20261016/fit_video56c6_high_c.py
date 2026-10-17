"""High-C continuation against the actual 56c6 data; isolated snapshots only."""
import json,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import least_squares
import fit_video56c6_rolling as g

g.OUT=g.f.ROOT/'runs/video56c6-high-C-fit-20261017'
g.OUT.mkdir(exist_ok=True)
BOUNDS=np.array([[.3,80/g.f.BASE['mu'],.475,.035,.90,.05,.0045,.0005,.218],
                 [3.,1600/g.f.BASE['mu'],.90,2.5,1.15,1.8,.009,.0065,.300]])

def residual(r):
    v=np.array(r['model_v']);radius=np.array(r['model_r'])
    z=np.r_[(v-g.TARGET[:,2])[g.QUIET]/5/np.sqrt(g.QUIET.sum()),
        (v-g.TARGET[:,2])[g.LAUNCH]/18/np.sqrt(g.LAUNCH.sum()),
        (v-g.TARGET[:,2])[g.LATE]/20/np.sqrt(g.LATE.sum()),
        (radius-g.TARGET[:,3])/1.2/np.sqrt(len(radius)),
        (r['period_s']-g.f.TARGET_PERIOD)/.025,
        max(0,r['overshoot']-.8)/1.0,
        max(0,r['rolling']['mu_high']-r['rolling']['mu_low'])/.0005]
    if 'status=0' not in r['status']:z[:]=1000
    return np.nan_to_num(z,nan=1000,posinf=1000,neginf=-1000)

def scored(x):
    r=g.evaluate(x);r['fit_score']=float(np.sum(residual(r)**2));return r

def optimize(x,name,nfev=14,indices=None,overshoot_weight=1):
    x=np.array(x);indices=np.arange(9) if indices is None else np.array(indices)
    history=[];best=1e99
    def fun(z):
        nonlocal best
        y=x.copy();y[indices]=z;r=scored(y)
        residual_value=residual(r);residual_value[-2]*=overshoot_weight
        r['fit_score']=float(residual_value@residual_value)
        history.append([r['key'],r['fit_score']])
        if r['fit_score']<best:
            best=r['fit_score'];(g.OUT/('best_'+name+'.json')).write_text(json.dumps(r,indent=2))
            print(name,len(history),round(best,3),'C',round(r['config']['c'],5),
                  'quiet',round(r['quiet_speed'],2),'startup',round(r['startup_rmse'],2),
                  'T',round(r['period_s'],4),'over',round(r['overshoot'],2),flush=True)
        return residual_value
    result=least_squares(fun,x[indices],bounds=BOUNDS[:,indices],diff_step=.003,
        x_scale='jac',max_nfev=nfev,ftol=.004,xtol=.0003,gtol=.02)
    (g.OUT/('optimization_'+name+'.json')).write_text(json.dumps(dict(history=history,
        initial=x.tolist(),indices=indices.tolist(),bounds=BOUNDS.tolist(),
        message=result.message,nfev=result.nfev,overshoot_weight=overshoot_weight),indent=2))
    return json.loads((g.OUT/('best_'+name+'.json')).read_text())

if __name__=='__main__':
    if '--smallover' in sys.argv:
        x=json.loads((g.OUT/'selected.json').read_text())['x']
        optimize(x,'smallover_priority',10,[0,2,3,4,6,7,8],overshoot_weight=2)
        sys.exit()
    if '--fixed-high' in sys.argv:
        x=np.array(json.loads((g.OUT/'baseline.json').read_text())['x'])
        x[2]=.65;x[0]*=(1-.4750199416865859)/(1-.65);x[6]=.0075;x[7]=.0018
        r=optimize(x,'fixed_C065',12,[0,3,4,6,7,8])
        optimize(r['x'],'fixed_C065_material',10,[0,1,3,4,5,6,7,8])
        sys.exit()
    prev=json.loads((g.f.ROOT/'runs/high-C-wait-study-20261017/selected_joint.json').read_text())
    c=prev['config'];rr=prev['rolling']
    x=np.array([c['sigma']/1e7,c['mu']/g.f.BASE['mu'],c['c'],c['K']/g.f.BASE['K'],
        c['r']/63.092177723551295,c['alpha']/g.f.BASE['alpha'],rr['mu_low'],rr['mu_high'],rr['vc_m_s']])
    baseline=scored(x);(g.OUT/'baseline.json').write_text(json.dumps(baseline,indent=2))
    print('BASELINE',baseline['quiet_speed'],baseline['startup_rmse'],baseline['period_s'],baseline['fit_score'],flush=True)
    # Direct C profiles, followed by paired C/conductivity compensation; no random search.
    jobs=[]
    for C in [.50,.55,.60,.70,.80]:
        y=x.copy();y[2]=C;jobs.append(('C_only_'+str(C),y))
        y=y.copy();y[0]*=(1-x[2])/(1-C);jobs.append(('C_sigma_'+str(C),y))
    with ThreadPoolExecutor(max_workers=2) as pool:
        profiles=list(pool.map(lambda z:dict(name=z[0],record=scored(z[1])),jobs))
    (g.OUT/'profiles.json').write_text(json.dumps(profiles,indent=2))
    # Two physically motivated basins: latest small-overshoot branch and previous video fit.
    old=json.loads((g.f.ROOT/'runs/video56c6-rolling-fit-20261017/selected.json').read_text())
    y=np.array(old['x']);y[2]=.55
    starts=[(x,'from_smallover'),(y,'from_video')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda p:optimize(*p),starts))
    best=min(results,key=lambda r:r['fit_score'])
    (g.OUT/'selected.json').write_text(json.dumps(best,indent=2))
    print('SELECTED',best['key'],best['fit_score'],flush=True)
