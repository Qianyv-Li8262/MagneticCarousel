"""Fit magnet moment and linear rolling damping, with tilt/material fixed."""
import argparse,json
import numpy as np
from scipy.optimize import least_squares
from audit_mu_moment import run,OUT
ap=argparse.ArgumentParser();ap.add_argument('--mu',type=float,nargs='+',default=[.0003,.001,.003,.01]);a=ap.parse_args()
base=run(1e-4,3.04);T=base['metrics']['period_s'];A=base['metrics']['amplitude_percent']
fits=[]
for mu in a.mu:
    period_data=json.loads((OUT/'period_matched.json').read_text())
    prior=next((r for r in period_data['fits'] if r['mu_high']==mu and 'metrics' in r),None)
    initial=[prior['moment_scale'] if prior else 3.04,.004]
    if mu==.0003:initial=[3.04,.0033]
    elif mu==.001:initial=[3.04,.002]
    elif mu==.003:initial=[3.1,.001]
    elif mu==.01:initial=[4.0,.001]
    evaluated=[]
    def fun(x):
        r=run(mu,float(x[0]),lam=float(x[1]))
        evaluated.append(r)
        if r['error'] or r['metrics']['period_s'] is None:return [50,50]
        return [100*(r['metrics']['period_s']/T-1)/.5,(r['metrics']['amplitude_percent']-A)/.5]
    opt=least_squares(fun,initial,bounds=([.5,0],[8,.025]),x_scale=[3,.004],diff_step=[.002,.05],max_nfev=15,ftol=.002,xtol=1e-5,gtol=.02)
    valid=[r for r in evaluated if not r['error'] and r.get('stationary')]
    def score(r):return (100*(r['metrics']['period_s']/T-1)/.5)**2+((r['metrics']['amplitude_percent']-A)/.5)**2
    best=min(valid,key=score) if valid else evaluated[-1]
    best=dict(best);best['score']=score(best) if valid else None
    best['period_error_percent']=100*(best['metrics']['period_s']/T-1) if valid else None
    best['amplitude_error_pp']=best['metrics']['amplitude_percent']-A if valid else None
    best['matched']=bool(valid and abs(best['period_error_percent'])<.5 and abs(best['amplitude_error_pp'])<.5)
    best['optimizer_nfev']=opt.nfev
    fits.append(best)
    path=OUT/f'both_matched_mu{mu:g}.json';path.write_text(json.dumps({'baseline':base,'best':best,'evaluations':evaluated},indent=2),encoding='utf-8')
    print('BOTH_MATCH '+json.dumps({k:best.get(k) for k in ['mu_high','moment_scale','lambda','period_error_percent','amplitude_error_pp','matched']}),flush=True)
