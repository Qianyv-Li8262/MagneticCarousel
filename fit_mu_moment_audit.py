"""Match reference period, then investigate amplitude with a second loss parameter."""
from pathlib import Path
import json
from scipy.optimize import brentq
from audit_mu_moment import run, OUT

base=run(1e-4,3.04)
target=base['metrics']['period_s']
results=[]
for mu in [0,2.5e-5,1e-4,3e-4,.001,.003,.01]:
    if mu==1e-4:
        fitted=base
    else:
        def residual(scale):
            r=run(mu,float(scale))
            if r['error'] or r['metrics']['period_s'] is None:
                raise RuntimeError(f'Cannot bracket mu={mu}, m={scale}: {r}')
            return r['metrics']['period_s']/target-1
        try:
            # Increasing drive may not be monotone near phase-locking transitions.
            points=[(3.04,residual(3.04))]
            direction=1 if points[0][1]>0 else -1
            for distance in [.1,.3,.6,1.,1.6,2.5,4.]:
                trial=max(.3,3.04+direction*distance)
                points.append((trial,residual(trial)))
                if points[-1][1]*points[0][1]<=0:break
            else:raise RuntimeError('No sign-changing bracket')
            lo,hi=sorted([points[0][0],points[-1][0]])
            value=brentq(residual,lo,hi,xtol=.0008,maxiter=18)
            fitted=run(mu,float(value))
        except RuntimeError as exc:
            results.append({'mu_high':mu,'error':str(exc)});continue
    entry=dict(fitted)
    entry['period_error_percent']=100*(fitted['metrics']['period_s']/target-1)
    entry['amplitude_error_pp']=fitted['metrics']['amplitude_percent']-base['metrics']['amplitude_percent']
    entry['moment_change_percent']=100*(fitted['moment_scale']/3.04-1)
    results.append(entry)
    (OUT/'period_matched.json').write_text(json.dumps({'baseline':base,'fits':results},indent=2),encoding='utf-8')
    print('MATCH '+json.dumps({k:entry[k] for k in ['mu_high','moment_scale','moment_change_percent','period_error_percent','amplitude_error_pp']}),flush=True)
