"""Vary tilt and moment to match baseline period with 6-8% modulation."""
import json
import numpy as np
from audit_mu_moment import run, OUT, TILT
period=json.loads((OUT/'period_matched.json').read_text())
target_T=2.3962746873886855
target_A=7.
fits=[]
for prior in period['fits']:
    mu=prior['mu_high'];m=prior['moment_scale']
    tilt=TILT*target_A/prior['metrics']['amplitude_percent']
    evaluated=[]
    def evaluate(m,t):
        r=run(mu,float(m),tilt=float(t))
        evaluated.append(r)
        if r['error'] or r['metrics']['period_s'] is None:raise RuntimeError(str(r))
        return r,np.array([r['metrics']['period_s']/target_T-1,(r['metrics']['amplitude_percent']-target_A)/100])
    try:
        for iteration in range(5):
            r,f=evaluate(m,tilt)
            if abs(f[0])<.001 and abs(f[1])<.003:break
            dm=.003;dtilt=tilt*.04
            _,fm=evaluate(m+dm,tilt);_,ft=evaluate(m,tilt+dtilt)
            J=np.column_stack([(fm-f)/dm,(ft-f)/dtilt])
            step=np.linalg.lstsq(J,-f,rcond=None)[0]
            step[0]=np.clip(step[0],-.05,.05)
            step[1]=np.clip(step[1],-.25*tilt,.25*tilt)
            m=float(np.clip(m+step[0],.5,8));tilt=float(np.clip(tilt+step[1],.00005,.00345))
        valid=[r for r in evaluated if r.get('stationary')]
        def score(r):return ((r['metrics']['period_s']/target_T-1)/.001)**2+((r['metrics']['amplitude_percent']-target_A)/.3)**2
        best=min(valid,key=score)
        validated=run(mu,best['moment_scale'],seconds=600,dt=.00025,tilt=best['tilt_rad'],save=True)
        validated=dict(validated)
        met=validated['metrics']
        validated['period_error_percent']=100*(met['period_s']/target_T-1)
        validated['tilt_deg']=float(np.degrees(validated['tilt_rad']))
        validated['equivalent_laser_offset_at_930mm']=float(930*np.tan(2*validated['tilt_rad']))
        validated['within_original_laser_bounds']=bool(TILT<=validated['tilt_rad']<=.5*np.arctan(6/870))
        validated['moment_change_percent']=100*(validated['moment_scale']/3.04-1)
        validated['accepted']=bool(validated.get('stationary') and abs(validated['period_error_percent'])<.1 and 6<=met['amplitude_percent']<=8)
        fits.append(validated)
        print('TILT_MATCH '+json.dumps({k:validated[k] for k in ['mu_high','moment_scale','tilt_deg','period_error_percent','accepted']}),flush=True)
    except Exception as exc:fits.append({'mu_high':mu,'error':str(exc)})
    (OUT/'tilt_matched.json').write_text(json.dumps({'period_target_s':target_T,'amplitude_target_percent':target_A,'fits':fits},indent=2),encoding='utf-8')
