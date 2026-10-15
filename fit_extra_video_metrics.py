"""Refit each video's basic targets before testing additional observables."""
import json
import numpy as np
from audit_mu_moment import run,OUT
from pathlib import Path
root=Path(__file__).resolve().parent.parent/'cv/new_video_review'
videos=json.loads((root/'extra_observables/video_metrics.json').read_text())
targets=json.loads((root/'disk_rpm/ball_fit_targets.json').read_text())
priors=json.loads((OUT/'tilt_matched.json').read_text())['fits']
results=[]
for video,target in zip(videos,targets):
    for mu in [.0001,.001,.01]:
        prior=next(r for r in priors if r['mu_high']==mu)
        m=prior['moment_scale'];tilt=prior['tilt_rad'];rpm=target['disk_rpm']
        T=target['ball_period_target_s'];A=video['methods']['affine_template']['harmonics_percent'][0]
        tilt*=A/prior['metrics']['amplitude_percent']
        evaluated=[]
        def evaluate(m,t):
            r=run(mu,float(m),tilt=float(t),rpm=rpm)
            evaluated.append(r)
            if r['error'] or r['metrics']['period_s'] is None:raise RuntimeError(str(r))
            return r,np.array([r['metrics']['period_s']/T-1,(r['metrics']['amplitude_percent']-A)/100])
        try:
            for iteration in range(5):
                r,f=evaluate(m,tilt)
                if abs(f[0])<.002 and abs(f[1])<.004:break
                dm=.003;dt=tilt*.05
                _,fm=evaluate(m+dm,tilt);_,ft=evaluate(m,tilt+dt)
                J=np.column_stack([(fm-f)/dm,(ft-f)/dt])
                step=np.linalg.lstsq(J,-f,rcond=None)[0]
                m=float(np.clip(m+np.clip(step[0],-.1,.1),.5,8))
                tilt=float(np.clip(tilt+np.clip(step[1],-.3*tilt,.3*tilt),.00005,.00345))
            good=[r for r in evaluated if r.get('stationary')]
            best=min(good,key=lambda r:((r['metrics']['period_s']/T-1)/.002)**2+((r['metrics']['amplitude_percent']-A)/.4)**2)
            final=run(mu,best['moment_scale'],tilt=best['tilt_rad'],rpm=rpm,seconds=600,dt=.00025,sample_hz=video['methods']['raw_template']['fps'],save=True)
            results.append({'video':video['video'],'target_period_s':T,'target_amplitude_percent':A,'fit':final})
            print('VIDEO_FIT '+json.dumps({'video':video['video'],'mu':mu,'m':best['moment_scale'],'tilt':best['tilt_rad'],'period':final.get('metrics',{}).get('period_s')}),flush=True)
        except Exception as exc:results.append({'video':video['video'],'mu':mu,'error':str(exc)})
        (root/'extra_observables/simulation_fits.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
