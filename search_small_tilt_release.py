"""Search zero-speed release with theta <= 1 degree; no geometry changes."""
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution
from search_prepared_release import OUT,simulate,load_hysteresis,scaled_model_parameters

NAME='small_tilt'


def cost(r):
    if r['status']!='ok':return 1e6
    end=r['final_displacement_rad'];miss=max(0,end+1,-2-end)
    return (300*miss+200*r['initial_forward_rad']+2000*max(0,r['initial_forward_rad']-.01)+45*r['max_slow_rebound_rad']+
            2*r['tail_position_box_mm'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['c38','free','radius','verify'],default='c38')
    args=p.parse_args()
    mod=load_hysteresis();params=mod.SolverParams(*scaled_model_parameters(8,sr=5/6,somega=1))
    rows=[];best=[None]
    if args.stage=='verify':
        pool=[]
        for name in ['c38','free','radius']:
            path=OUT/f'{NAME}_{name}.json'
            if path.exists():pool+=json.loads(path.read_text(encoding='utf-8'))
        pool.sort(key=cost);selected=[]
        for r in pool:
            if r['status']!='ok':continue
            if all(abs(r['settings'][2]-s['settings'][2])>.05 or
                   abs(np.log(r['settings'][0]/s['settings'][0]))>.5 or
                   abs(r['settings'][4]-s['settings'][4])>.002 for s in selected):selected.append(r)
            if len(selected)==4:break
        for i,r in enumerate(selected):
            for turns in [1,2,3]:
                for dt in [.001,.0005]:
                    record,_=simulate(mod,params,r['settings'],r['phase'],turns=turns,seconds=45,
                         dt=dt,stride=20,save=True,tag=f'{NAME}_verify_{i}_{turns}_{dt}',
                         radius_mm=r.get('release_radius_mm',70.))
                    record['fit_cost']=cost(record);record['candidate']=i;rows.append(record)
            print(json.dumps(dict(candidate=i,results=rows[-6:])),flush=True)
    else:
        # Normalized JA constants; no mechanical coefficients are adjustable here.
        if args.stage=='c38':
            bounds=[(np.log10(20),4),(-5,np.log10(.03)),(-5,np.log10(.03)),
                    (np.log10(np.deg2rad(.1)),np.log10(np.deg2rad(1))),(-np.pi,np.pi),(0,np.pi/4)]
            x0=[np.log10(2292.6667),np.log10(.0032),np.log10(.0010909091),np.log10(np.deg2rad(.5)),.1,.5]
        else:
            prior=json.loads((OUT/f'{NAME}_{"free" if args.stage=="radius" else "c38"}.json').read_text(encoding='utf-8'))
            seed=min(prior,key=cost);s=seed['settings']
            bounds=[(np.log10(20),4),(-5,np.log10(.03)),(.025,.98),(-5,np.log10(.03)),
                    (np.log10(np.deg2rad(.05)),np.log10(np.deg2rad(1))),(-np.pi,np.pi),(0,np.pi/4)]
            x0=[np.log10(s[0]),np.log10(s[1]),s[2],np.log10(s[3]),np.log10(s[4]),s[5],seed['phase']]
            if args.stage=='radius':bounds.append((65.,75.));x0.append(70.)
        def objective(x):
            if args.stage=='c38':settings=[10**x[0],10**x[1],.38,10**x[2],10**x[3],x[4]];phase=x[5]
            else:settings=[10**x[0],10**x[1],x[2],10**x[3],10**x[4],x[5]];phase=x[6]
            r,_=simulate(mod,params,settings,phase,seconds=25 if args.stage=='c38' else 40,
                         radius_mm=x[7] if args.stage=='radius' else 70.)
            r['fit_cost']=cost(r);rows.append(r)
            if r['status']=='ok' and (best[0] is None or cost(r)<cost(best[0])):best[0]=r
            if len(rows)%50==0:
                print(json.dumps(dict(evaluations=len(rows),best=best[0])),flush=True)
                (OUT/f'{NAME}_{args.stage}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            return cost(r)
        differential_evolution(objective,bounds,x0=x0,seed=95262,popsize=8,
             maxiter=18 if args.stage=='radius' else 24,polish=False,tol=.0001,workers=1)
    (OUT/f'{NAME}_{args.stage}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    valid=sorted([r for r in rows if r['status']=='ok'],key=cost)
    print(json.dumps(dict(completed=len(valid),failed=len(rows)-len(valid),
                         passes=sum(r['passes'] for r in valid),best=valid[:3])),flush=True)


if __name__=='__main__':main()
