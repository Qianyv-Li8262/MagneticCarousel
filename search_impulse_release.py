"""Search the experimentally clarified release with an initial air impulse.

An impulse is represented by an initial center velocity, not a new damping law.
Spin fraction 0 is a translational kick; 1 is already pure rolling.
All geometry and mechanical constants remain unchanged.
"""
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution
from search_prepared_release import OUT, simulate, load_hysteresis


def cost(r):
    if r['status']!='ok':return 1e6
    end=r['final_displacement_rad']
    miss=max(0,end+1,-2-end)
    return (300*miss+100*r['initial_forward_rad']+40*r['max_slow_rebound_rad']+
            r['tail_position_box_mm']+max(0,(r['settle_within_1mm_s'] or 30)-8)*.2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['old','free','zero_old','zero_free','verify'],default='zero_old')
    args=p.parse_args()
    mod=load_hysteresis()
    params=mod.SolverParams(sr=5/6, somega=1)
    records=[];best=[None]
    old=[2292.6667,.0032,.38,.0010909091]
    if args.stage=='verify':
        pool=[]
        for name in ['impulse_zero_old','impulse_zero_free']:
            path=OUT/(name+'.json')
            if path.exists():pool+=json.loads(path.read_text(encoding='utf-8'))
        pool=sorted([r for r in pool if r['status']=='ok'],key=cost)
        selected=[]
        for r in pool:
            if all(np.linalg.norm(np.array(r['settings'])-np.array(b['settings']))>.01 or
                   abs(r['kick_mm_s']-b['kick_mm_s'])>5 for b in selected):selected.append(r)
            if len(selected)==3:break
        for i,r in enumerate(selected):
            for turns in [1,2,3]:
                for dt in [.001,.0005]:
                    row,_=simulate(mod,params,r['settings'],r['phase'],turns,seconds=60,
                        dt=dt,stride=20,save=True,tag=f'impulse_verify_{i}_{turns}_{dt}',
                        kick_mm_s=r['kick_mm_s'],roll_fraction=r['roll_fraction'])
                    row['fit_cost']=cost(row);records.append(row)
                    print(json.dumps(row),flush=True)
    else:
        def objective(x):
            if args.stage in ['old','zero_old']:
                settings=old+[10**x[0],x[1]];phase=x[2]
                kick=10**x[3] if args.stage=='old' else 0.;spin=x[4] if args.stage=='old' else 0.
            else:
                settings=[10**x[0],10**x[1],x[2],10**x[3],10**x[4],x[5]]
                phase=x[6];kick=10**x[7] if args.stage=='free' else 0.;spin=x[8] if args.stage=='free' else 0.
            row,_=simulate(mod,params,settings,phase,seconds=20,kick_mm_s=kick,roll_fraction=spin)
            row['fit_cost']=cost(row);records.append(row)
            if row['status']=='ok' and (best[0] is None or cost(row)<cost(best[0])):best[0]=row
            if len(records)%50==0:
                print(json.dumps(dict(evaluations=len(records),best=best[0])),flush=True)
                (OUT/f'impulse_{args.stage}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            return cost(row)
        if args.stage in ['old','zero_old']:
            bounds=[(-3,np.log10(.08)),(-np.pi,np.pi),(0,np.pi/4),(np.log10(2),np.log10(500)),(0,1)]
            x0=[np.log10(.02),.1,.59,np.log10(30),0]
            if args.stage=='zero_old':bounds=bounds[:3];x0=x0[:3]
        else:
            prior=[]
            for name in ['coarse','refine','optimize','wide_optimize','impulse_zero_old']:
                path=OUT/(name+'.json')
                if path.exists():prior+=json.loads(path.read_text(encoding='utf-8'))
            seed=min([r for r in prior if r['status']=='ok'],key=cost);s=seed['settings']
            bounds=[(np.log10(20),4),(-5,np.log10(.03)),(.025,.98),(-5,np.log10(.03)),
                    (-3,np.log10(.08)),(-np.pi,np.pi),(0,np.pi/4),(np.log10(2),np.log10(500)),(0,1)]
            x0=[np.log10(s[0]),np.log10(s[1]),s[2],np.log10(s[3]),np.log10(s[4]),s[5],
                seed['phase'],np.log10(max(2,seed.get('kick_mm_s',2))),seed.get('roll_fraction',0)]
            if args.stage=='zero_free':bounds=bounds[:7];x0=x0[:7]
            x0[0]=np.clip(x0[0],*bounds[0])
        differential_evolution(objective,bounds,x0=x0,seed=53262,popsize=8,
                               maxiter=28 if args.stage in ['old','zero_old'] else 30,
                               polish=False,tol=.0001,workers=1)
    (OUT/f'impulse_{args.stage}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    valid=sorted([r for r in records if r['status']=='ok'],key=cost)
    print(json.dumps(dict(completed=len(valid),failed=len(records)-len(valid),
                         passes=sum(r['passes'] for r in valid),best=valid[:3])),flush=True)


if __name__=='__main__':main()
