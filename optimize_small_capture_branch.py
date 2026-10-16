"""Keep the captured branch: penalize raw motion over the final TEN seconds."""
import json
import numpy as np
from scipy.optimize import differential_evolution
from search_prepared_release import OUT, simulate, load_hysteresis


def cost(r):
    if r['status']!='ok':return 1e6
    end=r['final_displacement_rad'];miss=max(0,end+1,-2-end)
    return (300*miss+80*r['max_slow_rebound_rad']+30*max(0,r['tail10_box_mm']-3)+
            8*r['tail_position_box_mm']+2000*max(0,r['initial_forward_rad']-.01))


def main():
    mod=load_hysteresis();p=mod.SolverParams(sr=5/6, somega=1)
    rows=[];best=[None]
    def objective(x):
        settings=[10**x[0],10**x[1],x[2],10**x[3],x[4],x[5]]
        r,arrays=simulate(mod,p,settings,x[6],seconds=40)
        if arrays is not None:
            _,t,pos,_,_=arrays
            r['tail10_box_mm']=float(np.linalg.norm(np.ptp(pos[t>30],axis=0)))
        r['capture_cost']=cost(r);rows.append(r)
        if r['status']=='ok' and (best[0] is None or cost(r)<cost(best[0])):best[0]=r
        if len(rows)%40==0:
            print(json.dumps(dict(evaluations=len(rows),best=best[0])),flush=True)
            (OUT/'small_capture_optimize.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        return cost(r)
    x0=[np.log10(548.2937661954036),np.log10(1.5659942198532072e-5),.22572257947882368,
        np.log10(.00024125303926396708),.017,.6,.6]
    differential_evolution(objective,[(np.log10(100),np.log10(3000)),(-5.5,-3.5),(.03,.6),
                (-5,-2.8),(.0035,np.deg2rad(1)),(-.1,1.),(.35,np.pi/4)],
                x0=x0,seed=17262,popsize=6,maxiter=18,polish=False,tol=.0001,workers=1)
    (OUT/'small_capture_optimize.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(dict(evaluations=len(rows),best=best[0],passes=sum(r.get('passes',False) for r in rows))),flush=True)


if __name__=='__main__':main()
