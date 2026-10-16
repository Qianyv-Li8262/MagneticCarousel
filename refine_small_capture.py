"""Follow the small-tilt captured branch, rather than scoring a turning point."""
import json
from search_prepared_release import OUT, simulate, load_hysteresis
from search_small_tilt_release import cost


def main():
    seed=json.loads((OUT/'small_stop_candidate.json').read_text(encoding='utf-8'))
    mod=load_hysteresis();p=mod.SolverParams(sr=5/6, somega=1)
    rows=[]
    for c in [seed['settings'][2],.38]:
        for tilt in [.006,.010,.014,.017]:
            for axis in [0.,.3,.6,.9]:
                for phase in [.2,.4,.6]:
                    settings=seed['settings'][:4]+[tilt,axis];settings[2]=c
                    r,_=simulate(mod,p,settings,phase,seconds=40)
                    r['fit_cost']=cost(r);rows.append(r)
        print(json.dumps(dict(cja=c,best=sorted([r for r in rows if r['status']=='ok'],key=cost)[:2])),flush=True)
        (OUT/'small_capture_refine.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
