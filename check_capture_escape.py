"""Zero-speed release, then a translational impulse after capture.

All state components, including attitude, spin and q, are carried into the kick.
The external impulse is an instantaneous velocity increment along -e_phi.
"""
import json
import numpy as np
from search_prepared_release import OUT, simulate, load_hysteresis
from search_impulse_release import cost


def main():
    mod=load_hysteresis();params=mod.SolverParams(sr=5/6, somega=1)
    pool=[]
    for name in ['coarse','refine','optimize','wide_optimize','impulse_zero_free']:
        path=OUT/(name+'.json')
        if path.exists():pool+=json.loads(path.read_text(encoding='utf-8'))
    target=[r for r in pool if r['status']=='ok' and -2<=r['final_displacement_rad']<=-1]
    target.sort(key=cost)
    selected=[]
    for r in target:
        if all(abs(r['settings'][4]-b['settings'][4])>.005 or
               abs(np.log(r['settings'][0]/b['settings'][0]))>.5 for b in selected):selected.append(r)
        if len(selected)==3:break
    # Also keep an original C_JA=0.38 case, rather than silently replacing it.
    original=[r for r in target if abs(r['settings'][2]-.38)<1e-8 and abs(r['settings'][0]-2292.6667)<1]
    if original:selected.append(min(original,key=cost))
    records=[]
    for i,r in enumerate(selected):
        pre,arrays=simulate(mod,params,r['settings'],r['phase'],turns=2,seconds=35,
                            dt=.0005,stride=20,save=True,tag=f'capture_{i}')
        records.append(dict(kind='capture',candidate=i,**pre))
        if arrays is None:continue
        captured=arrays[0][-1,1:16].copy()
        for kick in [2.,5.,10.,20.,40.,80.,160.,320.]:
            state=captured.copy();p=state[1:3];tangent=np.array([-p[1],p[0]])/np.linalg.norm(p)
            state[3:5]-=kick/1000*(4/.07)/(100*2*np.pi/60)*tangent
            row,arrays2=simulate(mod,params,r['settings'],state[0],turns=0,seconds=50,
                          dt=.0005,stride=40,state_override=state,
                          save=True,tag=f'escape_{i}_{kick:g}',kick_mm_s=kick)
            if arrays2 is not None:
                table,times,xy,angle,smooth=arrays2
                tail=times>40
                rolling_fraction=float(np.mean(table[tail,-1]==0))
                net_cycles=float(angle[-1]/(2*np.pi))
                tail_rates=np.gradient(smooth,times)[tail]
                row.update(rolling_fraction_last10s=rolling_fraction,
                           ball_net_revolutions=net_cycles,
                           mean_reverse_rad_s_last10s=float(-np.mean(tail_rates)),
                           max_forward_rad_s_last10s=float(np.max(tail_rates)),
                           stable_reverse=bool(np.mean(tail_rates)<-.2 and
                                np.max(tail_rates)<.05 and rolling_fraction>.95))
            records.append(dict(kind='escape',candidate=i,**row))
        (OUT/'capture_escape.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps(dict(candidate=i,capture=pre,
             escapes=[x for x in records if x['candidate']==i and x['kind']=='escape'])),flush=True)


if __name__=='__main__':main()
