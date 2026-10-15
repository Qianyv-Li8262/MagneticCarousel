"""Screen passive high-speed rolling loss against measured full-lap periods."""
from pathlib import Path
import argparse,json,subprocess,sys
parser=argparse.ArgumentParser();parser.add_argument('--boundary',action='store_true');parser.add_argument('--fine',action='store_true');options=parser.parse_args()
root=Path(__file__).resolve().parent
out=root/'runs/laser-tilt-20261009'/('high_speed_fit_fine' if options.fine else 'high_speed_fit_boundary' if options.boundary else 'high_speed_fit');out.mkdir(parents=True,exist_ok=True)
targets=json.loads((root.parent/'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())
cases=[(mu,.006) for mu in [0,.000025,.00005,.0001,.0002]]
cases += [(mu,lam) for lam in [0,.001,.003,.005] for mu in [.0001,.0003,.001,.003]]
if options.boundary:
    cases=[(mu,lam) for lam in [.005,.00525,.0055,.00575] for mu in [0,.00005]]
    cases += [(mu,0) for mu in [.0012,.0015,.0018,.0022]]
if options.fine:
    cases=[(.00005,.0053),(.00005,.0054),(.00005,.00545),(0,.00555),(0,.0056),(.0013,0),(.0014,0),(.001,.0012),(.001,.0013)]
results=[]
for mu,lam in cases:
    tag=f'_highfit{"f" if options.fine else "b" if options.boundary else ""}_h{mu:g}_l{lam:g}'
    folder=root/'runs/laser-tilt-20261009'/f'rpm200_diam12_dt0.0005{tag}'
    args=[sys.executable,str(root/'laser_tilt_experiment.py'),'--rpm','200','--rolling','--v0','-.075','--force-mn','0','--push-seconds','0','--free-seconds','360','--angles','0.002777749200346311','--mu-low','0.00001','--mu-high',str(mu),'--rolling-lambda',str(lam),'--tag',tag]
    with (out/f'{tag}.log').open('w',encoding='utf-8') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((folder/'summary.json').read_text())[0]
    m=result.get('metrics',{})
    entry={'mu_low':.00001,'mu_high':mu,'lambda':lam,'rpm':200,'v0':-.075,'status':result['status'],'steady':m.get('steady_free_orbit',False),'period_s':m.get('mean_orbit_period_s'),'radius_mm':m.get('radius_mean_mm'),'speed_mm_s':m.get('mean_tangential_speed_mm_s'),'amplitude_percent':m.get('first_harmonic_relative_half_amplitude_percent'),'last5_radius_span_mm':m.get('last5_lap_mean_radius_span_mm'),'run_folder':str(folder)}
    if entry['period_s'] is not None:
        entry['period_errors_percent']=[100*(entry['period_s']/t['ball_period_target_s']-1) for t in targets]
    results.append(entry)
    (out/'screening.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(entry),flush=True)
