"""Long-run validation at measured RPM; fixed normalized loss parameters."""
from pathlib import Path
import argparse, json, subprocess, sys
ap=argparse.ArgumentParser();ap.add_argument('--video',required=True);a=ap.parse_args()
root=Path(__file__).resolve().parent
target=next(t for t in json.loads((root.parent/'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text()) if t['video']==a.video)
out=root/'runs/laser-tilt-20261009/high_speed_fit_validation';out.mkdir(exist_ok=True)
records=[]
for mu,lam in [(.00005,.00575),(0,.006),(.0014,0)]:
    tag=f'_highfit_val_{a.video}_h{mu:g}_l{lam:g}'
    args=[sys.executable,str(root/'laser_tilt_experiment.py'),'--rpm',str(target['disk_rpm']),'--rolling','--v0=-.075','--force-mn','0','--push-seconds','0','--free-seconds','600','--angles','0.002777749200346311','--mu-low','.00001','--mu-high',str(mu),'--rolling-lambda',str(lam),'--tag',tag]
    with (out/f'{tag}.log').open('w',encoding='utf-8') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    folder=root/'runs/laser-tilt-20261009'/f"rpm{target['disk_rpm']:g}_diam12_dt0.0005{tag}"
    record=json.loads((folder/'summary.json').read_text())[0]
    record.update(video=a.video,target_period_s=target['ball_period_target_s'],run_folder=str(folder))
    m=record.get('metrics',{})
    if m.get('mean_orbit_period_s'): record['period_error_percent']=100*(m['mean_orbit_period_s']/target['ball_period_target_s']-1)
    records.append(record)
    (out/f'{a.video}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'video':a.video,'mu':mu,'lambda':lam,'status':record['status'],'steady':m.get('steady_free_orbit'),'period':m.get('mean_orbit_period_s'),'radius':m.get('radius_mean_mm')}),flush=True)
