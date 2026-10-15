"""Scan guessed magnet moment in isolation, preserving steel and geometry."""
from pathlib import Path
import argparse,json,subprocess,sys
ap=argparse.ArgumentParser();ap.add_argument('--mu-high',type=float,default=0);ap.add_argument('--scales',type=float,nargs='+',default=[.25,.5,.75,1,1.25,1.5,2,3,4]);ap.add_argument('--lambda-value',type=float,default=.006);ap.add_argument('--rpm',type=float,default=200);ap.add_argument('--free',type=float,default=600);ap.add_argument('--tag',default='');ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--tilt',type=float,default=.002777749200346311);a=ap.parse_args()
root=Path(__file__).resolve().parent;base=root/'runs/laser-tilt-20261009'
out=base/f'magnet_moment_scan_h{a.mu_high:g}_l{a.lambda_value:g}{a.tag}';out.mkdir(exist_ok=True)
results=[]
for scale in a.scales:
    tag=f'_moment{scale:g}_h{a.mu_high:g}_l{a.lambda_value:g}{a.tag}'
    args=[sys.executable,str(root/'laser_tilt_experiment.py'),'--rpm',str(a.rpm),'--dt',str(a.dt),'--rolling','--v0=-.075','--force-mn','0','--push-seconds','0','--free-seconds',str(a.free),'--angles',str(a.tilt),'--mu-low','.00001','--mu-high',str(a.mu_high),'--rolling-lambda',str(a.lambda_value),'--moment-scale',str(scale),'--tag',tag]
    with (out/f'{tag}.log').open('w',encoding='utf-8') as log: subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    folder=base/f'rpm{a.rpm:g}_diam12_dt{a.dt:g}{tag}'
    r=json.loads((folder/'summary.json').read_text())[0];r['run_folder']=str(folder);results.append(r)
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    m=r.get('metrics',{})
    print(json.dumps({'moment_scale':scale,'tilt':a.tilt,'mu_high':a.mu_high,'lambda':a.lambda_value,'status':r['status'],'steady':m.get('steady_free_orbit'),'period_s':m.get('mean_orbit_period_s'),'radius_mm':m.get('radius_mean_mm'),'amplitude_percent':m.get('first_harmonic_relative_half_amplitude_percent')}),flush=True)
