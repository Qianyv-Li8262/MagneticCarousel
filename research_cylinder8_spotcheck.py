"""Reproducible selected-point audit of the cylinder and exact point scans.

Uses separate processes for extensions with the same Python module name.
Never changes production builds or original scans.
"""
from pathlib import Path
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/cylinder8-spotcheck'
REFERENCE = (.33333333, 1.7212357e-4, 2.64965696, 7830.34224, 176182.704)


def classify(status, metrics):
    if status:
        return 'lut_exit' if status == 7 else f'error_{status}'
    if abs(metrics[0]-1) <= .05:
        return 'speed_one'
    return 'backward' if metrics[0] < 0 else 'other_forward'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    cyl = np.load(ROOT/'runs/phase-behavior/sr_somega_2d-8-CYL.npz')
    point = np.load(ROOT/'runs/phase-behavior/sr_somega_2d.npz')
    summary = {}
    for label, data in [('cylinder', cyl), ('point_archive', point)]:
        ok = data['status'] == 0
        m = data['metrics']
        rfinal = np.linalg.norm(data['final_state'][..., 1:3], axis=-1)
        summary[label] = dict(shape=list(ok.shape),
            status_counts={str(k):int(v) for k,v in zip(*np.unique(data['status'],return_counts=True))},
            time={k:float(data[k]) for k in ['dt','total_time','observation_window','sample_time']},
            backward=int(np.count_nonzero(ok & (m[...,0] < 0))),
            speed_one=int(np.count_nonzero(ok & (abs(m[...,0]-1)<=.05))),
            phase_locked=int(np.count_nonzero(ok & (abs(m[...,3])<.05))),
            mean_radius_range=[float(np.nanmin(m[...,19])),float(np.nanmax(m[...,19]))],
            radius_outside_common_01=int(np.count_nonzero(ok & (abs(m[...,19]-4)>.1))),
            radius_outside_common_08=int(np.count_nonzero(ok & (abs(m[...,19]-4)>.8))),
            failed_last_radius_range=[float(rfinal[~ok].min()),float(rfinal[~ok].max())] if np.any(~ok) else None,
            failed_last_radius_outward=int(np.count_nonzero((~ok)&(rfinal>5.9))),
            failed_last_radius_inward=int(np.count_nonzero((~ok)&(rfinal<2.1))))
    # Adjacent pixels across the secondary stripe and the principal escape band.
    candidates=[]
    for sr in (1., 2., 3., 5.):
        j=int(np.argmin(abs(cyl['sr']-sr)))
        for i in range(30,36):
            candidates.append((f'stripe_sr{sr:g}_row{i}',i,j))
    j=int(np.argmin(abs(cyl['sr']-3)))
    for i in (15,17,18,19,26,27,28,29):
        candidates.append((f'main_edge_row{i}',i,j))
    for sr,sw in ((.4,2.),(1.,2.),(2.5,2.),(2.8,2.),(2.5,4.),(2.8,4.),(3.,.3),(5.,.3),(5.,1.)):
        i=int(np.argmin(abs(cyl['somega']-sw))); j=int(np.argmin(abs(cyl['sr']-sr)))
        candidates.append((f'control_{sr:g}_{sw:g}',i,j))
    points=[]
    for label,i,j in candidates:
        # Reproduce CUDA's FP32 coordinate arithmetic, rather than assuming the
        # saved np.linspace axis has identical rounding at every pixel.
        sr=float(np.float32(.1)+np.float32(np.float32(6)-np.float32(.1))*np.float32(j/255))
        sw=float(np.float32(.1)+np.float32(np.float32(5)-np.float32(.1))*np.float32(i/255))
        points.append(dict(label=label,row=i,col=j,sr=sr,somega=sw,
            archived_status=int(cyl['status'][i,j]),
            archived_metrics=cyl['metrics'][i,j].astype(float).tolist(),
            archived_class=classify(int(cyl['status'][i,j]),cyl['metrics'][i,j])))
    (OUT/'points.json').write_text(json.dumps(points,indent=2,allow_nan=True),encoding='utf-8')
    (OUT/'archive_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def run(args):
    binary=next(args.module_dir.glob('*.pyd'))
    manifest=json.loads((args.module_dir/'build_config.json').read_text())
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==manifest['binary_sha256']
    if 'lut' in manifest:
        assert hashlib.sha256(Path(manifest['lut']).read_bytes()).hexdigest()==manifest['lut_sha256']
    spec=importlib.util.spec_from_file_location('hysteresis',binary)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dtype=np.dtype(module.real_dtype)
    zero=np.zeros(1,dtype=dtype)
    params=module.SolverParams(*REFERENCE)
    points=json.loads((OUT/'points.json').read_text())
    if args.select:
        points=[p for p in points if p['label'] in args.select]
    jobs=[]
    for p in points:
        for dt in args.dts:
            jobs.append((p,dt,args.horizon))
    def worker(job):
        p,dt,horizon=job
        actual_dt=float(dtype.type(dt))
        start=time.perf_counter()
        status,metrics,state=module.scan_initial_grid(zero,zero,actual_dt,
            round(horizon/dt),round(500/dt),round(.05/dt),p['sr'],p['somega'],params,1)
        status=int(status[0,0]); m=metrics[0,0].astype(float)
        s=state[0,0].astype(float); r=float(np.linalg.norm(s[1:3]))
        return dict(label=p['label'],sr=p['sr'],somega=p['somega'],dt=dt,
            actual_dt=actual_dt,horizon=horizon,status=status,category=classify(status,m),
            mean_v=m[0],std_v=m[1],phase=m[3],std_phase=m[4],slip=m[5],
            sliding_fraction=m[6],delta_v=m[12],delta_phase=m[13],mean_radius=m[19],
            last_radius=r,radial_v=float(np.dot(s[1:3],s[3:5])/r),
            metrics=m.tolist(),state=s.tolist(),elapsed=time.perf_counter()-start)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(worker,jobs):
            results.append(result)
            print(args.label,result['label'],result['dt'],result['status'],
                  round(result['mean_v'],6),round(result['mean_radius'],6),flush=True)
    payload=dict(manifest=manifest,binary=str(binary.resolve()),results=results)
    (OUT/f'{args.label}.json').write_text(json.dumps(payload,indent=2,allow_nan=True),encoding='utf-8')
    fields=[k for k in results[0] if k not in ('metrics','state')]
    with (OUT/f'{args.label}.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--module-dir',type=Path)
    parser.add_argument('--label')
    parser.add_argument('--dts',type=float,nargs='+',default=[.001,.0005,.00025])
    parser.add_argument('--horizon',type=float,default=1500.)
    parser.add_argument('--select',nargs='+')
    args=parser.parse_args()
    prepare() if args.prepare else run(args)
