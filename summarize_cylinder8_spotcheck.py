"""Trace and summarize the selected-point audit; export scientific figures."""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'runs/cylinder8-spotcheck'
FIG=ROOT/'scratch/cylinder8-spotcheck'
FIG.mkdir(parents=True,exist_ok=True)
points={p['label']:p for p in json.loads((DATA/'points.json').read_text())}
jobs=[]
for label in ('stripe_sr2_row32','stripe_sr3_row32','stripe_sr3_row33','stripe_sr5_row33','main_edge_row19'):
    for dt in (.001,.0005,.00025,.000125,.0000625):
        jobs.append(('cylinder',label,dt))
for label in ('main_edge_row19','main_edge_row28','stripe_sr3_row31','stripe_sr3_row32'):
    jobs.append(('point',label,.00025))
    jobs.append(('point_lut',label,.00025))
for label in ('main_edge_row28','stripe_sr3_row31','main_edge_row18'):
    jobs.append(('cylinder',label,.00025))
def trace(job):
    model,label,dt=job;p=points[label]
    exe=ROOT/('build/research-point-exact8/research_point8_trace.exe' if model=='point' else 'build/release/research_cylinder8_trace.exe')
    path=DATA/f'trace_{model}_{label}_{dt:g}.csv'
    result=subprocess.run([str(exe),str(p['sr']),str(p['somega']),str(dt),'1500',
        str(ROOT/('build/release/field-lut/field_lut_point_fp64.bin' if model=='point_lut' else 'build/release/field-lut/field_lut_fp64.bin')),str(path)],capture_output=True,text=True,check=True)
    a=np.genfromtxt(path,delimiter=',',names=True)
    terminal=a[(a['time']>=1000)&(a['status']==0)]
    return dict(model=model,label=label,dt=dt,log=result.stdout.strip(),
        last_t=float(a['time'][-1]),status=int(a['status'][-1]),
        max_radius=float(a['radius'].max()),last_radius=float(a['radius'][-1]),
        last_radial_v=float(a['radial_v'][-1]),minimum_normal_sampled=float(a['normal_force'].min()),
        first_crossing6_sampled=float(a['time'][a['radius']>6][0]) if np.any(a['radius']>6) else None,
        terminal_mean_orbital_speed=float(np.mean(4*terminal['v']/terminal['radius'])) if len(terminal) else None)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    traces=list(pool.map(trace,jobs))
(DATA/'trace_summary.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
for t in traces:print(t,flush=True)

raw_cyl=np.load(ROOT/'runs/phase-behavior/sr_somega_2d-8-CYL.npz')
raw_point=np.load(ROOT/'runs/phase-behavior/sr_somega_2d.npz')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
cmap=ListedColormap(['#d94b42','#3365bc','#36a566','#777777'])
for ax,data,name in zip(axes[0],[raw_point,raw_cyl],['Point dipoles: original K=128','Finite cylinders + LUT: K=256']):
    m=data['metrics'];cat=np.where(abs(m[...,0]-1)<=.05,0,np.where(m[...,0]<0,1,2))
    cat=np.where(data['status']==0,cat,3)
    ax.pcolormesh(data['sr'],data['somega'],cat,cmap=cmap,vmin=0,vmax=3,shading='nearest')
    ax.set(xlabel='$s_r$',ylabel='$s_\omega$',title=name,ylim=(.1,1.15))
axes[0,1].scatter([p['sr'] for p in points.values()],[p['somega'] for p in points.values()],s=13,facecolors='none',edgecolors='k',linewidths=.6)
axes[0,0].text(.02,.97,'Red: speed near 1 | Blue: negative speed\nGreen: other speed | Gray: LUT exit',transform=axes[0,0].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
for dt in (.001,.0005,.00025,.000125,.0000625):
    a=np.genfromtxt(DATA/f'trace_cylinder_stripe_sr3_row32_{dt:g}.csv',delimiter=',',names=True)
    axes[1,0].plot(a['time'],a['radius'],label=f'dt={dt:g}',lw=1)
axes[1,0].axhline(6,color='k',ls='--',lw=.8)
axes[1,0].set(xlabel='Time',ylabel='Radius',title='Sensitive stripe: sr=2.992157, somega=0.714902')
axes[1,0].legend(fontsize=8)
for model,label in [('point','main_edge_row19'),('cylinder','main_edge_row19'),('point','main_edge_row28'),('cylinder','main_edge_row28')]:
    a=np.genfromtxt(DATA/f'trace_{model}_{label}_{.00025:g}.csv',delimiter=',',names=True)
    axes[1,1].plot(a['time'],a['radius'],label=f'{model}, sw={points[label]["somega"]:.4f}',lw=1)
axes[1,1].axhline(6,color='k',ls='--',lw=.8)
axes[1,1].set(xlabel='Time',ylabel='Radius',title='Principal band: same parameters, different source fields',yscale='log')
axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.15)
fig.savefig(FIG/'comparison_and_traces.png',dpi=180);plt.close(fig)

all_runs={name:json.loads((DATA/f'{name}.json').read_text())['results'] for name in
    ('cylinder_fp64','cylinder_cpu_fp32','point_fp64','cylinder_finer','cylinder_long')}
group={}
for r in all_runs['cylinder_fp64']+all_runs['cylinder_finer']:
    group.setdefault(r['label'],[]).append(r)
unstable=[label for label,rows in group.items() if len({r['status'] for r in rows})>1]
stable=[label for label in group if label not in unstable]
cpu={r['label']:r for r in all_runs['cylinder_cpu_fp32']}
cpu_gpu_mismatch=[label for label,r in cpu.items() if r['status']!=points[label]['archived_status']]
cpu64_mismatch=[label for label,r in cpu.items() if r['status']!=next(x for x in group[label] if x['dt']==.001)['status']]
details=dict(number_of_points=len(points),total_metric_runs=sum(len(r) for r in all_runs.values()),
    total_trace_runs=len(traces),status_changes_under_dt=unstable,
    archived_gpu_vs_cpu_fp32_status_changes=cpu_gpu_mismatch,cpu_fp32_vs_fp64_status_changes=cpu64_mismatch,
    common_step_max_delta_v_stable=float(max(abs(next(r for r in group[label] if r['dt']==.001)['mean_v']-
        next(r for r in group[label] if r['dt']==.00025)['mean_v']) for label in stable
        if all(r['status']==0 for r in group[label]))))
(DATA/'findings.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
print(json.dumps(details,indent=2),flush=True)

fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,label in zip(axes,['stripe_sr3_row32','stripe_sr3_row33']):
    for dt in (.001,.0005,.00025,.000125,.0000625):
        a=np.genfromtxt(DATA/f'trace_cylinder_{label}_{dt:g}.csv',delimiter=',',names=True)
        ax.plot(a['time'],a['radius'],label=f'dt={dt:g}',lw=1)
    ax.axhline(6,color='k',ls='--',lw=.8)
    ax.set(xlabel='Time',ylabel='Radius',title=f'sr={points[label]["sr"]:.6f}, sw={points[label]["somega"]:.6f}')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.savefig(FIG/'sensitive_stripe.png',dpi=180);plt.close(fig)
