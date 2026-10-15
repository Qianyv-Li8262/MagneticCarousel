"""Inspect actual model forces on saved zero-force trajectories."""
from pathlib import Path
import importlib.util,json
import numpy as np
import animate_one_trajectory as defaults
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
build=root/'scratch/laser-tilt-20261009-source/build-no-rolling'
spec=importlib.util.spec_from_file_location('hysteresis',next(build.glob('hysteresis*.pyd')))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
params=mod.SolverParams(*defaults.scaled_model_parameters(8,sr=1,somega=1))
k=4/.07;omega=100*2*np.pi/60;mass=4*np.pi/3*.006**3*7850
out=root/'runs/laser-tilt-20261009';fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
records=[]
for tag,label in [('_H2.264_nopush','No push'),('_H2.264_short_f0.3_t0.2','0.3 mN, 0.2 s push'),('_H2.264_initialv180','Initial rolling speed 180 mm/s')]:
    folder=out/('rpm100_diam12_dt0.0005'+tag)/'tilt_0.000000000'
    table=np.load(folder/'failed_stage_trace.npz')['state_table'];s=table[:,1:16]
    t=table[:,0];r=np.linalg.norm(s[:,1:3],axis=1);v=s[:,3:5]*omega/k
    take=np.arange(0,len(t),max(1,len(t)//1200));ss=s[take];rr=r[take]
    force=np.array([mod.laser_magnetic_force(row,params) for row in ss])*mass*omega**2/k*1000
    radial=np.sum(force[:,:2]*ss[:,1:3],axis=1)/rr
    vt=(ss[:,1]*ss[:,4]-ss[:,2]*ss[:,3])/rr*omega/k
    required=mass*vt**2/(rr/k)*1000
    near=(rr/k>.065)&(rr/k<.075)
    record={'case':label,'time_to_175mm_s':float(t[-1]),'magnetic_inward_force_near_70mm_mN_mean':float(-radial[near].mean()),'magnetic_inward_force_near_70mm_mN_range':[-float(radial[near].max()),-float(radial[near].min())],'circular_force_near_70mm_mN_mean':float(required[near].mean()),'mean_normal_load_over_weight_near_70mm':float(np.mean(1-force[near,2]/(mass*9.81*1000)))}
    records.append(record)
    axes[0,0].plot(s[:,1]/k*1000,s[:,2]/k*1000,label=label)
    axes[0,1].plot(t,r/k*1000,label=label)
    axes[1,0].plot(t,np.linalg.norm(v,axis=1)*1000,label=label)
    if 'Initial' in label:
        axes[1,1].plot(t[take],-radial,label='Inward magnetic force')
        axes[1,1].plot(t[take],required,label='m v_t^2 / r (circular motion)')
axes[0,0].set(xlabel='x (mm)',ylabel='y (mm)',title='Horizontal trajectories');axes[0,0].set_aspect('equal')
axes[0,1].set(xlabel='Time after force removal (s)',ylabel='Radius (mm)');axes[0,1].axhline(175,color='gray',ls='--')
axes[1,0].set(xlabel='Time after force removal (s)',ylabel='Speed (mm/s)')
axes[1,1].set(xlabel='Time (s)',ylabel='Force (mN)');axes[1,1].set_xlim(0,7)
for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('100 RPM | gap 39.62 mm | flat plate | added rolling resistance OFF')
fig.savefig(out/'escape_diagnostic_100rpm.png',dpi=160)
(out/'escape_diagnostic_100rpm.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
