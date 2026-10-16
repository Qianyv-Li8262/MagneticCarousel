"""Document the converged rolling-resistance candidates and production check."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import animate_one_trajectory as animation
from laser_tilt_experiment import summarize
root=Path(__file__).resolve().parent;run=root/'runs/laser-tilt-20261009'
out=run/'rolling_candidates';out.mkdir(exist_ok=True)
module=animation.load_hysteresis();params=module.SolverParams(sr=1.0, somega=2)
omega=200*2*np.pi/60
fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
records=[];table_rows=[]
for damping in [.006,.0065,.007]:
    folder=run/f'rpm200_diam12_dt0.0005_roll_final_low1e5_high1e4_l{damping:g}_v0075'/'tilt_0.002777749'
    original=json.loads((folder/'summary.json').read_text())
    table=np.load(folder/'trajectory.npz')['free']
    metric,_=summarize(table,omega,module,params)
    assert metric['steady_free_orbit'] and table[-1,0]>599
    record={'mu_low':.00001,'mu_high':.0001,'lambda':damping,'rpm':200,'v0':-.075,'duration_s':float(table[-1,0]),'metrics':metric,'trajectory':str(folder/'trajectory.npz'),'binary_sha256':original['binary_sha256'],'field_lut_sha256':original['field_lut_sha256']}
    records.append(record)
    t=table[:,0];s=table[:,1:16];r=np.linalg.norm(s[:,1:3],axis=1)*17.5
    speed=np.linalg.norm(s[:,3:5],axis=1)*omega*17.5
    label=f'lambda={damping:g}; T={metric["mean_orbit_period_s"]:.3f} s'
    axes[0].plot(t[::30],r[::30],label=label,lw=.9)
    tail=t>t[-1]-20
    axes[1].plot(t[tail]-t[-1],speed[tail],label=label,lw=.8)
    lap=(t>=metric['analysis_time_range_s'][1]-metric['mean_orbit_period_s'])&(t<=metric['analysis_time_range_s'][1])
    axes[2].plot(s[lap,1]*17.5,s[lap,2]*17.5,label=label)
    table_rows.append(f'| {damping:g} | {metric["mean_orbit_period_s"]:.4f} | {metric["radius_mean_mm"]:.2f} | {metric["mean_tangential_speed_mm_s"]:.2f} | {metric["first_harmonic_relative_half_amplitude_percent"]:.2f}% |')
axes[0].set(xlabel='Physical time (s)',ylabel='Orbit radius (mm)')
axes[1].set(xlabel='Time relative to simulation end (s)',ylabel='COM speed (mm/s)')
axes[2].set(xlabel='x (mm)',ylabel='y (mm)');axes[2].set_aspect('equal')
for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=9)
fig.suptitle('200 RPM | ball 12 mm | gap 39.62 mm | tilt 0.15915 deg\nmu_low=0.00001; mu_high=0.0001; v0=-0.075; no blowing')
fig.savefig(out/'rolling_candidates.png',dpi=150);plt.close(fig)
production=json.loads((out/'production_600s_summary.json').read_text())
half=json.loads((run/'rpm200_diam12_dt0.00025_roll_final_halfdt/summary.json').read_text())[0]
scan=[]
for path in sorted(run.glob('rpm200*roll*/summary.json')):
    for item in json.loads(path.read_text()):
        scan.append({'case':path.parent.name,'status':item['status'],'v0':item.get('v0'),'mu_low':item.get('rolling_mu_low'),'mu_high':item.get('rolling_mu_high'),'lambda':item.get('rolling_lambda'),'steady':item.get('metrics',{}).get('steady_free_orbit',False),'period_s':item.get('metrics',{}).get('mean_orbit_period_s')})
(out/'candidates.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(out/'all_screened_cases.json').write_text(json.dumps(scan,indent=2),encoding='utf-8')
description=f'''# 降低低速滚动阻力后的稳定轨道

固定视频转速 200 RPM、球径 12 mm、磁间距 39.62 mm、倾角 0.1591533°，材料参数不变。
原低速系数 0.01 降至 0.00001（下降 1000 倍）；高速系数仍为 0.0001。
初速 v0=-0.075（初始切向速度约 -109.96 mm/s），初始纯滚动；全程无吹气。
固定磁化两圈（0.6 s），下盘相位 0.58 rad，球心初始 (70,0) mm。

三个候选均运行 600 s，末段始终反向公转；最近五圈周期跨度小于 1%，圈均半径跨度小于 0.5 mm。

| 线性阻尼 lambda | 球公转周期/s | 平均半径/mm | 平均切向速度/mm/s | 一次谐波速度相对半幅 |
|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

正式构建选 lambda=0.006；实际正式模块也运行 600 s，获得周期 {production['metrics']['mean_orbit_period_s']:.6f} s，
最近五圈平均半径跨度 {production['metrics']['last5_lap_mean_radius_span_mm']:.5f} mm。
步长减半的独立验证周期 {half['metrics']['mean_orbit_period_s']:.6f} s。
正式构建 12/12 检查通过。正式磁场表范围 35–105 mm，所选轨道未越界。

lambda=0.008 在保留非零低速项和高速系数的这组配置下退回摆动，排除。
只降低低速项、保留原 lambda=0.003 的筛选未得到持续稳定绕圈；故也对线性阻尼作小幅调整。
这些是数值上稳定的候选，不是实验阻力的实测值或定量拟合。球公转周期尚未匹配视频约 2.4–2.5 s。
倾斜仍用现有小角度等效力模型，未重新计算倾斜几何中的磁场间隙。

## 启动（已设为动画默认）

```powershell
& 'C:\\Users\\38359\\miniconda3\\envs\\daily\\python.exe' .\\src2d\\animate_one_trajectory.py --fast --fps 60 --steps-per-frame 350
```

三个候选的 NPZ 路径和指标见 candidates.json；全部筛选记录见 all_screened_cases.json。
production_600s_summary.json 中保存正式构建的完整初态、参数、文件校验和及验证指标。
lambda=0.0065 或 0.007 的编译配置需要修改 kRollingLambda 后重新构建。
'''
(out/'README.md').write_text(description,encoding='utf-8')
print(json.dumps([{'lambda':r['lambda'],'T_s':r['metrics']['mean_orbit_period_s'],'R_mm':r['metrics']['radius_mean_mm'],'A1_percent':r['metrics']['first_harmonic_relative_half_amplitude_percent']} for r in records],indent=2))
print('Production T:',production['metrics']['mean_orbit_period_s'],'half-step T:',half['metrics']['mean_orbit_period_s'])
