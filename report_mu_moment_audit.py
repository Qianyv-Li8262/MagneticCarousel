"""Summarize the completed sensitivity and matched-observable experiments."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_mu_moment import OUT

records=[]
for path in OUT.glob('h*.json'):
    r=json.loads(path.read_text());r['_path']=path;records.append(r)
def find(mu,m,lam,seconds,dt):
    return next(r for r in records if abs(r['mu_high']-mu)<1e-12 and abs(r['moment_scale']-m)<1e-8 and abs(r['lambda']-lam)<1e-12 and r['seconds']==seconds and r['dt']==dt)
base=find(.0001,3.04,.004,600,.00025)
selected=[base,find(0,3.0415,.005,600,.00025),find(.0003,3.041,.0025,600,.00025)]
period=json.loads((OUT/'period_matched.json').read_text())
fixed=period['fits']
for r in selected:
    r['period_error_percent']=100*(r['metrics']['period_s']/base['metrics']['period_s']-1)
    r['amplitude_error_pp']=r['metrics']['amplitude_percent']-base['metrics']['amplitude_percent']
    r['phase_difference_deg']=float(np.degrees(np.angle(np.exp(1j*(r['metrics']['phase_rad']-base['metrics']['phase_rad'])))))
    r['matched']=bool(r['stationary'] and abs(r['period_error_percent'])<.1 and abs(r['amplitude_error_pp'])<.5)
def clean(r):return {k:(str(v) if isinstance(v,type(OUT)) else v) for k,v in r.items()}
summary={'baseline':clean(base),'period_only_fixed_lambda':fixed,'both_matched_candidates':[clean(r) for r in selected],
         'acceptance':{'period_error_percent':.1,'amplitude_error_percentage_points':.5},
         'fixed':{'rpm':200,'ball_diameter_mm':12,'ring_radius_mm':70,'magnet_diameter_mm':28,'magnet_thickness_mm':15,'center_height_mm':39.62,'tilt_rad':float(.5*np.arctan(4/930)),'mu_low':1e-5,'contact_mu':.02,'transition_omega':1.5,'regularization_omega':.002,'startup_v0':-.075,'push':0},
         'note':'mu_high is added rolling torque coefficient, not contact sliding friction. All material constants and tilt remain fixed. Amplitude is first spatial harmonic of tangential speed divided by fitted mean.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
mu=[r['mu_high'] for r in fixed]
axs[0].plot(mu,[r['moment_scale'] for r in fixed],'o-',color='#8c4aa5')
axs[0].axhline(3.04,color='gray',ls='--',lw=1)
axs[0].set(xscale='symlog',xlabel='High-speed rolling coefficient',ylabel='Magnet moment / original reference',title='Same period; linear damping fixed')
axs[0].set_xscale('symlog',linthresh=2.5e-5)
axs[1].plot(mu,[r['metrics']['amplitude_percent'] for r in fixed],'o-',color='#d7782a')
axs[1].axhline(base['metrics']['amplitude_percent'],color='gray',ls='--',lw=1)
axs[1].set(xlabel='High-speed rolling coefficient',ylabel='Speed first harmonic / mean (%)',title='Period alone does not match modulation')
axs[1].set_xscale('symlog',linthresh=2.5e-5)
axs[0].set_xlim(0,.012)
axs[1].set_xlim(0,.012)
grid=np.linspace(0,2*np.pi,500)
colors=['#3267b8','#21966e','#d56445']
for r,color in zip(selected,colors):
    rows=np.load(r['_path'].with_suffix('.npz'))['rows'];rows=rows[rows[:,0]>=480]
    s=rows[:,1:16];phi=np.arctan2(s[:,2],s[:,1]);rad=np.hypot(s[:,1],s[:,2])
    v=-(s[:,1]*s[:,4]-s[:,2]*s[:,3])/rad
    co=np.linalg.lstsq(np.c_[np.ones(len(phi)),np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi)],v,rcond=None)[0]
    fit=np.c_[np.ones(len(grid)),np.cos(grid),np.sin(grid),np.cos(2*grid),np.sin(2*grid)]@co
    label=f"mu={r['mu_high']:g}, m={r['moment_scale']:.4f}, lambda={r['lambda']:g}"
    axs[2].plot(np.degrees(grid),100*(fit/co[0]-1),label=label,color=color,lw=2)
axs[2].set(xlabel='Orbital phase (degrees)',ylabel='Fitted speed deviation / mean (%)',title='Matched period and similar modulation')
axs[2].legend(fontsize=8,loc='lower left')
for ax in axs:ax.grid(alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
lines=['# 高速滚阻与拟合磁矩独立测试（2026-10-10）','',
'基线取此前3.04倍磁矩，每块12.83689539 A·m²。200 RPM，球径12 mm，磁铁排列圆半径70 mm，球心至磁铁中心面39.62 mm；倾角固定4 mm/930 mm对应值。保持Q235四参数、接触摩擦0.02、低速滚阻1e-5、过渡自转角速度1.5、初速度和初始磁化方案不变。无吹气。',
'','## 方法','',
'使用独立的SI参数计算和统计代码，调用已有FP64诊断核心；高速滚阻与磁矩在独立进程内调整，没有改动正式配置。先固定线性滚阻lambda=0.004，只用磁矩匹配基线周期；再允许线性滚阻改变以匹配速度一阶半振幅。周期用末120秒展开公转角的回归斜率计算，另用完整圈截线周期交叉检查。振幅用切向速度对公转相位的前两阶谐波拟合，一阶幅值除以拟合均值。',
'','## 仅匹配周期：固定lambda=0.004，240秒筛选','',
'| 高速滚阻 | 磁矩倍数 | 每块磁矩 / A·m² | 相对基线磁矩变化 | 周期 / s | 一阶半振幅 |',
'|---:|---:|---:|---:|---:|---:|']
for r in fixed:
    met=r['metrics'];lines.append(f"| {r['mu_high']:g} | {r['moment_scale']:.6f} | {r['moment_A_m2']:.6f} | {r['moment_change_percent']:+.3f}% | {met['period_s']:.6f} | {met['amplitude_percent']:.3f}% |")
lines+=['','## 同时匹配周期和近似波动：600秒、步长0.00025验证','',
'接受条件：周期差<0.1%，一阶半振幅差<0.5个百分点，并在末三个60秒窗口中无明显漂移。以下数值均为重新运行结果。',
'','| 高速滚阻 | lambda | 磁矩倍数 | 每块磁矩 / A·m² | 周期 / s | 半振幅 | 周期差 | 幅值差 / 百分点 | 相位差 / 度 | 接受 |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in selected:
    met=r['metrics'];lines.append(f"| {r['mu_high']:g} | {r['lambda']:g} | {r['moment_scale']:.4f} | {r['moment_A_m2']:.6f} | {met['period_s']:.6f} | {met['amplitude_percent']:.3f}% | {r['period_error_percent']:+.4f}% | {r['amplitude_error_pp']:+.3f} | {r['phase_difference_deg']:+.3f} | {'通过' if r['matched'] else '未通过'} |")
lines+=['','## 解释与限制','',
'高速滚阻从1e-4改变到0或3e-4，可以通过线性滚阻补偿，以几乎相同的磁矩得到同周期、近似一阶波动。仅固定lambda时，mu=1e-3和3e-3的周期匹配磁矩变化约1.28%和3.61%；mu=1e-2时需要增大约45.61%，但此时半振幅约13.67%，不能当作同时匹配成功。',
'','对mu=1e-3另作磁矩与lambda的局部联合搜索（lambda允许0至0.025，磁矩倍数0.5至8），最好的所选候选仍约9.96%半振幅，未达到0.5个百分点的幅值容差。这是有限范围局部搜索结果，不证明所有参数空间都没有解。',
'','mu_high调整的是随球面内自转角速度变化的附加滚动转矩，高速系数也影响过渡区；它不等于接触滑动摩擦Mu。长期候选主要为滑动状态。磁矩是模型拟合输入，摩擦不会改变磁铁实际固有磁矩。本试验没有解决厂家近面场与原3倍磁矩的物理标定矛盾。',
'','诊断场表范围比正式表更宽；本次选中轨道均在约69至71 mm范围内。主结果比较了600秒的0.0005和0.00025步长；对基线与mu=3e-4候选，周期差约百万分之一量级，幅值差小于0.0001个百分点。保留全部筛选JSON、选中轨迹和联合搜索结果以供复查。',
'','![比较图](comparison.png)']
(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'runs':len(records),'matched':[r['matched'] for r in selected]}))
