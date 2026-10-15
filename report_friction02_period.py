"""Mu=0.2 comparison, joint fit, verification and explicit contact-mode limits."""
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_friction02_period as s
f=s.f
allr=s.records(['baseline','joint','refined']);validated=s.records(['validated']);checks=s.records(['final_check'])
eligible=[r for r in validated if s.settled(r)]
low=[r for r in eligible if r['normal_spin_rms_rad_s']<=s.LIMIT]
best=min(low,key=s.score)
baseline=next(r for r in s.records(['baseline']) if r['seconds']==1200 and r['seed'] is None and 'initial_slip_normalized' not in r)
sliding=next(r for r in s.records(['baseline']) if r.get('initial_slip_normalized')==-.075)
oldsliding=json.loads((f.OUT/'old_mu_sliding_start.json').read_text())
comparison=json.loads((f.OUT/'friction_only_comparison.json').read_text())
def same(r):return r['spin_tau_s']==best['spin_tau_s'] and np.allclose(list(r['parameters'].values()),list(best['parameters'].values()),rtol=1e-12,atol=1e-15)
fine=next(r for r in checks if same(r) and r['dt']==.0005)
other=next(r for r in checks if same(r) and r['rpm']==f.TARGETS[0]['disk_rpm'])
step_error=abs(fine['period_s']/best['period_s']-1)
mass=4*np.pi/3*f.BALL_R**3*f.DENSITY;inertia=.4*mass*f.BALL_R**2
summary=dict(contact_friction=.2,best=best,baseline=baseline,old_candidate=s.CENTER,
             initial_run_count=len(allr),friction_only_comparison=comparison,old_mu_sliding_start=oldsliding,
             new_mu_sliding_start=sliding,validated=validated,final_checks=checks,time_step_relative_error=step_error,
             matched=False,build=json.loads((f.SOURCE/'build_manifest.json').read_text()),
             workflow_sha256={n:hashlib.sha256((f.ROOT/n).read_bytes()).hexdigest() for n in ['prepare_friction02_period.py','fit_friction02_period.py','fit_horizontal_period.py','report_friction02_period.py']})
(f.OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 接触摩擦固定为 0.2：测试与联合调参','',
       f"目前通过 1200 秒复核的低自旋候选：周期 **{best['period_s']:.8f} s**，平均半径 **{best['radius_mean_mm']:.5f} mm**，Z 自旋 RMS **{best['normal_spin_rms_rad_s']:.6f} rad/s**。",'',
       f"相对原候选周期仅改变 {100*(best['period_s']/s.CENTER['period_s']-1):+.5f}%，公转周期基本不变。本轮仍未匹配约 2.42–2.49 s 的视频周期，平均半径也高于 65–75 mm 的先验。",'',
       '## 固定项与接触模型','',
       '**接触摩擦 Mu=0.2 固定，不参与任何采样或优化。** 当前模型将同一个 Mu 用于滑动库仑摩擦和纯滚动可维持的摩擦阈值，并未另外拆分静/动摩擦。平面滚动阻力、自旋阻力是独立的耗散项。',
       '球心到磁铁中心面高度固定 39.62 mm，每块磁铁磁矩约 4.222663 A·m²、Beta=0.056999326368103。水平盘，下盘每次运行中保持恒定 RPM；主拟合用长视频 199.831340 RPM，再用短视频 201.814562 RPM 检查。同一套有量纲滚阻及自旋阻力跨 RPM 换算。',
       '球直径 12 mm、密度 7850 kg/m³、惯量系数 0.4；8 个交替极性圆柱磁铁，直径 28 mm、厚 15 mm、中心环半径 70 mm；场模型与查表不变。滚阻过渡角速度 1.5、正则化角速度 0.002 固定。没有拟合速度波动，没有持续外力。','',
       '## 只改摩擦系数的对照','',
       f"原候选在 Mu=0.02 时周期 {s.CENTER['period_s']:.9f} s；将其余参数和初态保持不变，仅令 Mu=0.2，周期为 {baseline['period_s']:.9f} s。",
       f"两次完整 1200 秒轨迹均为 {comparison['samples']} 个记录点，最大绝对状态差 {comparison['max_absolute_trajectory_difference']:.4g}，全程纯滚动比例均为 100%。因此这条轨迹没有触及滑动摩擦或其容量限制，调大 Mu 不改变该解；不能将 Mu 当作纯滚动阻力的倍率。",'',
       '另增加“球心有初速、平面自旋为零”的滑动启动对照，仍保持相同预磁化协议：', '',
       '|摩擦系数|滑动初态的最终周期/s|有界稳态判据|', '|---:|---:|---|',
       f"|0.02|{oldsliding.get('period_s')}|{oldsliding.get('stationary')}|",
       f"|0.2|{sliding.get('period_s')}|{sliding.get('stationary')}|",'',
       '## 联合调参','',
       f'本轮初筛/对照合计 {len(allr)} 次。先测原候选的重新释放、旧稳态接续、零平面自旋启动、较弱自旋阻力与 Q235 基线；再做 128 次当前候选附近采样、64 次较宽范围采样，以及较好候选附近局部细化。',
       '局部主采样的 Mu_r、Alpha_JA、K 和三个平面滚阻系数约为当前候选的 0.25–4 倍，C=0.08–0.73，tau_z 约为当前值的 0.1–10 倍。宽范围采样额外覆盖 Mu_r=100–1e5、Alpha_JA=1e-4 至约 0.032、K=1e-7 至约 0.032、C=0.02–0.95、自旋衰减时间约 0.1–316 s。精确采样与种子见 fit_friction02_period.py。',
       '低自旋参考阈值沿用 RMS≤2π rad/s；它是此前口述数量级参考，不是新增的精确观测值。更快但自旋较高的候选也纳入长期复核，未以低自旋筛选掩盖快周期。','',
       '|参数|固定/候选值|', '|---|---:|', '|接触 Mu（固定）|0.2|']
for key,value in best['parameters'].items():lines.append(f'|{key}|{value:.15g}|')
lines += [f"|tau_z / s|{best['spin_tau_s']:.15g}|",f"|c_z / N m s|{inertia/best['spin_tau_s']:.12g}|",'',
          '以上仍是模型试验候选值，尤其 tau_z 未经独立自由衰减实验标定。','',
          '## 验证及与视频比较','',
          '选定候选运行 1200 s，dt*=0.00025；末尾三个窗口角速度变化 <1%、平均半径变化 <0.5 mm、逆盘运动占比 >99%，自旋 RMS 也需收敛。候选同时检查原步长 0.0005、另一视频 RPM、较低初速 v0*=-0.05，以及初始平面自旋为零。',
          f"步长减半的周期相对差为 {step_error:.4g}。最佳候选最后统计段的纯滚动比例为 {best['rolling_fraction']*100:.4f}%。",'',
          '|视频|实测周期/s|模拟周期/s|相对误差|', '|---|---:|---:|---:|']
for r,t in [(other,f.TARGETS[0]),(best,f.TARGETS[1])]:
    lines.append(f"|{t['video']}|{t['ball_period_target_s']:.6f}|{r['period_s']:.6f}|{100*(r['period_s']/t['ball_period_target_s']-1):+.3f}%|")
lines += ['', '本轮有限搜索不构成不存在其他参数或分支的证明。轨道半径还需与此前 65–75 mm 的视频先验比较；越界、未收敛和暂态周期不称为成功匹配。',
          '正式二维配置 Mu 已更新为 0.2，CPU FP64、FP32 Kahan 和四个 CUDA 扫描模块均重建；滚滑切换、滚动阻力、倾斜盘三项原生测试通过。其余材料/滚阻默认值未用尚未匹配实验的候选覆盖。', '',
          '## 复现','',
          '用 daily Python 运行 prepare_friction02_period.py，再依次运行 fit_friction02_period.py 的 baseline、joint、refine、verify、final-check 动作，最后运行 report_friction02_period.py。每条结果保存 contact_friction=0.2，程序也从实际编译模块读取并断言该值。构建、磁场表和脚本哈希见 manifest.json、summary.json。候选 NPZ 保存完整轨迹，文件名与 JSON key 相同。','']
(f.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

fig,ax=plt.subplots(2,1,figsize=(10,6.5),layout='constrained',sharex=True)
for r,label,root in [(s.CENTER,'Old candidate, contact Mu=0.02',s.OLD),(best,'Selected candidate, contact Mu=0.2',f.OUT)]:
    rows=np.load(root/(r['key']+'.npz'))['rows'];t=rows[:,0];phi=np.unwrap(np.arctan2(rows[:,3],rows[:,2]));tc=[];period=[]
    for lo in np.arange(0,t[-1]-30,30):
        ok=(t>=lo)&(t<lo+30);rate=np.polyfit(t[ok]-t[ok].mean(),phi[ok],1)[0]
        tc.append(lo+15);period.append(-2*np.pi/rate if rate<-.01 else np.nan)
    ax[0].plot(tc,period,label=label)
    ax[1].plot(t[::5],np.linalg.norm(rows[::5,2:4],axis=1)/f.KSPACE*1000,label=label)
ax[0].axhspan(2.416257,2.485880,color='orange',alpha=.3,label='Measured mean periods')
ax[0].set(ylabel='Orbital period (s)',title='Fixed contact friction Mu=0.2: joint parameter search',ylim=(2.25,4.5))
ax[1].axhspan(65,75,color='orange',alpha=.2,label='Video radius prior')
ax[1].set(xlabel='Time (s)',ylabel='Orbit radius (mm)')
for a in ax:a.legend();a.grid(alpha=.2)
image=f.ROOT/'scratch/friction02-period-20261015';image.mkdir(exist_ok=True);fig.savefig(image/'comparison.png',dpi=170)
print(json.dumps(dict(key=best['key'],period_s=best['period_s'],radius_mm=best['radius_mean_mm'],spin_rms_rad_s=best['normal_spin_rms_rad_s'],tau_s=best['spin_tau_s'],step_error=step_error,count=len(allr)),indent=2))
