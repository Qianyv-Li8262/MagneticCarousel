"""Report joint horizontal spin-damping period search without hiding spin cost."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_horizontal_spin_period as s
f=s.f
records=s.load(['baseline','joint','refined'])
validated=s.load(['validated'])
checks=s.load(['final_check'])
eligible=[r for r in validated if r.get('stationary') and s.spin_settled(r)]
low=[r for r in eligible if r['normal_spin_rms_rad_s']<=s.SPIN_LIMIT]
best=min(low,key=s.score)
baseline=next(r for r in validated if r['spin_tau_s']==15 and np.allclose(list(r['parameters'].values()),s.VALUES))
def same(r):
    return r['spin_tau_s']==best['spin_tau_s'] and np.allclose(list(r['parameters'].values()),list(best['parameters'].values()),rtol=1e-12,atol=1e-15)
fine=next(r for r in checks if same(r) and r['dt']==.0005)
other=next(r for r in checks if same(r) and r['rpm']==f.TARGETS[0]['disk_rpm'])
startup=next(r for r in checks if same(r) and r['v0']==-.05)
mass=4*np.pi/3*f.BALL_R**3*f.DENSITY;inertia=.4*mass*f.BALL_R**2
step_error=abs(fine['period_s']/best['period_s']-1)
change=best['period_s']/s.CENTER['period_s']-1
spin_reduction=1-best['normal_spin_rms_rad_s']/abs(s.CENTER['normal_spin_mean_rad_s'])
old_rows=np.load(s.OLD/(s.CENTER['key']+'.npz'))['rows']
old_tail=old_rows[old_rows[:,0]>=old_rows[-1,0]-90]
old_spin_rms=float(np.sqrt(np.mean(old_tail[:,12]**2))*s.CENTER['rpm']*2*np.pi/60)
spin_reduction=1-best['normal_spin_rms_rad_s']/old_spin_rms
summary=dict(best=best,baseline_with_tau15=baseline,baseline_without_spin=s.CENTER,
             initial_run_count=len(records),validated=validated,final_checks=checks,
             period_relative_change=change,spin_relative_reduction=spin_reduction,
             time_step_relative_error=step_error,spin_c_N_m_s=inertia/best['spin_tau_s'],inertia_kg_m2=inertia,
             original_spin_rms_rad_s=old_spin_rms,
             matched_measured_periods=bool(all(abs(r['period_s']/t['ball_period_target_s']-1)<.01 for r,t in [(best,f.TARGETS[1]),(other,f.TARGETS[0])])),
             workflow_sha256={n:hashlib.sha256((f.ROOT/n).read_bytes()).hexdigest() for n in ['fit_horizontal_period.py','fit_horizontal_spin_period.py','prepare_horizontal_spin_period.py','report_horizontal_spin_period.py']})
(f.OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 水平盘：在 2.78579 s 候选附近加入自旋阻力并联合调参','',
       f"最接近且通过长期复核的低自旋候选：周期 **{best['period_s']:.8f} s**，Z 自旋 RMS **{best['normal_spin_rms_rad_s']:.6f} rad/s**，平均半径 **{best['radius_mean_mm']:.5f} mm**。相对原候选周期变化 {change*100:+.3f}%，自旋降低约 {spin_reduction*100:.2f}%。",'',
       '## 保持的条件','',
       '球心到磁铁中心面高度 39.62 mm；每块磁铁参考磁矩约 4.222663 A·m²，Beta=0.056999326368103；水平盘，倾斜项在独立构建中编译关闭。圆柱磁铁直径 28 mm、厚 15 mm、8 个交替极性、环半径 70 mm；钢球直径 12 mm。接触摩擦 0.02。',
       '下盘转速在每次运行中恒定，两视频用各自测得的平均 RPM；不拟合速度波动。滚阻时间归一化随 RPM 换算，保持同一有量纲规律。',
       '初态沿用半径 70 mm、v0*=-0.075、初始纯滚动、Z 自旋为零；固定球预磁化两圈后在盘相位 0.58 rad 释放，不吹气。最后还用 v0*=-0.05 复核。','',
       '## 新增自旋阻力','',
       '只新增法向被动黏性力矩 Tz=−cz·omega_z；tau_z=I/cz，以秒指定。归一化角加速度为 −lambda_z·omega_z*，lambda_z=1/(Omega·tau_z)。耗散功率 −cz·omega_z²≤0；它始终独立于平面滚动阻力。tau=None 表示零新增阻尼，不能把 tau=0 当成无阻尼。',
       '自旋使用均方根、绝对值均值与峰值，避免正负抵消。此前“1–2 秒一圈”仅作参考，因此另列 RMS≤2π rad/s 的低自旋候选；本轮没有把该口述参考当作精确拟合目标，也没有要求自旋必须落在该区间下限以上。','',
       '## 对照','', '|方案|公转周期/s|Z 自旋 RMS/rad s⁻¹|平均半径/mm|', '|---|---:|---:|---:|',
       f"|原候选，无 Z 阻力|{s.CENTER['period_s']:.6f}|{old_spin_rms:.4f}|{s.CENTER['radius_mean_mm']:.4f}|",
       f"|只加自旋阻力，tau=15 s|{baseline['period_s']:.6f}|{baseline['normal_spin_rms_rad_s']:.4f}|{baseline['radius_mean_mm']:.4f}|",
       f"|联合调整磁滞、平面滚阻与 Z 阻力|{best['period_s']:.6f}|{best['normal_spin_rms_rad_s']:.4f}|{best['radius_mean_mm']:.4f}|",'',
       '仅新增阻力的对照从原稳定状态和重新释放两种初态出发，tau 扫描为 0.1、0.3、1、3、10、15、30、100、300 s，并有无阻尼对照；各运行 600 s。此对照中压低自旋不会自动缩短公转周期。',
       '## 联合搜索与参数','',
       f'本轮初筛 {len(records)} 次运行（20 次只改阻尼、160 次附近联合采样及后续局部细化）。联合采样的 mu_r、Alpha_JA 为原候选的 0.5–2 倍，C=0.18–0.60，K 与三个滚阻系数为原值的 0.25–4 倍，自旋衰减时间约 0.1–316 s；局部细化围绕较好的低自旋候选，参数再作约 0.625–1.6 倍扰动，tau 作约 0.5–2 倍扰动。具体可复现采样种子和范围见脚本。过渡滚动角速度与正则化角速度仍固定为参考无量纲 1.5、0.002。','',
       '|参数|候选值|', '|---|---:|']
for key,value in best['parameters'].items():lines.append(f'|{key}|{value:.15g}|')
lines += [f"|tau_z / s|{best['spin_tau_s']:.15g}|",f"|c_z / N m s|{inertia/best['spin_tau_s']:.12g}|",'',
          '这些是动力学试验候选参数，尤其 tau_z 并非球的实测自由衰减时间；不能将本轮选值当成已确认的真实接触阻力。', '',
          '## 长期与数值复核','',
          '选定候选运行 1200 s、dt*=0.00025；末尾 90 s 分成三个窗口，角速度变化 <1%、平均半径变化 <0.5 mm、反向运动占比 >99%，自旋 RMS 窗口变化小于 max(0.02 rad/s,1%)。这些是有限时间判据，不是严格吸引子证明。',
          f"步长 dt*=0.0005 与 0.00025 的周期相对差 {step_error:.4g}。初速 v0*=-0.05 的检查：周期 {startup.get('period_s')}, 有界稳态判据 {startup.get('stationary')}。",'',
          '|视频|盘速/RPM|实测周期/s|模拟周期/s|相对误差|', '|---|---:|---:|---:|---:|']
for r,t in [(other,f.TARGETS[0]),(best,f.TARGETS[1])]:
    lines.append(f"|{t['video']}|{t['disk_rpm']:.6f}|{t['ball_period_target_s']:.6f}|{r['period_s']:.6f}|{100*(r['period_s']/t['ball_period_target_s']-1):+.3f}%|")
lines += ['', '两段视频仍分别慢约 14.63% 和 11.42%，本轮未匹配实测周期。该组仅为低自旋数值候选；平均半径约 81.60 mm，也高于此前 65–75 mm 的视频先验。逃逸/越界的瞬时周期不计入成功候选，搜索没有证明其他参数或分支不存在。',
          '## 检查与复现','',
          'verification.json 包含无磁驱动下的解析指数衰减核对，以及零阻尼重现原 2.78578872 s 周期的回归。正式默认参数未被试验候选覆盖；新阻尼与参数均保存在独立水平盘试验构建。',
          '依次使用 daily Python 运行 prepare_horizontal_spin_period.py，然后运行 fit_horizontal_spin_period.py 的 check、baseline、joint、refine、verify、final-check 动作，最后运行 report_horizontal_spin_period.py。构建与源文件/磁场表哈希见 manifest.json；每条 JSON 保存实际参数、tau、RPM、步长、初态和结果，1200 秒验证的完整轨迹在同 key 的 NPZ 文件中。','']
(f.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

fig,ax=plt.subplots(2,1,figsize=(10,6.5),layout='constrained',sharex=True)
for r,label,color,root in [(s.CENTER,'Original, no Z damping','0.55',s.OLD),(best,'Joint low-spin candidate','tab:blue',f.OUT)]:
    rows=np.load(root/(r['key']+'.npz'))['rows'];t=rows[:,0]
    phi=np.unwrap(np.arctan2(rows[:,3],rows[:,2]));centres=[];period=[]
    for lo in np.arange(0,t[-1]-30,30):
        ok=(t>=lo)&(t<lo+30);rate=np.polyfit(t[ok]-t[ok].mean(),phi[ok],1)[0]
        centres.append(lo+15);period.append(-2*np.pi/rate if rate<-.01 else np.nan)
    ax[0].plot(centres,period,color=color,label=label)
    ax[1].plot(t[::5],abs(rows[::5,12])*r['rpm']*2*np.pi/60,color=color,label=label)
ax[0].axhspan(2.416257,2.485880,color='orange',alpha=.35,label='Measured mean periods')
ax[0].set(ylabel='Orbital period (s)',ylim=(2.25,4.5),title='Fixed height / magnet moment, horizontal disk')
ax[1].set(xlabel='Time (s)',ylabel='Absolute normal spin (rad/s)')
for a in ax:a.grid(alpha=.2);a.legend()
image=f.ROOT/'scratch/horizontal-spin-period-20261015';image.mkdir(exist_ok=True)
fig.savefig(image/'comparison.png',dpi=170)
print(json.dumps(dict(key=best['key'],period_s=best['period_s'],spin_rms=best['normal_spin_rms_rad_s'],
                     radius_mm=best['radius_mean_mm'],tau_s=best['spin_tau_s'],period_change_percent=100*change,
                     spin_reduction_percent=100*spin_reduction,step_error=step_error),indent=2))
