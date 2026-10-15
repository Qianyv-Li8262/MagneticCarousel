"""Compare all tested interventions and show verified steady periods."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import low_spin_field_grid as g

def main():
 scan=json.loads((g.OUT/'scan_summary.json').read_text())
 for r in scan:
  if not r['stable_clockwise_orbit']:r['speed_first_harmonic_percent']=None
 verified=json.loads((g.OUT/'long_confirmation.json').read_text())
 fields=json.loads((g.OUT/'field_heights.json').read_text())
 labels={'release':'原倾角，直接释放','blow_weak':'原倾角，轻吹','blow_strong':'原倾角，较强吹气','half_tilt_blow':'半倾角，轻吹','flat_blow':'水平盘，轻吹'}
 chosen=[]
 for f in [100,200,300]:
  for tau in [12.,15.,23.]:
   matches=[r for r in verified if r['field_G']==f and r['tau_z_s']==tau and r['stable_clockwise_orbit']]
   if matches:chosen.append(matches[0])
 (g.OUT/'selected_periods.json').write_text(json.dumps(chosen,indent=2))
 with (g.OUT/'all_periods.csv').open('w',newline='',encoding='utf-8-sig') as handle:
  keys=['field_G','tau_z_s','protocol','period_s','stable_clockwise_orbit','spin_z_last180_mean_rad_s','radius_mm','speed_first_harmonic_percent','completed_s','seconds','dt']
  writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
  for r in scan+verified:writer.writerow({k:r[k] for k in keys})
 lines=['# 零自旋释放：100 / 200 / 300 G × 12 / 15 / 23 秒自旋阻尼','',
 '12、15、23 为自由自旋指数衰减时间 τz（秒），不是粘性系数本身。粘性力矩 Tz=−cz ωz，cz=I/τz，I=1.022761825×10⁻⁷ kg m²。',
 '', '## 条件与坡向修正','',
 '本轮初始球心为 (70,0) mm，速度为零，三个自旋分量都为零。先固定球的姿态和位置让磁滞记忆经历两圈盘转动，再释放。盘转速固定 199.83134 RPM，磁矩保持参考值，Q235 材料与接触阻力参数不变。',
 '', '原先的角度 π/2 实际给出 +x 方向驱动力：Fx+=θN sinφ，Fy−=θN cosφ。之前口头说明和旧轨迹图标注 +y 是错误的，旧图已修正；本轮取 φ=π，明确沿 +y 下坡。原倾角约 0.123216°，半倾角约 0.061608°。',
 '', '场强定义为球心高度、半径 70 mm 处，一个磁场周期中 |B| 的 RMS。通过改变高度得到目标场强，磁矩未放大；降低高度会同时改变空间梯度。100 G 的 LUT 作了小幅高度外推，并以独立磁荷面分验证。',
 '', '|场强/G|磁体中心至球心高度/mm|独立场验证相对误差|','|---:|---:|---:|']
 for f in fields:lines.append(f"|{f['target_G']}|{f['height_mm']:.5f}|{f['independent_check']['relative_field_norm_error']:.3g}|")
 lines+=['', '## 干预规则与稳定判据','',
 '所有组从同一零自旋释放条件重新开始，未继承旧高速自旋状态。吹气等效为通过球心、沿逆盘切向的外力：30.0–30.3 秒内施加 0.5 或 1.0 m/s² 加速度（约 3.55 或 7.10 mN）。吹气不直接施加自旋力矩，但允许接触摩擦自然产生自旋；结束后外力归零。减小倾角是整段测试的参数，不在结束后恢复原倾角。',
 '', '先测试 45 组，每组最多 600 秒、归一化步长 0.0005。最后三个 60 秒窗口的公转角速度变化 <1%、平均半径变化 <0.5 mm，并持续逆盘公转且无出界，才标为稳定公转。同一场强/阻尼下周期相差超过 1% 的不同运动各自从零自旋初态重算 1200 秒，步长减半为 0.00025。该判据为有限时长数值检查，不是严格吸引子证明。',
 '', '## 验证后的可维持公转','', '|场强/G|τz/s|方式|周期/s|平均半径/mm|平均 Z 自旋/rad s⁻¹|第一谐波速度半幅/%|','|---:|---:|---|---:|---:|---:|---:|']
 for r in verified:
  if r['stable_clockwise_orbit']:lines.append(f"|{r['field_G']}|{r['tau_z_s']:g}|{labels[r['protocol']]}|{r['period_s']:.6f}|{r['radius_mm']:.4f}|{r['spin_z_last180_mean_rad_s']:.4f}|{r['speed_first_harmonic_percent']:.3f}|")
 lines+=['', '## 全部 600 秒扫描','', '|场强/G|τz/s|方式|稳定公转周期/s|结局|','|---:|---:|---|---:|---|']
 for r in scan:
  status='稳定公转' if r['stable_clockwise_orbit'] else ('出 LUT 范围，不能作为稳定周期' if r['error'] else '停留/未稳定公转')
  lines.append(f"|{r['field_G']}|{r['tau_z_s']:g}|{labels[r['protocol']]}|{format(r['period_s'],'.5f') if r['stable_clockwise_orbit'] else '—'}|{status}|")
 lines+=['', '## 结论边界','', '低初始自旋不等于低最终自旋。200 G 和 300 G 的原坡直接释放组仍由磁驱动自行形成高速自旋，不能据此认为这三个阻尼时间同时复现实验的低自旋。若希望最终维持 1–2 秒自旋一圈，必须在相应场强下重新约束磁驱动力矩与阻尼。',
 '', '没有稳定运动的轨迹不能用局部振动、暂态角速度或出界前的绕行估算值作为稳定公转周期。所列分支仅限本次释放与干预范围；未证明穷尽全部初态。']
 (g.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
 colors={100:'#2581b5',200:'#dc7722',300:'#36955b'}
 for f in [100,200,300]:
  q=sorted([r for r in chosen if r['field_G']==f],key=lambda r:r['tau_z_s'])
  axes[0].plot([r['tau_z_s'] for r in q],[r['period_s'] for r in q],'o-',color=colors[f],label=f'{f} G'+(' (original tilt + blow)' if f==100 else ' (release)'))
 axes[0].axhline(2.48587994094,color='.4',ls='--',label='Video mean 2.486 s')
 axes[0].set(xlabel='Normal-spin free decay time (s)',ylabel='Orbital period (s)',title='Verified sustained orbits');axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
 axes[1].add_patch(Circle((0,0),70,fill=False,ls='--',color='.5'))
 for r in chosen:
  if r['tau_z_s']!=15:continue
  for p in g.OUT.glob('*.json'):
   try:record=json.loads(p.read_text())
   except Exception:continue
   if isinstance(record,dict) and all(record.get(k)==r[k] for k in ['field_G','tau_z_s','protocol','seconds','dt']):
    rows=np.load(p.with_suffix('.npz'))['rows'];break
  rows=rows[rows[:,0]>rows[-1,0]-30];xy=rows[:,2:4]/g.a.K*1000
  axes[1].plot(xy[:,0],xy[:,1],color=colors[r['field_G']],lw=.9,label=f"{r['field_G']} G")
  axes[2].plot(rows[:,0]-rows[0,0],rows[:,12]*g.a.OMEGA,color=colors[r['field_G']],lw=.8,label=f"{r['field_G']} G")
 axes[1].set(xlabel='x (mm)',ylabel='y (mm)',title='Final trajectories, τz=15 s');axes[1].set_aspect('equal');axes[1].legend(fontsize=8);axes[1].grid(alpha=.25)
 axes[2].set(xlabel='Time in final 30 s (s)',ylabel='Normal-axis spin (rad/s)',title='Spin generated after zero-spin release');axes[2].legend(fontsize=8);axes[2].grid(alpha=.25)
 fig.savefig(g.OUT/'periods_and_trajectories.png',dpi=180)
 print(json.dumps([{k:r[k] for k in ['field_G','tau_z_s','protocol','period_s','spin_z_last180_mean_rad_s']} for r in chosen],indent=2))

if __name__=='__main__':main()
