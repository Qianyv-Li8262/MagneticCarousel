"""Summarize the fixed-height horizontal-disk search and its limits."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_horizontal_period as fit

OUT=fit.OUT
initial=[]
for name in ['coarse','local','refined','weak_k']:
    initial.extend(json.loads((OUT/(name+'.json')).read_text()))
validated=json.loads((OUT/'validated.json').read_text())
checks=json.loads((OUT/'final_check.json').read_text())
best=min([r for r in validated if r.get('stationary') and r['rpm']==fit.TARGETS[1]['disk_rpm']],key=fit.score)
same=lambda r: np.allclose(list(r['parameters'].values()),list(best['parameters'].values()),rtol=1e-12,atol=1e-15)
compared={r['rpm']:r for r in [best,*checks] if same(r) and r.get('stationary') and r['dt']==.00025 and r['v0']==-.075}
base=next(r for r in checks if r['dt']==.0005)
step_error=abs(base['period_s']/best['period_s']-1)
height=json.loads((OUT/'height_recheck.json').read_text())
native=json.loads((fit.SOURCE/'build_manifest.json').read_text())
summary=dict(success=False,initial_run_count=len(initial),unique_initial_parameters=len({tuple(r['parameters'].values()) for r in initial}),
             best=best,video_checks=[dict(video=t['video'],target_period_s=t['ball_period_target_s'],
                    model_period_s=compared[t['disk_rpm']]['period_s'],
                    relative_error=compared[t['disk_rpm']]['period_s']/t['ball_period_target_s']-1) for t in fit.TARGETS],
             relative_step_error=step_error,height_recheck=height,final_checks=checks,native_build=native)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 水平盘：固定磁矩与高度的周期拟合，2026-10-15','',
       '**结论：本轮未找到与两段视频周期匹配的长期有界候选。** 目前最接近且通过 1200 秒复核的周期为 '
       f"{best['period_s']:.8f} s，平均半径 {best['radius_mean_mm']:.5f} mm。有限搜索未成功不构成不存在解的证明。",'',
       '## 高度重新核对','',
       '已回读“分析新视频”对话的原始测量及其后续纠正。实测：上盘厚 5 mm、下盘厚 3 mm、磁铁厚 15 mm、磁铁底面低于下盘上表面 1.5 mm、小球直径 12 mm；上盘上表面到下盘下表面距离 42.62 mm。','',
       '以下盘上表面为基准：球心高度 42.62−3+12/2=45.62 mm；磁铁中心高度 15/2−1.5=6.00 mm。因此球心到磁铁中心面的距离为 **39.62 mm**，对应 H=2.264。上盘厚度已经包含在表面间距中，不能再次加减 5 mm。磁铁上表面到上盘上表面为 26.12 mm。',
       '原始对话与消息标识保存在 height_recheck.json。此处采用该实测高度，没有采用此前按目标 Gauss 反求的其他高度。','',
       '## 固定项与搜索项','',
       '- 倾角为零；独立诊断构建在编译时关闭倾斜盘。没有增加法向 Z 自旋阻力；原有磁力矩仍可改变 Z 自旋。',
       '- 每块磁铁磁矩固定为参考值约 4.222663 A·m²，Beta=0.056999326368103；磁场表、球心高度、圆柱尺寸均固定。',
       '- 磁铁直径 28 mm、厚度 15 mm、8 个交替极性磁铁，中心环半径 70 mm；钢球直径 12 mm，密度 7850 kg/m³，接触摩擦系数 0.02。',
       '- 下盘在每次运行中保持恒定转速；两段视频分别采用实测平均 201.814562、199.831340 RPM。没有拟合或外加速度波动。',
       '- 调整线性化 JA 的 mu_r、Alpha_JA、C_JA、K，以及平面滚阻低/高系数和线性阻尼。过渡角速度 1.5、零速正则化 0.002（长视频无量纲单位）保持固定。',
       '- 跨视频检查保持同一套有量纲滚阻规律：lambda、过渡角速度及正则化角速度均按 Omega_ref/Omega 换算。',
       '- 初态主要为 70 mm 处纯滚动、v0*=-0.075，固定球磁化两圈下盘时间后在盘相位 0.58 rad 释放，无额外推动；另测试较高初速 v0*=-0.12 和最佳候选的 v0*=-0.05。','',
       '## 搜索与长期验证','',
       f'初筛共 {len(initial)} 次运行：112 次基线/拉丁超立方扫描、36 次较高初速检查、90 次局部细化、90 次小 K 细化。初筛运行 180–300 s；原先较好的 10 条进行 600 s、dt*=0.00025 检查，新的较好候选进一步进行 1200 s 复核。','',
       '全局采样覆盖 mu_r=100–10000、Alpha_JA=1e-4 至约 0.02、C_JA=0.02–0.8、K=1e-4 至约 0.05；局部补充 C_JA 至 0.995、K 下探至 1e-7。滚阻测试包含零阻力，正值网格与随机采样范围见 fit_horizontal_period.py。此范围不是穷尽搜索，也不是钢球实测材料参数范围。','',
       '稳定判据：完整运行至指定时间，最后 90 s 分成三个窗口，平均角速度变化 <1%、平均半径变化 <0.5 mm，反向运动占比 >99%。这描述有限时间数值稳定性，不构成严格吸引子证明。逃逸导致超过 175 mm 磁场表边界的候选判为失败，不将其逃逸前周期计为成功。','',
       '## 最佳长期候选（仍未匹配）','',
       '|参数|数值|','|---|---:|']
for key,value in best['parameters'].items():lines.append(f'|{key}|{value:.15g}|')
lines += ['', '|视频|恒定盘速/RPM|实测周期/s|模拟周期/s|相对误差|', '|---|---:|---:|---:|---:|']
for t in fit.TARGETS:
    r=compared[t['disk_rpm']]
    lines.append(f"|{t['video']}|{t['disk_rpm']:.6f}|{t['ball_period_target_s']:.6f}|{r['period_s']:.6f}|{100*(r['period_s']/t['ball_period_target_s']-1):+.3f}%|")
window_text=', '.join(format(w['period_s'],'.9f') for w in best['windows'])
lines += ['',f"长视频 1200 s、dt*=0.00025 的末尾三窗口周期为 {window_text} s。平均 Z 自旋 {best['normal_spin_mean_rad_s']:.4f} rad/s，仅作输出记录，没有对自旋速度做拟合。",
          f"同一候选 dt*=0.0005 与 0.00025 的周期相对差为 {step_error:.3g}；见 final_check.json。",
          '平均半径约 80.15 mm，也超过此前视频 65–75 mm 的半径先验；所以不能将此组参数称为实验拟合，更不能据此认定材料参数已标定。',
          '正式默认配置已关闭倾斜盘并重建 CPU FP64、FP32 Kahan 与四个 CUDA 扫描模块；滚动、倾斜盘、滚滑切换三项原生测试通过。默认磁滞与滚阻值仍保留原基线，未用这个不匹配候选覆盖。','',
          '## 复现与输出','',
          '使用 daily Python 依次运行 prepare_horizontal_period.py、fit_horizontal_period.py scan、local、refine、weak-k、confirm、verify、final-check；最后运行 report_horizontal_period.py。各动作独立缓存，实际命令中的 action 名称使用 weak-k 和 final-check。',
          'manifest.json 保存最初扫描的旧诊断模块来源；summary.json 的 native_build 保存最终水平盘构建及源文件、磁场表的 SHA256。初始模块通过运行时设置倾角为零，最终模块编译时彻底禁用倾斜项。所有逐条 JSON 均保存各自二进制与磁场表哈希。',
          '每条长验证轨迹以其 key 命名保存 NPZ，rows 的列顺序为 [time_s, disk_phase,x,y,vx,vy,quat_w,quat_x,quat_y,quat_z,omega_x,omega_y,omega_z,qx,qy,qz,sliding]；状态采用模型无量纲单位。','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

rows=np.load(OUT/(best['key']+'.npz'))['rows']
t=rows[:,0];phi=np.unwrap(np.arctan2(rows[:,3],rows[:,2]))
centres=[];periods=[]
for lo in np.arange(0,t[-1]-30,30):
    ok=(t>=lo)&(t<lo+30)
    slope=np.polyfit(t[ok]-t[ok].mean(),phi[ok],1)[0]
    centres.append(lo+15);periods.append(-2*np.pi/slope if slope<-.01 else np.nan)
fig,ax=plt.subplots(2,1,figsize=(10,6.5),layout='constrained',sharex=True)
ax[0].plot(centres,periods,label='Best bounded candidate (30 s windows)')
ax[0].axhspan(2.416257,2.485880,color='orange',alpha=.35,label='Measured mean periods')
ax[0].set(ylabel='Orbital period (s)',ylim=(2.25,5.5),title='Horizontal disk, fixed 39.62 mm height and reference magnet moment')
ax[0].legend(loc='upper right');ax[0].grid(alpha=.2)
ax[1].plot(t[::5],np.linalg.norm(rows[::5,2:4],axis=1)/fit.KSPACE*1000)
ax[1].axhspan(65,75,color='orange',alpha=.25,label='Video radius prior')
ax[1].set(xlabel='Time (s)',ylabel='Orbit radius (mm)');ax[1].grid(alpha=.2);ax[1].legend()
image=fit.ROOT/'scratch/horizontal-period-20261015'
image.mkdir(exist_ok=True);fig.savefig(image/'best_candidate.png',dpi=170)
print(json.dumps(dict(best_period_s=best['period_s'],radius_mm=best['radius_mean_mm'],step_error=step_error,
                     video_checks=summary['video_checks'],report=str(OUT/'REPORT.md')),indent=2))
