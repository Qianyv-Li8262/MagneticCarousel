"""Document the moment/rolling/tilt scan and the actual video measurement caveat."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;base=root/'runs/laser-tilt-20261009';out=base/'magnet_moment_report';out.mkdir(exist_ok=True)
targets=json.loads((root.parent/'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())
recheck=json.loads((root.parent/'cv/new_video_review/disk_rpm/measurement_recheck.json').read_text())
all_records=[];screen=[];validation=[]
for path in sorted(base.glob('magnet_moment_scan*/summary.json')):
    rows=json.loads(path.read_text());all_records.extend(rows)
    if any(tag in path.parent.name for tag in ['video','halfdt']):validation.extend(rows)
    else:screen.extend(rows)
production=json.loads((out/'amp8_production/production_600s_summary.json').read_text())
selected={'magnet_moment_scale':3.04,'magnet_beta':.056999326368103*3.04,'magnet_moment_A_m2':4.222662956003016*3.04,'tilt_rad':.5*np.arctan(4/930),'tilt_deg':float(np.degrees(.5*np.arctan(4/930))),'laser_offset_mm':4,'reflected_path_mm':930,'mu_low':1e-5,'mu_high':1e-4,'lambda':.004,'v0':-.075,'blowing':False,'production':production}
data={'screen_count':len(screen),'all_records':all_records,'selected_trial':selected,'video_measurement_recheck':recheck,'video_targets':targets,'raw_amplitude_target_is_provisional':True,'laser_allowed_offset_mm':[4,6],'laser_allowed_reflected_path_mm':[870,930],'steel_material_and_geometry_changed':False,'production_changed':True}
(out/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(out/'selected_trial.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
ax=axes[0]
for lam in [.004,.006,.008,.01]:
    rows=[r for r in screen if r['status']=='ok' and r['rolling_mu_high']==.0001 and r['rolling_lambda']==lam and abs(r['tilt_rad']-.002777749200346311)<1e-10]
    # One point per repeated physical parameter setting.
    rows=list({r['magnet_moment_scale']:r for r in rows}.values());rows.sort(key=lambda r:r['magnet_moment_scale'])
    ax.plot([r['magnet_moment_scale'] for r in rows],[r['metrics']['mean_orbit_period_s'] for r in rows],'o-',ms=4,label=f'lambda={lam:g}, tilt=0.159 deg')
for t in targets:ax.axhline(t['ball_period_target_s'],color='gray',ls='--',lw=.8)
ax.set(xlabel='Magnet moment / guessed reference',ylabel='Ball period (s)',title='Moment response is not monotonic',ylim=(0,9));ax.grid(alpha=.2);ax.legend(fontsize=8)
ax=axes[1];rows=[r for r in screen if r['status']=='ok' and 'mean_orbit_period_s' in r.get('metrics',{})]
im=ax.scatter([r['metrics']['mean_orbit_period_s'] for r in rows],[r['metrics']['first_harmonic_relative_half_amplitude_percent'] for r in rows],c=[np.degrees(r['tilt_rad']) for r in rows],s=22,cmap='viridis',alpha=.7)
for t,q in zip(targets,recheck):ax.scatter(t['ball_period_target_s'],q['raw_template_percent'],s=90,marker='X',label=f"Video {t['video']} (image-space target)")
b=production['bounded_assessment'];ax.scatter(b['mean_window_period_s'],b['mean_window_amplitude1_percent'],s=150,marker='*',c='red',label='Compiled trial at 200 RPM')
ax.set(xlabel='Ball period (s)',ylabel='Speed first-harmonic half-amplitude / mean (%)',title='Image-space targets are not yet physical calibration',xlim=(2.1,3.7),ylim=(0,20));ax.grid(alpha=.2);ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Tilt (deg)')
fig.savefig(out/'parameter_scan.png',dpi=160);plt.close(fig)
lines=['# 磁矩、滚阻、激光倾角联合扫描','',
f'本轮共筛选 {len(screen)} 组参数，另做两视频转速、步长减半和正式动画模块的长期复核。磁矩原值是估计值，允许改变；球的 Q235 参数、磁化参考值、球径、磁铁尺寸、磁球中心间距均未改动。','',
'## 已重新确认的真实视频数据','',
'| 视频 | 公转周期 s | 磁盘 RPM | 原图模板半振幅 | 独立圆检测半振幅 | 椭圆仿射模板 |','|---|---:|---:|---:|---:|---:|']
for t,q in zip(targets,recheck):lines.append(f"| {t['video']} | {t['ball_period_target_s']:.6f} | {t['disk_rpm']:.4f} | {q['raw_template_percent']:.2f}% | {q['raw_circle_percent']:.2f}% | {q['affine_template_percent']:.2f}% |")
lines += ['', '振幅定义：切向速度的一阶相位谐波半振幅/拟合均速，不是峰峰值，也不是未滤波极值。0.3 s 平滑；首尾各去掉15帧。原图和仿射结果分别约 6.3%/7.7%；8%近似长视频，不是两视频统一确认的物理振幅。','纸面单应得到约 16–17%，但标定未通过全部校验；原图也含投影误差，不能以8%宣称物理速度真值或据此唯一标定磁矩。球离纸5 cm、镜头离纸30 cm给出的5/6尺度修正不改变相对振幅。','',
'## 激光约束与持续性','',
'偏移允许4–6 mm，反射光程允许870–930 mm。盘面倾角取半角 atan(offset/path)/2，对应0.12321597°–0.19756852°。这是允许范围而非统计置信区间。','用户确认拍摄后仍稳定绕圈很多分钟，后段持续时长、周期、振幅未录制；因此排除径向逃逸的短暂高速轨道。20分钟是本次模拟验证窗口，不是实验实测时长。','',
'## 两种试验候选','',
'原滚阻 λ=0.006、μ∞=1e-4 下，将磁矩增至3.025倍、保持名义0.159°倾角：两视频转速下通过1200 s检查，周期约2.480/2.453 s，相对视频偏差+2.6%/-1.3%，一阶半振幅约15%。它较好解释周期，但不符合暂定8%振幅。',
'为尝试原图约8%的目标，当前动画改用磁矩3.04倍（每块约12.83690 A·m²，Beta=0.17327795215903312），λ=0.004、μ∞=1e-4、μ低=1e-5，倾角取允许下限0.12321597°（4 mm/930 mm）。v0=-0.075，无吹气，磁盘默认200 RPM。',
f"正式模块1200 s复核：平均周期{b['mean_window_period_s']:.4f} s，平均半振幅{b['mean_window_amplitude1_percent']:.2f}%，半径{b['radius_mean_mm']:.3f} mm。它是可观看的试验参数，不是已完成的双视频定量拟合。",'',
'## 实际视频转速复核（8%试验候选）','',
'| 视频 | 实测周期 s | 模拟周期 s | 周期偏差 | 模拟半振幅 | 原图模板半振幅 | 有界检查 |','|---|---:|---:|---:|---:|---:|---|']
for r in validation:
    if '_amp8video' not in r['run_folder']:continue
    vid='161858' if '161858' in r['run_folder'] else '162555';t=next(t for t in targets if t['video']==vid);q=next(q for q in recheck if q['video']==vid);v=r['bounded_assessment'];err=100*(v['mean_window_period_s']/t['ball_period_target_s']-1)
    lines.append(f"| {vid} | {t['ball_period_target_s']:.4f} | {v['mean_window_period_s']:.4f} | {err:+.2f}% | {v['mean_window_amplitude1_percent']:.2f}% | {q['raw_template_percent']:.2f}% | {'通过' if v['bounded_stationary'] else '未通过'} |")
lines += ['', '该组仍未同时满足两段视频的原图振幅；尤其短视频偏高。参数附近有接触状态与拍频变化，因此不能只用200 RPM下单个数字代替双视频验证。','半步长复核仍得到约2.396 s、8.24%并持续有界，与正式/隔离模块结果接近；有拍频时使用连续100 s窗口，避免末8圈采样偏差。','',
'## 稳定性与物理限制','',
'严格末五圈周期跨度<1%的旧判据会把有界拍频运动也标成未稳态；长视频本身有明显圈间变化。本轮保留该标记，另外检查连续三个100 s窗口：平均半径跨度<0.5 mm，均速/均周期跨度<2%，径向斜率绝对值<0.001 mm/s，逆转比例>99%。这是工程判据，允许有界准周期运动，不证明无限时稳定。',
f"当前候选末段滚动比例约{production['metrics']['rolling_fraction']*100:.1f}%，主要是滑动；接触状态尚无实验验证，因此匹配周期不能直接证明真实磁矩。球的不可逆JA内态已随磁矩重新准备，未沿用旧内态。磁矩缩放同时作用于场与梯度；固定内态的力分解分别符合二次/一次缩放检查。",
'倾斜力仍是小角度 theta*(gamma_g-Fmag_z)，磁间距固定；没有重算倾斜后位置相关间距。两个视频的复核保持同一无量纲滚阻/过渡自转率，物理阻尼c与磁盘角速度对应，不能称为严格的物理c联合估计。',
'正式CPU、相关扫描模块均已重新编译，12项本地检查通过；实际动画5帧更新与图像渲染通过。最新释放初态记录在 ../animation_initial_state_200rpm_rolling.json。','',
'启动当前试验动画（项目根目录）：','```powershell',"& 'C:\\Users\\38359\\miniconda3\\envs\\daily\\python.exe' .\\src2d\\animate_one_trajectory.py --fast --fps 60 --steps-per-frame 350",'```','',
'## 扫描明细','',
'“严格稳态”沿用末五圈判据；“否”不必然代表逃逸，详见有界检查。失败原因、轨迹、编译/LUT摘要在 results.json。','',
'| 磁矩倍数 | μ高速 | λ | 倾角° | 状态 | 严格稳态 | 周期s（末8圈） | 半振幅%（末8圈） |','|---:|---:|---:|---:|---|---|---:|---:|']
for r in screen:
    m=r.get('metrics',{});period=f"{m['mean_orbit_period_s']:.4f}" if 'mean_orbit_period_s' in m else '—';amp=f"{m['first_harmonic_relative_half_amplitude_percent']:.2f}" if 'first_harmonic_relative_half_amplitude_percent' in m else '—'
    lines.append(f"| {r['magnet_moment_scale']:g} | {r['rolling_mu_high']:g} | {r['rolling_lambda']:g} | {np.degrees(r['tilt_rad']):.5f} | {r['status']} | {'是' if m.get('steady_free_orbit') else '否'} | {period} | {amp} |")
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'screen_count':len(screen),'production_assessment':b,'report':str(out/'README.md')},indent=2))
