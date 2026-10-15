"""Report measured targets, screening, and long-run validation without claiming a fit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root=Path(__file__).resolve().parent
base=root/'runs/laser-tilt-20261009'
out=base/'high_speed_fit_report';out.mkdir(exist_ok=True)
targets=json.loads((root.parent/'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())
screen=[]
for group in ['high_speed_fit','high_speed_fit_boundary','high_speed_fit_fine']:
    screen.extend(json.loads((base/group/'screening.json').read_text()))
validation=[]
for video in ['161858','162555']:
    validation.extend(json.loads((base/'high_speed_fit_validation'/f'{video}.json').read_text()))
near=[json.loads(p.read_text())[0] for p in base.glob('*_highfit_verify/summary.json')]
startup=json.loads((base/'high_speed_fit_startup/summary.json').read_text())
data={'targets':targets,'screening':screen,'validation':validation,'near_boundary_600s':near,'faster_startup_checks':startup,'production_config_changed':False}
data['post_video_observation']={'source':'User report, 2026-10-09','ball_kept_orbiting_for_many_minutes_after_recording':True,'duration_s':None,'unrecorded_period_and_amplitude':None,'interpretation':'A short-lived orbit is insufficient; exact later period and amplitude were not recorded. The 600 s checks are an engineering validation window, not a measured experimental duration.'}
(out/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
best=[r for r in validation if r['rolling_mu_high']==0 and r['rolling_lambda']==.006]
(out/'best_validated_candidate.json').write_text(json.dumps({'status':'Stable trial candidate; period still mismatched, not an experimental calibration','mu_low':1e-5,'mu_high':0,'lambda':.006,'v0':-.075,'blowing':False,'validation':best,'production_config_changed':False},indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
ax=axes[0]
ok=[s for s in screen if s['steady']]
other=[s for s in screen if s.get('period_s') and not s['steady']]
ax.scatter([s['period_s'] for s in ok],[s['radius_mm'] for s in ok],s=25,label='Screen: locally steady at 200 RPM')
ax.scatter([s['period_s'] for s in other],[s['radius_mm'] for s in other],s=28,marker='x',c='gray',label='Screen: still drifting')
for i,t in enumerate(targets):
    ax.plot([t['ball_period_target_s']]*2,[65,75],lw=3,label=f"Video {t['video']}: radius prior 65-75 mm")
for r in validation:
    m=r.get('metrics',{})
    if m.get('steady_free_orbit'):
        ax.scatter(m['mean_orbit_period_s'],m['radius_mean_mm'],s=65,marker='*',c='black')
ax.set(xlabel='Ball lap period (s)',ylabel='Mean orbit radius (mm)',title='Period and radius cannot yet be matched')
ax.grid(alpha=.2);ax.legend(fontsize=8)
fixed=sorted([s for s in screen if s['lambda']==.006 and s['steady']],key=lambda s:s['mu_high'])
ax=axes[1]
ax.plot([s['mu_high'] for s in fixed],[s['period_s'] for s in fixed],'o-',label='Only vary mu_high; lambda=0.006')
for t in targets:ax.axhline(t['ball_period_target_s'],ls='--',label=f"Video {t['video']}: {t['ball_period_target_s']:.3f} s")
ax.set(xlabel='High-speed coefficient mu_high',ylabel='Ball lap period (s)',title='Reducing mu_high to zero is insufficient')
ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
lines=['# 高速滚阻拟合复核（2026-10-09）','',
'结论：在本次扫描的被动滚阻参数、当前固定磁模型及启动条件下，尚未找到同时符合两个视频公转周期与估测轨道半径、并能长期稳定绕圈的参数。不能将振幅相近称为速度拟合成功。正式动画的已有参数暂未替换。','',
'用户补充：虽然视频没有拍十分钟，球在之后确实稳定绕圈很多分钟。这作为独立的定性持续性约束计入，短时间绕圈后逃逸的参数不能因符合视频时长就接受。确切持续时间未知，600 s 是本次验证窗口，不是实验实测时长；未录制部分的周期和振幅未知。','',
'## 实验目标','',
'| 视频 | 磁盘转速 RPM | 小球一圈 s | 完整圈数 | 单圈周期标准差 s |','|---|---:|---:|---:|---:|']
for t in targets:
    m=t['methods'][0];lines.append(f"| {t['video']} | {t['disk_rpm']:.4f} | {t['ball_period_target_s']:.6f} | {m['complete_laps']} | {m['lap_period_std_s']:.6f} |")
lines += ['', '两种独立捕球方法得到的平均周期差分别只有约 0.00012 s、0.00004 s。长视频使用约 60–107 s，排除手扶阶段。65–75 mm 是轨道半径先验估测，尚无通过校验的精确平面标定。周期标准差是圈间变化，不是测量置信区间。','',
'## 参数与扫描','',
f'200 RPM 下扫描 {len(screen)} 组 μ∞、λ：μ∞ 范围 0–0.003，λ 范围 0–0.006；不是完整连续参数空间的穷尽证明。低速系数固定 1e-5，过渡自转率固定 1.5，启动 v0=-0.075，无吹气。另测 v0=-0.125、-0.2 的四种启动组合。',
'球径 12 mm，磁铁直径 28 mm、厚 15 mm，磁球中心间距 39.62 mm，磁铁中心环半径 70 mm，激光推得倾角 0.1591533°。Q235 与磁化强度、几何均未调参。',
'λ=c/(I Ω盘)，这里两视频复核使用相同无量纲 λ 和过渡率，因此不是严格相同物理 c 的联合估计；转速差仅约 1%。倾斜力采用 theta*(gamma_g-Fmag_z)，固定磁间距，尚未做倾斜后的磁场几何重算。','',
'固定 λ=0.006 时，高速系数降至零仍只有约 3.826 s/圈；原动画 0.0001、0.006 约 4.127 s/圈。',
'同时降低 λ 后，200 RPM、μ∞=5e-5、λ=0.0054 的 360 s 筛选曾得到约 3.511 s/圈、74.90 mm；但两实际 RPM 的 600 s 复核均失败，约 511/575 s 径向漂移至 175 mm 扩展磁场表外。因此这组不能推荐为长期稳态。错误 status 6 表示径向磁场表越界，并非竖直起飞。','',
'## 两视频转速下 600 s 复核','',
'| 视频 | μ∞ | λ | 结果 | 周期 s | 周期偏差 | 半径 mm | 一阶速度半振幅/均速 |','|---|---:|---:|---|---:|---:|---:|---:|']
for r in validation:
    m=r.get('metrics',{});steady=m.get('steady_free_orbit',False)
    status='稳定' if steady else '未达稳态' if r['status']=='ok' else '径向越界'
    def val(key,fmt='.3f'):return format(m[key],fmt) if key in m else '—'
    err=f"{r['period_error_percent']:+.1f}%" if 'period_error_percent' in r else '—'
    lines.append(f"| {r['video']} | {r['rolling_mu_high']:g} | {r['rolling_lambda']:g} | {status} | {val('mean_orbit_period_s')} | {err} | {val('radius_mean_mm')} | {val('first_harmonic_relative_half_amplitude_percent')}% |")
lines += ['', '稳态判据：末五圈周期跨度 <1%，末五圈平均半径跨度 <0.5 mm，末段逆向旋转占比 >99%；判据只证明观测窗口稳定，不能证明无限时稳定。','',
'两视频转速均通过长期检查的较快候选：μ低=1e-5、μ∞=0、λ=0.006。高速库仑项为零但线性阻尼仍在；约 3.82 s/圈、73.8–73.9 mm，速度一阶半振幅约 9.6%，接近 10% 预期，但周期偏慢约 53–59%。详见 best_validated_candidate.json。','',
'## 速度振幅的限制','',
'视频速度半振幅对标定方式敏感：短视频约 6.2%（原图/椭圆仿射）或 16.2%（纸平面单应）；长视频约 7.6% 或 17.1%。纸平面单应未通过全部校验，这些数字只能作为处理方法差异的范围，不能当作精确实验拟合目标。球离纸 5 cm、镜头离纸 30 cm 对已正确标定的平行平面给出约 5/6 的长度/速度尺度修正，不改变相对振幅。','',
'只调损耗暂时无法完成当前周期目标；下一步应核验磁力径向约束与磁场/磁化模型，或检查倾斜导致的间距变化，再进行联合拟合。不能用负滚阻、外加吹气或只截取快速暂态掩盖差异。','',
'完整扫描、600 s 复核、启动检查、原始轨迹目录均记录在 results.json。']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(out)
