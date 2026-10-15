"""Report tilt-adjusted matches and document the actual sliding force."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_mu_moment import OUT,TILT
d=json.loads((OUT/'tilt_matched.json').read_text())
fits=[r for r in d['fits'] if 'metrics' in r]
lines=['# 倾角可调的高速滚阻—磁矩比较（2026-10-10）','',
'重新按视频原图处理的暂定6–8%范围匹配，一阶半振幅中心目标取7%。周期继续匹配旧3.04倍磁矩基线的2.396274687 s；本轮不是两视频周期的联合拟合。固定200 RPM、全部尺寸和Q235参数、滑动摩擦Mu=0.02、低速滚阻1e-5、线性滚阻lambda=0.004、启动v0=-0.075，无吹气。只改变高速滚阻、磁矩与倾角。',
'','每组均240秒筛选后重新运行600秒，验证步长为0.00025。统计取末120秒，另检查末三个60秒窗口的周期、半径和转向是否保持稳定。接受标准为周期差<0.1%、一阶半振幅在6–8%之间。',
'','| 高速滚阻 | 磁矩倍数 | 磁矩 / A·m² | 相对旧基线磁矩变化 | 倾角 / 度 | 周期 / s | 一阶半振幅 | 周期差 | 等效930 mm激光偏移 / mm | 接受 |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in fits:
    m=r['metrics'];lines.append(f"| {r['mu_high']:g} | {r['moment_scale']:.6f} | {r['moment_A_m2']:.6f} | {r['moment_change_percent']:+.3f}% | {r['tilt_deg']:.6f} | {m['period_s']:.6f} | {m['amplitude_percent']:.3f}% | {r['period_error_percent']:+.4f}% | {r['equivalent_laser_offset_at_930mm']:.3f} | {'通过' if r['accepted'] else '未通过'} |")
lines+=['','## 倾角范围的物理含义','',
'原先激光允许值4–6 mm、反射后光程870–930 mm，对应倾角0.123216°–0.197569°。此次用户明确允许进一步调倾角，因此将其作为探索变量；表中低于0.123216°的解需要重新测量调平状态，不能称为落在原激光允许范围内的拟合。6–8%来自原图/仿射视频处理，透视精密标定尚未完成，因此仍为暂定物理目标。',
'','## 滑动阻力','',
'当前为常系数库仑摩擦，Mu=0.02。无量纲接触点滑移速度s=(vx-alpha*omega_y, vy+alpha*omega_x)，法向载荷N=gamma_g-Fmag_z。滑动时f=-Mu*N*s/|s|；平动方程中的空气阻力为-delta*|v|*v。接触摩擦的角加速度贡献为(f_y,-f_x,0)/(eta*alpha)。零滑移时由接触事件算法切换为纯滚动所需的静摩擦，摩擦上限仍用同一Mu。',
'','有量纲写法：F_slide=-Mu*F_N*v_slip/|v_slip|，F_N≈m*g-Fmag_z（当前小倾角近似）；v_slip=v+a*(-omega_y,omega_x)。空气阻力F_air=-(rho_air*Cd*pi*a^2/2)*|v|*v。',
'','此前扫描的mu_high属于独立的附加滚动转矩项：tau_roll=-mu_roll(w)*F_N*a*omega_xy/sqrt(w^2+epsilon^2)-c*omega_xy；mu_roll(w)=mu_high+(mu_low-mu_high)*exp(-(w/wc)^2)。因此模型目前没有“高速滑动摩擦系数下降”的滑移速度依赖，速度依赖是在滚动阻力项中。',
'','正式构建参数未修改。全部结果和完整轨迹见tilt_matched.json与文件名带tilt后缀的JSON/NPZ。']
(OUT/'TILT_MATCH_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
mus=[r['mu_high'] for r in fits]
axs[0].plot(mus,[r['moment_scale'] for r in fits],'o-',color='#7d4aa8');axs[0].axhline(3.04,color='gray',ls='--')
axs[0].set(ylabel='Moment / original reference',title='Matched period, tilt adjustable')
axs[1].plot(mus,[r['tilt_deg'] for r in fits],'o-',color='#267da1');axs[1].axhspan(np.degrees(TILT),np.degrees(.5*np.arctan(6/870)),alpha=.13,color='gray',label='Original laser range')
axs[1].set(ylabel='Tilt (degrees)',title='Required tilt');axs[1].legend(fontsize=8)
axs[2].plot(mus,[r['metrics']['amplitude_percent'] for r in fits],'o-',color='#2e9375');axs[2].axhspan(6,8,color='#2e9375',alpha=.12)
axs[2].set(ylabel='First speed harmonic / mean (%)',title='Target range: 6-8%',ylim=(5,9))
for ax in axs:ax.set_xscale('symlog',linthresh=2.5e-5);ax.set_xlim(0,.012);ax.set_xlabel('High-speed rolling coefficient');ax.grid(alpha=.2)
fig.savefig(OUT/'tilt_comparison.png',dpi=170);plt.close(fig)
print(json.dumps([{'mu':r['mu_high'],'m':r['moment_scale'],'deg':r['tilt_deg'],'T':r['metrics']['period_s'],'A':r['metrics']['amplitude_percent'],'accepted':r['accepted']} for r in fits],indent=2))
