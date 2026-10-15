"""Report the measured-spin calibration and its basin limitations."""
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spin_damping_response as s

def main():
 rows=json.loads((s.OUT/'observed_spin_calibration.json').read_text())
 rows.sort(key=lambda r:r['tau_s'])
 taus=np.array([r['tau_s'] for r in rows]);w=np.array([r['spin_z_mean_rad_s'] for r in rows])
 f=PchipInterpolator(taus,w)
 limits=[brentq(lambda t:f(t)-2*np.pi/T,10,30) for T in [2,1]]
 result=dict(observation_spin_period_s=[1,2],inertia_kg_m2=s.INERTIA,tau_s_range=limits,c_z_N_m_s_range=[s.INERTIA/limits[1],s.INERTIA/limits[0]],lambda_z_range=[1/(s.a.OMEGA*limits[1]),1/(s.a.OMEGA*limits[0])],trial_tau_s=15,trial_c_z_N_m_s=s.INERTIA/15,trial_lambda_z=1/(s.a.OMEGA*15),method='PCHIP interpolation of 600 s, dt*=0.00025 low-spin branch simulations; conditional on current height, reference moment and JA baseline')
 (s.OUT/'calibration_result.json').write_text(json.dumps(result,indent=2))
 basins=json.loads((s.OUT/'calibrated_basins.json').read_text())
 fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 ax[0].plot(taus,2*np.pi/w,'o-',label='Low-spin continuation')
 ax[0].axhspan(1,2,color='tab:green',alpha=.15,label='Observed estimate')
 ax[0].set(xlabel='Free spin decay time (s)',ylabel='Mean spin period (s)',title='Nearly parked ball, original height')
 ax[0].legend();ax[0].grid(alpha=.25)
 for name,label in [('parked_fresh','Release from rest'),('parked_high_spin','Start from high spin')]:
  q=sorted([r for r in basins if r['seed_branch']==name],key=lambda r:r['tau_s'])
  ax[1].plot([r['tau_s'] for r in q],[r['spin_z_mean_rad_s'] for r in q],'o-',label=label)
 ax[1].plot(taus,w,'k:',label='Low-spin continuation')
 ax[1].axhspan(np.pi,2*np.pi,color='tab:green',alpha=.15)
 ax[1].set(xlabel='Free spin decay time (s)',ylabel='Mean normal-axis spin (rad/s)',title='Final 60 s mean after 600 s');ax[1].legend();ax[1].grid(alpha=.25)
 fig.savefig(s.OUT/'spin_damping_calibration.png',dpi=180)
 text=['# Z 轴粘性自旋阻尼：按 1–2 秒一圈标定','',
 '采用 $T_z=-c_z\\omega_z$，自由衰减时间 $\\tau_z=I/c_z$；无量纲角加速度新增 $-\\lambda_z\\omega_z^*$，其中 $\\lambda_z=1/(\\Omega\\tau_z)$。阻尼始终独立作用于 Z 轴，包括关闭平面滚动阻力时；耗散功率为 $-c_z\\omega_z^2\\le0$。',
 '',f'球转动惯量 {s.INERTIA:.9g} kg m²，参考转速 {s.a.OMEGA:.9g} rad/s。保持参考磁矩、原 Q235 参数和原高度 39.62 mm。',
 '', '## 低自旋分支：600 秒，半步长验证','', '|自由衰减时间/s|平均自旋/rad s⁻¹|自旋周期/s|阻尼系数/N m s|','|---:|---:|---:|---:|']
 for r in rows:text.append(f"|{r['tau_s']:g}|{r['spin_z_mean_rad_s']:.5f}|{r['spin_period_s']:.5f}|{r['c_z_N_m_s']:.5g}|")
 text += ['',f"插值估计：1–2 秒一圈对应 τ≈{limits[0]:.2f}–{limits[1]:.2f} s，c≈{result['c_z_N_m_s_range'][0]:.3g}–{result['c_z_N_m_s_range'][1]:.3g} N m s，λ≈{result['lambda_z_range'][0]:.4g}–{result['lambda_z_range'][1]:.4g}。可先试 τ=15 s，即 c={s.INERTIA/15:.5g}、λ={1/(s.a.OMEGA*15):.6g}。边界值为插值估计，未单独精确求根验证。",'', '## 初态检查','', '|初态|τ/s|自旋/rad s⁻¹|轨道周期/s|稳定轨道判据|','|---|---:|---:|---:|---|']
 for r in basins:text.append(f"|{r['seed_branch']}|{r['tau_s']:g}|{r['spin_z_mean_rad_s']:.5f}|{r['period_s'] if r['period_s'] else '几乎不走'}|{r['stationary']}|")
 text += ['','稳定判据比较最后三个 60 秒窗口的轨道角速度和平均半径；不构成严格吸引子证明。所有标定运行到 600 秒，归一化步长 0.00025。只有最终低自旋分支用于上面的阻尼反算。',
 '', '## 验证与解释','', '自由衰减测试：τ=1 s、无有效磁驱动时，3 秒后应为 10 exp(−3) rad/s；相对误差 2.85×10⁻¹⁰。拒绝负值、NaN 和无穷阻尼。零阻尼回归重现原有轨道周期。',
 '', '1–3 秒仅为此前的强阻尼敏感性测试，不是实验估计。观测自旋速度本身不能独立确定阻尼；上述数值依赖模型给出的磁驱动力矩。场强、磁滞参数或接触条件改变时，需要重新标定。高速自旋态是否被消除必须检查初态；不能仅凭存在一个低自旋解就认定所有高速分支消失。',
 '', '改动在独立试验版中，正式模型的参考默认参数没有用这个未实测的系数覆盖。结果不支持继续把之前无 Z 阻尼模型所需的约 280 G 当作唯一磁场估计。']
 (s.OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
