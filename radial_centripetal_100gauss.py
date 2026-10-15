"""Phase quadrature of radial gradient force; no mechanical integration."""
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import height_response_audit as a
OUT=a.ROOT/'runs/radial-centripetal-100gauss-20261013';OUT.mkdir(exist_ok=True)
HEIGHT=39.890150002283924;PERIOD=2.49;MASS=a.common.MASS
MU0=a.common.MU0;V=a.common.V;D=(a.common.BASE[0]-1)/(a.common.BASE[0]+2)
CHI=3*D;PHASE=np.arange(1024)*np.pi/2/1024

def mean_force(radius_mm):
 state=np.zeros(15);state[5]=1;state[1]=radius_mm*.001*a.K
 force=[];b2=[]
 for phi in PHASE:
  state[0]=phi;diag=a.m.material_diagnostics(state,a.P)
  force.append(diag[6]*MASS*a.OMEGA**2/a.K)
  b2.append(np.dot(diag[:3],diag[:3])*(a.common.MS*MU0)**2)
 return float(np.mean(force)),float(np.mean(b2))

def main():
 a.config(HEIGHT)
 curve=[]
 for r in np.arange(60,105.0001,.25):
  fr,b2=mean_force(r);required=MASS*(2*np.pi/PERIOD)**2*r*.001
  curve.append([r,fr,required,np.sqrt(b2)*1e4])
 curve=np.array(curve)
 def balance(r):return -mean_force(r)[0]-MASS*(2*np.pi/PERIOD)**2*r*.001
 roots=[]
 for left,right in zip(curve[:-1],curve[1:]):
  if balance(left[0])*balance(right[0])<0:roots.append(brentq(balance,left[0],right[0],xtol=1e-7))
 peak=minimize_scalar(lambda r:mean_force(r)[0]/(r*.001),bounds=(70,100),method='bounded')
 critical_T=2*np.pi/np.sqrt(-mean_force(peak.x)[0]/(MASS*peak.x*.001))
 checks=[]
 for r in [65,70,75,80,85,90]:
  fr,b2=mean_force(r);eps=.01
  derivative=(mean_force(r+eps)[1]-mean_force(r-eps)[1])/(2*eps*.001)
  independent=V*CHI/(2*MU0)*derivative
  need=MASS*(2*np.pi/PERIOD)**2*r*.001
  checks.append(dict(radius_mm=r,field_rms_G=np.sqrt(b2)*1e4,mean_radial_force_N=fr,inward_force_N=-fr,required_centripetal_N=need,support_ratio=-fr/need,d_mean_B_squared_dr_T2_per_m=derivative,independent_energy_gradient_force_N=independent,relative_force_check_error=abs(independent-fr)/max(abs(fr),1e-12)))
 rec=dict(mass_kg=MASS,ball_radius_mm=6,material_mu_r=float(a.common.BASE[0]),effective_sphere_susceptibility=CHI,height_mm=HEIGHT,field_definition='100 G RMS at R=70 mm at ball-center height, fixed reference source moment',orbital_period_s=PERIOD,magnetic_assumption='Reversible induced magnetization Q=0, sphere demagnetization included. M=3(mu_r-1)/(mu_r+2) H. No mechanical or self-spin trajectory is integrated.',force_formula='mean Fr = V chi_eff/(2 mu0) d<|B|^2>/dr',centripetal_formula='Fc=m r (2pi/T)^2',checks=checks,mean_force_balance_roots_mm=roots,supported_radius_interval_mm=roots if len(roots)==2 else None,min_period_by_magnetic_mean_force_alone_s=float(critical_T),critical_radius_mm=float(peak.x),scope='Comparison of magnetic gradient force alone against centripetal demand; contact force, magnetic torque, hysteresis-memory history, phase-dependent excursions and dynamical stability are not included. A mean balance is not proof of a stable orbit.')
 (OUT/'summary.json').write_text(json.dumps(rec,indent=2));np.savetxt(OUT/'radial_force_curve.csv',curve,delimiter=',',header='radius_mm,mean_Fr_N,required_centripetal_N,field_rms_G',comments='')
 lines=['# 100 G：平均径向梯度力与 2.49 秒向心需求','',
 f'球质量 {MASS*1000:.6f} g，磁体中心至球心高度 {HEIGHT:.5f} mm；100 G 指 R=70 mm 处的场模 RMS，其他半径场强不同。参考磁矩不变。',
 '', '只计算磁场相位平均，不进行机械运动或自旋模拟。采用带球形退磁的可逆磁化 Q=0：χeff=3(μr−1)/(μr+2)，Fr=Vχeff/(2μ0) ∂r〈B²〉。这是一项明确的材料状态假设，不是完整磁滞运动平均。',
 '', '|半径/mm|当地场强 RMS/G|平均向内磁力/mN|2.49 s 所需向心力/mN|磁力/需求|','|---:|---:|---:|---:|---:|']
 for q in checks:lines.append(f"|{q['radius_mm']}|{q['field_rms_G']:.3f}|{q['inward_force_N']*1000:.5f}|{q['required_centripetal_N']*1000:.5f}|{q['support_ratio']:.4f}|")
 lines+=['',f'平均力平衡的半径根：{roots} mm。磁力单独满足向心需求的最短周期估计 {critical_T:.6f} s，出现在 R≈{peak.x:.4f} mm。',
 '', '径向磁力同时用现有解析梯度和独立的磁场能量有限差分核对；后者由场强平方的相位均值求径向导数。外向为正，表中“向内磁力”为 −〈Fr〉，负值表示磁力向外。',
 '', '在约 70 mm 的实验轨道上，平均磁梯度力明显不足以单独承担目标向心力；但径向外移可能遇到更强的向内梯度力。因此，不能从 70 mm 的不足直接推出球必然飞出整个势阱。若根位于实验轨道范围之外，也不能称为复现了实验轨道。',
 '', '平均径向磁力不是总径向力。本计算未加接触反力、磁力矩耦合和完整磁滞记忆，不证明所求半径有稳定机械运动。纯滚动时也不能无条件把磁力与 mrω² 的平衡当作完整运动方程；应连同自旋力矩和接触力核算。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 ax.plot(curve[:,0],-curve[:,1]*1000,label='Mean inward magnetic force (Q=0)')
 ax.plot(curve[:,0],curve[:,2]*1000,label='Required centripetal force, T=2.49 s')
 ax.axvspan(65,75,color='.5',alpha=.12,label='Observed radius range 65–75 mm')
 if len(roots)==2:ax.axvspan(*roots,color='tab:green',alpha=.15,label='Magnetic mean force exceeds demand')
 ax.axhline(0,color='.5',lw=.7);ax.set(xlabel='Radius (mm)',ylabel='Force (mN)',title='100 G at R=70 mm: radial mean-force comparison');ax.legend(fontsize=8);ax.grid(alpha=.25)
 fig.savefig(OUT/'radial_force_vs_centripetal.png',dpi=180)
 print(json.dumps(rec,indent=2))

if __name__=='__main__':main()
