"""Full LUT-radius mean-force scan, fixed geometry/attitude, JA memory only."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radial_centripetal_100gauss as c
a=c.a;OUT=c.OUT/'all-radius';OUT.mkdir(exist_ok=True)

def material(radius,points=1024,cycles=96):
 a.config(c.HEIGHT)
 state=np.zeros(15);state[5]=1;state[1]=np.clip(radius*.001*a.K,.8,10.)
 dt=np.pi/(2*points)
 state=a.m.magnetize_held_ball(state,dt,cycles*points,a.P)
 q0=state[12:15].copy();forces=[];b2=[];rev=[]
 for _ in range(points):
  diag=a.m.material_diagnostics(state,a.P)
  forces.append(diag[6]*c.MASS*a.OMEGA**2/a.K)
  b2.append(np.dot(diag[:3],diag[:3])*(a.common.MS*c.MU0)**2)
  # Same gradient, reversible M=chi_eff H, evaluated independently of Q.
  grad=diag[18:]*a.common.MS*a.K;H=diag[:3]*a.common.MS
  rev.append(c.MU0*a.common.V*c.CHI*np.dot(grad[:3],H))
  state=a.m.magnetize_held_ball(state,dt,1,a.P)
 return dict(radius_mm=radius,mean_Fr_JA_N=float(np.mean(forces)),mean_Fr_Q0_N=float(np.mean(rev)),required_centripetal_N=c.MASS*(2*np.pi/c.PERIOD)**2*radius*.001,field_rms_G=float(np.sqrt(np.mean(b2))*1e4),memory_period_difference_dimensionless=float(np.linalg.norm(state[12:15]-q0)),phase_samples=points,material_cycles=cycles)

def worker(radius):
 path=OUT/f'r{radius:.4f}.json'
 if path.exists():return json.loads(path.read_text())
 rec=material(radius);path.write_text(json.dumps(rec,indent=2));return rec

def report(records,validation):
 data=np.array([[r[k] for k in ['radius_mm','mean_Fr_JA_N','mean_Fr_Q0_N','required_centripetal_N','field_rms_G']] for r in records])
 spl=CubicSpline(data[:,0],data[:,1]);mass=c.MASS;orb=2*np.pi/c.PERIOD
 def balance(r):return -spl(r)-mass*orb**2*r*.001
 roots=[]
 for lo,hi in zip(data[:-1,0],data[1:,0]):
  if balance(lo)*balance(hi)<0:roots.append(brentq(balance,lo,hi))
 supported=[]
 for lo,hi in zip([data[0,0]]+roots,roots+[data[-1,0]]):
  if balance((lo+hi)/2)>0:supported.append([lo,hi])
 ratio=-data[:,1]/(mass*data[:,0]*.001)
 best=np.argmax(ratio);peak=minimize_scalar(lambda r:spl(r)/(mass*r*.001),bounds=(data[max(0,best-3),0],data[min(len(data)-1,best+3),0]),method='bounded')
 minperiod=2*np.pi/np.sqrt(-spl(peak.x)/(mass*peak.x*.001))
 zero=[]
 for lo,hi in zip(data[:-1,0],data[1:,0]):
  if spl(lo)*spl(hi)<0:zero.append(brentq(spl,lo,hi))
 rootchecks=[material(float(r),2048,128) for r in roots]
 summary=dict(radius_range_mm=[14,175],spacing_mm=.25,total_radius_points=len(records),period_s=c.PERIOD,height_mm=c.HEIGHT,field_definition='100 G RMS at R=70 mm; same height and magnet moment at every radius',mean_force_balance_roots_mm=roots,supported_radius_intervals_mm=supported,zero_mean_radial_force_radii_mm=zero,shortest_period_by_mean_magnetic_force_s=float(minperiod),radius_for_shortest_period_mm=float(peak.x),validation=validation,root_checks=rootchecks,assumptions='Only fixed-position/fixed-attitude linear JA material memory is integrated; no mechanical motion or self-spin integration. Phase-averaged radial magnetic force alone is compared with mr(2pi/T)^2. Mean sufficiency does not prove stable rolling motion.')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));np.savetxt(OUT/'all_radius_force.csv',data,delimiter=',',header='radius_mm,mean_Fr_JA_N,mean_Fr_Q0_N,required_centripetal_N,local_field_rms_G',comments='')
 fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for panel in ax:
  panel.plot(data[:,0],-data[:,1]*1000,label='Mean inward force: linear JA')
  panel.plot(data[:,0],-data[:,2]*1000,ls=':',label='Reversible Q=0 reference')
  panel.plot(data[:,0],data[:,3]*1000,ls='--',label='Centripetal demand, T=2.49 s')
  panel.axhline(0,color='.5',lw=.7)
  for lo,hi in supported:panel.axvspan(lo,hi,color='tab:green',alpha=.15)
  panel.set(xlabel='Radius (mm)',ylabel='Force (mN)');panel.grid(alpha=.25);panel.legend(fontsize=8)
 ax[0].set(title='All available radii, same source height',xlim=(14,175))
 ax[1].axvspan(65,75,color='.5',alpha=.1);ax[1].set(title='Mean-force balance near the magnetic ring',xlim=(60,100),ylim=(-2.5,5))
 fig.savefig(OUT/'all_radius_force.png',dpi=180)
 lines=['# 100 G 场：全半径平均磁梯度力核算','',
 f'扫描半径 14–175 mm、步长 0.25 mm，共 {len(records)} 点。保持磁源高度 {c.HEIGHT:.5f} mm 和参考磁矩不变；100 G 是 R=70 mm 处 RMS，不把其他半径重新调成 100 G。',
 '', '固定球心位置与姿态，仅计算 JA 材料记忆的周期响应及磁场相位平均，不模拟机械轨迹。每个半径先经历 96 个磁场周期，再平均一个周期（1024 个相位点）。同时给出 Q=0 可逆磁化参考。',
 '', f'2.49 秒一圈所需向心力为 Fc=m r(2π/2.49)²，球质量 {mass*1000:.6f} g。',
 '', f'JA 平均磁力满足向心需求的半径区间：{supported} mm。力平衡半径根：{roots} mm。',
 '', f'平均磁力约束下允许的最短周期：{minperiod:.6f} 秒，出现在 R≈{peak.x:.5f} mm。此值只由径向平均力决定，不是模拟得到的实际运动周期。',
 '', '|半径/mm|当地场强 RMS/G|JA 平均向内力/mN|2.49 s 向心需求/mN|','|---:|---:|---:|---:|']
 for radius in [14,30,50,65,70,75,78,80,82,85,90,100,125,150,175]:
  r=next(r for r in records if r['radius_mm']==radius)
  lines.append(f"|{radius}|{r['field_rms_G']:.3f}|{-r['mean_Fr_JA_N']*1000:.6f}|{r['required_centripetal_N']*1000:.6f}|")
 lines+=['', '正的“向内力”表示指向盘中心；负值表示平均磁力向外。图中绿色为平均磁力超过向心需求的区间。',
 '', '力平衡根由完整 0.25 mm 网格的样条插值得到，随后在根处使用 2048 相位点、128 个材料周期复核。另与独立 SI 材料方程及加密相位积分核对，具体误差见 summary.json。',
 '', '平均径向磁力够用不代表机械稳定轨道已存在；未将接触反力、自旋力矩、实际滚动姿态历史、径向振荡或瞬时场波动纳入。本核算不能证明 ball 会稳定公转，也不能因 R=70 mm 磁力不足就断言它必然逃出整个势阱。65–75 mm 的原实验轨道范围与向外偏移后的可满足区域应分开判断。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':
 checks=[];old=json.loads((c.OUT/'ja_material_mean.json').read_text())
 for radius in [70,80]:
  low=material(radius);high=material(radius,2048,128)
  reference=next(r for r in old if r['radius_mm']==radius and r['model']=='linear JA')
  checks.append(dict(radius_mm=radius,relative_error_vs_independent_SI=abs(low['mean_Fr_JA_N']/reference['mean_radial_force_N']-1),relative_phase_refinement_error=abs(high['mean_Fr_JA_N']/low['mean_Fr_JA_N']-1),low=low,high=high))
 print(json.dumps(checks),flush=True)
 records=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for i,r in enumerate(pool.map(worker,np.arange(14,175.0001,.25).tolist())):
   records.append(r)
   if (i+1)%100==0:print(f'Computed {i+1}/645 radii',flush=True)
 (OUT/'all_radius_records.json').write_text(json.dumps(records,indent=2));report(records,checks)
