"""Material-only periodic JA response; no mechanical trajectory simulation."""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import radial_centripetal_100gauss as audit
import langevin_force_audit as ja
a=audit.a

def source(radius):
 state=np.zeros(15);state[1]=radius*.001*a.K;state[5]=1
 phases=np.linspace(0,2*np.pi,1025);H=[];J=[]
 for phase in phases:
  state[0]=phase/4;d=a.m.material_diagnostics(state,a.P)
  H.append(d[:3]*a.common.MS)
  g=d[18:]*a.common.MS*a.K
  J.append([[g[0],g[1],g[2]],[g[1],g[3],g[4]],[g[2],g[4],g[5]]])
 H=np.array(H);J=np.array(J);H[-1]=H[0];J[-1]=J[0]
 return CubicSpline(phases,H,bc_type='periodic'),CubicSpline(phases,J,bc_type='periodic')

def material_mean(radius,full=False,step=.025,cycles=96):
 H,J=source(radius);period=2*np.pi
 # Phase, not physical mechanical time: this JA closure is rate independent.
 sol=solve_ivp(lambda u,q:ja.q_rhs(H(u%period),H(u%period,1),q,full),(0,cycles*period),np.zeros(3),rtol=1e-9,atol=1e-5,max_step=step,dense_output=True)
 assert sol.success,sol.message
 u=np.linspace((cycles-1)*period,cycles*period,2049);Q=sol.sol(u).T
 M=np.array([ja.constitutive(h,q,full)[1] for h,q in zip(H(u%period),Q)])
 F=a.common.MU0*a.common.V*np.einsum('nij,nj->ni',J(u%period),M)
 mean=np.trapezoid(F,u,axis=0)/period
 need=audit.MASS*(2*np.pi/audit.PERIOD)**2*radius*.001
 rec=dict(radius_mm=radius,model='full Langevin JA' if full else 'linear JA',mean_radial_force_N=float(mean[0]),required_centripetal_N=need,support_ratio=float(-mean[0]/need),periodic_memory_difference_A_m=float(np.linalg.norm(Q[-1]-Q[0])),cycles=cycles,max_phase_step=step)
 print(json.dumps(rec),flush=True);return rec

def write_report(result):
 path=audit.OUT/'REPORT.md';base=path.read_text(encoding='utf-8').split('\n\n## 固定机械位置/姿态的 JA 材料周期响应核对')[0]
 lines=['','## 固定机械位置/姿态的 JA 材料周期响应核对','',
 '只积分材料记忆，不积分机械轨迹。平均力在相对场相位上积分；本 JA 闭合为速率无关模型。固定姿态响应仍不等于滚动/自旋球的完整响应。计算 96 个场周期并取最后一个周期均值；R=70 mm 进一步使用 128 周期、相位步长减半核对。',
 '', '|半径/mm|材料响应|平均向内力/mN|向心需求/mN|材料记忆首尾差/A m⁻¹|','|---:|---|---:|---:|---:|']
 for r in result:lines.append(f"|{r['radius_mm']}|{r['model']}|{-r['mean_radial_force_N']*1000:.6f}|{r['required_centripetal_N']*1000:.6f}|{r['periodic_memory_difference_A_m']:.6g}|")
 relative=abs(result[-1]['mean_radial_force_N']/result[0]['mean_radial_force_N']-1)
 lines+=['',f'R=70 mm 的长周期/半步长平均力相对变化：{relative:.4g}。',
 '', '结论：在 65–75 mm 实验半径范围，径向磁梯度力均值不足以独自提供 2.49 秒运动所需的向心力；到约 80 mm 均值可略超过需求。这不足以证明实际球能稳定运行，也不足以说明 70 mm 向外偏离后必然飞出势阱。']
 path.write_text(base+'\n'+'\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
 a.config(audit.HEIGHT)
 result=[]
 for radius in [70,75,80,85]:
  result.append(material_mean(radius));(audit.OUT/'ja_material_mean.json').write_text(json.dumps(result,indent=2))
 result.append(material_mean(70,True));result.append(material_mean(80,True))
 result.append(material_mean(70,False,.0125,128))
 (audit.OUT/'ja_material_mean.json').write_text(json.dumps(result,indent=2))
 write_report(result)
