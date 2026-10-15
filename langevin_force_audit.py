"""Self-consistent nonlinear vector JA force at fixed magnet moment/geometry.

Uses the repository's isotropic irreversible projection closure, with full
Langevin anhysteretic magnetization and tensor differential susceptibility.
No mechanical orbit is re-integrated; field history is prescribed explicitly.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import material_response_audit as field
OUT=field.ROOT/'runs/langevin-force-20261013';OUT.mkdir(exist_ok=True)
MU0=4*np.pi*1e-7;MS=1200019.0724085916;AA=1514.216550463362;KP=5913.52056037147;C=.22415154978863677;AL=.0021323083849413885;N=1/3;D=N-AL;S=MS/(3*AA)
def langevin(x):
 if abs(x)<.05:return x/3-x**3/45+2*x**5/945-x**7/4725+2*x**9/93555
 return 1/np.tanh(x)-1/x
def langevin_prime(x):
 if abs(x)<.05:return 1/3-x*x/15+2*x**4/189-x**6/675+2*x**8/10395
 return 1/(x*x)-(1/np.sinh(x)**2 if abs(x)<350 else 0.)
def constitutive(H,Q,full):
 b=H+D*(1-C)*Q;bn=np.linalg.norm(b);u=b/bn if bn else np.array([1.,0,0])
 if not full:
  he=bn/(1+D*S);man=S*he*u;jp=jt=S;x=he/AA
 else:
  # Solve |He|+D Ms L(|He|/a)=|H+D(1-C)Q|. Positive D gives a unique root.
  he=bn/(1+D*S);lower=0.;upper=bn
  for _ in range(30):
   value=he+D*MS*langevin(he/AA)-bn
   if abs(value)<1e-9+1e-12*bn:break
   if value>0:upper=he
   else:lower=he
   new=he-value/(1+D*MS/AA*langevin_prime(he/AA))
   he=new if lower<new<upper else (lower+upper)/2
  x=he/AA;man=MS*langevin(x)*u;jp=MS/AA*langevin_prime(x);jt=MS*langevin(x)/he if he else S
 M=man-(1-C)*Q
 return he*u,M,u,jp,jt,x
def q_rhs(H,Hd,Q,full):
 he,M,u,jp,jt,x=constitutive(H,Q,full)
 def inv(v):
  parallel=(u@v)*u
  return (v-parallel)/(1+D*C*jt)+parallel/(1+D*C*jp)
 hed=inv(Hd);qn=np.linalg.norm(Q);active=Q@hed>0
 if active:
  n=Q/qn;z=qn/KP;w=inv(n);lam=D*(1-C)*z
  hed-=lam*w*(n@hed)/(1+lam*(n@w))
 qd=jt*hed+(jp-jt)*(u@hed)*u
 if active:qd-=z*n*(n@hed)
 return qd
def source(radius_mm,moment=1.):
 field.configure(field.BASE,moment);u=np.linspace(0,2*np.pi,513);H=[];J=[];state=np.zeros(15);state[5]=1;state[1]=radius_mm*.001*field.KSPACE
 for phase in u:
  state[0]=phase/4;d=field.diag(state);H.append(d[:3]*field.MS);g=d[18:]*field.MS*field.KSPACE
  J.append([[g[0],g[1],g[2]],[g[1],g[3],g[4]],[g[2],g[4],g[5]]])
 H=np.array(H);J=np.array(J);H[-1]=H[0];J[-1]=J[0]
 return CubicSpline(u,H,bc_type='periodic'),CubicSpline(u,J,bc_type='periodic')
def solve_case(radius_mm,full,moment=1.,rtol=2e-9,max_step=.06,cycles=24):
 spline,gradient=source(radius_mm,moment);period=2*np.pi;start=time.perf_counter()
 sol=solve_ivp(lambda t,q:q_rhs(spline(t%period),spline(t%period,1),q,full),(0,cycles*period),np.zeros(3),rtol=rtol,atol=2e-5,max_step=max_step,dense_output=True)
 if not sol.success:raise RuntimeError(sol.message)
 sample=np.linspace((cycles-1)*period,cycles*period,1025);Q=sol.sol(sample).T;H=spline(sample%period);J=gradient(sample%period);M=[];xs=[];res=[]
 for h,q in zip(H,Q):
  he,mag,u,jp,jt,x=constitutive(h,q,full);M.append(mag);xs.append(x);res.append(np.linalg.norm(he-h+D*mag))
 M=np.array(M);F=MU0*field.V*np.einsum('nij,nj->ni',J,M);torque=MU0*field.V*np.cross(M,H)
 rec=dict(radius_mm=radius_mm,model='Langevin' if full else 'linear',moment_scale=moment,force_rms_N=float(np.sqrt(np.mean(np.sum(F*F,axis=1)))),radial_force_rms_N=float(np.sqrt(np.mean(F[:,0]**2))),tangential_force_rms_N=float(np.sqrt(np.mean(F[:,1]**2))),normal_attraction_mean_N=float(-F[:,2].mean()),force_max_N=float(np.linalg.norm(F,axis=1).max()),radial_force_mean_N=float(F[:,0].mean()),magnetization_rms_A_m=float(np.sqrt(np.mean(np.sum(M*M,axis=1)))),max_M_over_Ms=float(np.linalg.norm(M,axis=1).max()/MS),max_He_over_a=float(max(xs)),torque_rms_N_m=float(np.sqrt(np.mean(np.sum(torque*torque,axis=1)))),periodic_Q_difference_A_m=float(np.linalg.norm(Q[-1]-Q[0])),constitutive_constraint_max_error_A_m=float(max(res)),nfev=sol.nfev,wall_s=time.perf_counter()-start,rtol=rtol,max_step=max_step,cycles=cycles)
 rows=np.c_[sample-(cycles-1)*period,H,M,F,torque,Q,np.array(xs)]
 if rtol==2e-9:np.savetxt(OUT/f'r{radius_mm:g}_m{moment:g}_{rec["model"]}.csv',rows,delimiter=',',header='field_phase_rad,Hx_A_m,Hy_A_m,Hz_A_m,Mx_A_m,My_A_m,Mz_A_m,Fx_N,Fy_N,Fz_N,taux_N_m,tauy_N_m,tauz_N_m,Qx_A_m,Qy_A_m,Qz_A_m,He_over_a',comments='')
 print(json.dumps(rec),flush=True);return rec,rows
def static_curve(full):
 radius=np.linspace(55,95,401);F=[];phase_state=np.zeros(15);phase_state[5]=1;field.configure(field.BASE)
 for r in radius:
  phase_state[1]=r*.001*field.KSPACE;q=field.diag(phase_state);h=q[:3]*field.MS;g=q[18:]*field.MS*field.KSPACE
  J=np.array([[g[0],g[1],g[2]],[g[1],g[3],g[4]],[g[2],g[4],g[5]]]);he,M,*_=constitutive(h,np.zeros(3),full);F.append(MU0*field.V*J@M)
 F=np.array(F);ix=np.flatnonzero((F[:-1,0]>0)&(F[1:,0]<=0))[0];root=np.interp(0,F[ix:ix+2,0][::-1],radius[ix:ix+2][::-1]);stiff=-(F[ix+1,0]-F[ix,0])/(.0001)
 return dict(model='Langevin' if full else 'linear',reversible_q0_equilibrium_radius_mm=float(root),stiffness_N_m=float(stiff),pure_rolling_frequency_hz=float(np.sqrt(stiff/(1.4*field.MASS))/(2*np.pi))),np.c_[radius,F]
if __name__=='__main__':
 records=[];series={}
 for r in [65,70,75,80]:
  for full in [False,True]:
   rec,rows=solve_case(r,full);records.append(rec);series[(r,full)]=rows
 # Same material, expanded magnet moment only as historical force reference.
 rec,rows=solve_case(70,False,3.04);records.append(rec)
 refined,refrows=solve_case(70,True,rtol=5e-10,max_step=.03,cycles=32)
 default=series[(70,True)];reference=next(q for q in records if q['radius_mm']==70 and q['model']=='Langevin')
 convergence=dict(default_force_rms_N=reference['force_rms_N'],refined_force_rms_N=refined['force_rms_N'],relative_force_difference=(refined['force_rms_N']/reference['force_rms_N']-1),refined=refined)
 stat=[];static_series={}
 for full in [False,True]:q,a=static_curve(full);stat.append(q);static_series[full]=a
 comparison=[]
 for r in [65,70,75,80]:
  lin=next(q for q in records if q['radius_mm']==r and q['model']=='linear' and q['moment_scale']==1);non=next(q for q in records if q['radius_mm']==r and q['model']=='Langevin');comparison.append(dict(radius_mm=r,force_ratio_full_to_linear=non['force_rms_N']/lin['force_rms_N'],normal_ratio=non['normal_attraction_mean_N']/lin['normal_attraction_mean_N'],torque_ratio=non['torque_rms_N_m']/lin['torque_rms_N_m']))
 result=dict(material_SI=dict(Ms_A_m=MS,a_A_m=AA,k_A_m=KP,c=C,alpha=AL,demagnetizing_factor=N),geometry=dict(ball_diameter_mm=12,magnet_diameter_mm=28,magnet_thickness_mm=15,ring_radius_mm=70,ball_center_to_magnet_center_plane_mm=39.62,magnet_moment_A_m2=4.222662956003016),protocol='Fixed ball position and fixed attitude, source disk rotating; field period = 1/4 turn. Irreversible closure follows repository isotropic projection model. 24 field periods from Q=0; last period reported. No mechanical trajectory or orbital-period fit.',records=records,comparison=comparison,static=stat,convergence=convergence)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2))
 fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for full,label in [(False,'Linear'),(True,'Langevin')]:
  a=series[(70,full)];ax[0,0].plot(a[:,0],a[:,7]*1000,label=label);ax[0,1].plot(a[:,0],-a[:,9]*1000,label=label);ax[1,0].plot(a[:,1]/1000,a[:,4]/1000,label=label);p=static_series[full];ax[1,1].plot(p[:,0],p[:,1]*1000,label=label)
 ax[0,0].set(xlabel='Source field phase (rad)',ylabel='Radial force (mN)',title='Reference moment, 70 mm fixed position');ax[0,1].set(xlabel='Source field phase (rad)',ylabel='Downward magnetic force (mN)');ax[1,0].set(xlabel='External Hx (kA/m)',ylabel='Ball Mx (kA/m)',title='Self-consistent magnetization, sphere N=1/3');ax[1,1].set(xlabel='Ball radius (mm)',ylabel='Radial force (mN)',title='Stopped source, reversible Q=0 reference')
 for a in ax.ravel():a.grid(alpha=.2);a.legend()
 fig.savefig(OUT/'linear_vs_langevin_force.png',dpi=160);plt.close(fig)
 print('COMPARISON',json.dumps(comparison),flush=True)
