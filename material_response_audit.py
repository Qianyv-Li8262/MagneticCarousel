"""Reference-moment material scan, SI force audit, and stationary radial well."""
from pathlib import Path
import importlib.util,json,sys,time,hashlib
import numpy as np
from scipy.optimize import brentq
import animate_one_trajectory as animation
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/material-response-20261013';OUT.mkdir(exist_ok=True)
BINARY=next((ROOT/'scratch/material-response-20261013-source').glob('material_audit*.pyd'))
spec=importlib.util.spec_from_file_location('material_audit',BINARY);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
BASE=np.array([605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882]);OMEGA=199.8313400018371*2*np.pi/60;KSPACE=4/.07;RHO=7850;A=.006;MS=1.1e6;V=4*np.pi*A**3/3;MASS=RHO*V;MU0=4*np.pi*1e-7
ALPHA=KSPACE*A;GAMMA=3*MU0*MS*MS*KSPACE*KSPACE/(RHO*OMEGA*OMEGA)
params=mod.SolverParams(ALPHA,3*1.225*.47/(8*RHO*A*KSPACE),KSPACE*9.81/OMEGA**2,GAMMA,GAMMA/(.4*ALPHA**2));params.reverse_push_acceleration=0
TILT=.5*np.arctan(4/930)
def configure(material,moment=1.):
 mod.configure_material(*material);mod.configure_magnet_moment_scale(moment);mod.configure_laser_tilt(TILT,np.pi/2);mod.configure_rolling_resistance(1e-5,1e-4,.004)
def diag(state):return mod.material_diagnostics(state,params)
def static_well(moment):
 configure(BASE,moment);s=np.zeros(15);s[5]=1;s[0]=0
 radii=np.linspace(.0141,.15,1001);out=[]
 for r in radii:
  s[1]=r*KSPACE;q=diag(s);H=q[:3]*MS;J=q[18:]*MS*KSPACE;F=q[6:9]*MASS*OMEGA**2/KSPACE
  U=-1.5*MU0*V*q[15]*np.dot(H,H)
  out.append([r,*H,*F,U,*J])
 data=np.array(out);rootguess=radii[np.argmin(data[:,7])]
 def force(r):s[1]=r*KSPACE;return diag(s)[6]*MASS*OMEGA**2/KSPACE
 idx=np.argmin(abs(radii-rootguess));left=radii[max(0,idx-10)];right=radii[min(len(radii)-1,idx+10)];r0=brentq(force,left,right)
 dr=1e-5;stiff=-(force(r0+dr)-force(r0-dr))/(2*dr);s[1]=r0*KSPACE;q=diag(s);H=q[:3]*MS;depth=1.5*MU0*V*q[15]*np.dot(H,H)
 rec=dict(moment_scale=moment,model='Stationary disk, q=0 reversible sphere response; upper reversible benchmark, not a full moving JA/contact simulation.',equilibrium_radius_mm=r0*1000,radial_stiffness_N_m=stiff,frequency_free_translation_hz=float(np.sqrt(stiff/MASS)/(2*np.pi)),frequency_pure_rolling_hz=float(np.sqrt(stiff/(1.4*MASS))/(2*np.pi)),well_depth_to_zero_field_J=depth,equivalent_rise_mm=depth/(MASS*9.81)*1000,escape_speed_pure_rolling_m_s=float(np.sqrt(2*depth/(1.4*MASS))),field_at_equilibrium_mT=float(np.linalg.norm(H)*MU0*1000),peak_radial_force_N=float(np.max(abs(data[:,4]))),peak_gradient_T_m=float(np.max(np.linalg.norm(data[:,8:]*MU0,axis=1))))
 np.savetxt(OUT/f'static_radial_m{moment:g}.csv',data,delimiter=',',header='radius_m,Hx_A_m,Hy_A_m,Hz_A_m,Fx_N,Fy_N,Fz_N,U_J,dHxx_A_m2,dHxy_A_m2,dHxz_A_m2,dHyy_A_m2,dHyz_A_m2,dHzz_A_m2',comments='')
 return rec
static=[]

def loop(material):
 configure(material);s=animation.make_initial_state(mod,alpha=ALPHA);s[0]=0;dt=.0005;out=[]
 for j in range(6*160):
  steps=round((j+1)*2*np.pi/(160*dt))-round(j*2*np.pi/(160*dt));s=mod.magnetize_held_ball(s,dt,steps,params)
  if j>=4*160:out.append(diag(s))
 d=np.array(out);force=d[:,6:9]*MASS*OMEGA**2/KSPACE;H=d[:,:3]*MS;M=d[:,3:6]*MS;J=d[:,18:]*MS*KSPACE
 # Independent SI reconstruction F = mu0 V J_H^T M.
 tensors=np.array([[[j[0],j[1],j[2]],[j[1],j[3],j[4]],[j[2],j[4],j[5]]] for j in J]);direct=MU0*V*np.einsum('nij,nj->ni',tensors,M)
 rec=dict(force_rms_N=float(np.sqrt(np.mean(np.sum(force**2,axis=1)))),normal_attraction_mean_N=float(-force[:,2].mean()),tangent_mean_N=float(force[:,1].mean()),magnetization_rms_A_m=float(np.sqrt(np.mean(np.sum(M*M,axis=1)))),field_rms_mT=float(MU0*np.sqrt(np.mean(np.sum(H*H,axis=1)))*1000),torque_rms_N_m=float(np.sqrt(np.mean(np.sum(d[:,9:12]**2,axis=1)))*(.4*MASS*A*A*OMEGA**2)),force_SI_reconstruction_relative_error=float(np.linalg.norm(force-direct)/np.linalg.norm(force)),D=float(d[0,15]),E=float(d[0,16]))
 return rec

def run(material,seconds=180,dt=.0005,save=False,label=''):
 key=hashlib.sha256(json.dumps([material.tolist(),seconds,dt,1.,TILT,.004]).encode()).hexdigest()[:16];path=OUT/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 configure(material);began=time.perf_counter();state=animation.prepare_initial_state(mod,params,dt,v0=-.075);held_diag=diag(state);mode=direction=None;step=0;maxsteps=round(seconds*OMEGA/dt);rows=[np.r_[0,state,0]];chunk=round(.02*OMEGA/dt);error=None
 try:
  while step<maxsteps:
   n=min(1 if mode is None else chunk,maxsteps-step);state,mode,direction=mod.evolve_trajectory(state,dt,n,params,mode,direction,step);step+=n;rows.append(np.r_[step*dt/OMEGA,state,float(mode)])
 except RuntimeError as e:error=str(e)
 rows=np.array(rows);rec=dict(label=label,material=dict(zip(['mu_r','alpha_JA','C_JA','K'],material.tolist())),moment_scale=1.,seconds=seconds,dt=dt,wall_s=time.perf_counter()-began,error=error,completed_s=float(rows[-1,0]),held=held_diag.tolist())
 if len(rows)>20:
  lo=max(0,rows[-1,0]-min(60,seconds/2));used=rows[rows[:,0]>=lo];t=used[:,0];s=used[:,1:16];phi=np.unwrap(np.arctan2(s[:,2],s[:,1]));rad=np.linalg.norm(s[:,1:3],axis=1);rate=np.polyfit(t-t.mean(),phi,1)[0];tang=-(s[:,1]*s[:,4]-s[:,2]*s[:,3])/rad;X=np.c_[np.ones(len(t)),np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi)];co=np.linalg.lstsq(X,tang,rcond=None)[0]
  rec.update(period_s=float(-2*np.pi/rate) if rate<0 else None,radius_mean_mm=float(rad.mean()/KSPACE*1000),radius_range_mm=[float(rad.min()/KSPACE*1000),float(rad.max()/KSPACE*1000)],reverse_fraction=float(np.mean(tang>0)),first_harmonic_percent=float(np.hypot(co[1],co[2])/abs(co[0])*100) if abs(co[0])>1e-12 else None,rolling_fraction=float(np.mean(used[:,16]==0)))
  width=min(30,rows[-1,0]/3);wins=[]
  for j in [3,2,1]:
   u=rows[(rows[:,0]>=rows[-1,0]-j*width)&(rows[:,0]<=rows[-1,0]-(j-1)*width)]
   if len(u)<5:continue
   ang=np.unwrap(np.arctan2(u[:,3],u[:,2]));rate=np.polyfit(u[:,0]-u[:,0].mean(),ang,1)[0]
   wins.append(dict(period_s=float(-2*np.pi/rate) if rate<0 else None,radius_mm=float(np.linalg.norm(u[:,2:4],axis=1).mean()/KSPACE*1000)))
  rec['windows']=wins
  periods=[w['period_s'] for w in wins]
  rec['stationary']=bool(error is None and len(wins)==3 and all(q is not None for q in periods) and np.ptp(periods)/np.mean(periods)<.01 and np.ptp([w['radius_mm'] for w in wins])<.5 and rec['reverse_fraction']>.99)
 if save:np.savez_compressed(path.with_suffix('.npz'),rows=rows)
 path.write_text(json.dumps(rec,indent=2));print(json.dumps({k:rec.get(k) for k in ['label','period_s','radius_mean_mm','error','wall_s']}),flush=True);return rec

cases=[('baseline',BASE.copy())]
for mur in [2,10,30,100,300,1000,3000,1e4,1e5,1e6]:q=BASE.copy();q[0]=mur;cases.append((f'mur={mur:g}',q))
for factor in [.01,.03,.1,.3,3,10,30,100]:q=BASE.copy();q[1]*=factor;cases.append((f'alpha*x{factor:g}',q))
for c in [0,.05,.1,.5,.8,.95,1]:q=BASE.copy();q[2]=c;cases.append((f'C={c:g}',q))
for factor in [.01,.03,.1,.3,3,10,30,100]:q=BASE.copy();q[3]*=factor;cases.append((f'K*x{factor:g}',q))
if __name__=='__main__':
 static=[static_well(1),static_well(3.04)];(OUT/'static_well.json').write_text(json.dumps(static,indent=2));print('STATIC',json.dumps(static),flush=True)
 loops=[];runs=[]
 for label,material in cases:
  try:
   lp=dict(label=label,material=material.tolist(),**loop(material));loops.append(lp);print('LOOP',json.dumps(lp),flush=True)
   runs.append(run(material,label=label))
  except Exception as e:loops.append(dict(label=label,error=str(e)));print('CASE_ERROR',label,str(e),flush=True)
 (OUT/'scan_summary.json').write_text(json.dumps(dict(baseline=BASE.tolist(),binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),static=static,loops=loops,runs=runs),indent=2))
