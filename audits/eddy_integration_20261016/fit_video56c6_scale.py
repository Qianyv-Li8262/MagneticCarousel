"""Fit a single CV spatial calibration factor; time and apparatus geometry stay fixed."""
from pathlib import Path
import sys,json,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
import fit_video56c6 as f
OUT=f.ROOT/'runs/video56c6-scale-fit-20261017';OUT.mkdir(exist_ok=True)
f.OUT=OUT
RAW=pd.read_csv(f.VIDEO/'kinematics.csv');RELEASE_RADIUS=63.092177723551295
TARGETS=[]
for lo in np.arange(7.,72.,1):
 w=(RAW.source_pts_s>=lo)&(RAW.source_pts_s<lo+1)
 v=w&(RAW.derivative_valid==1);p=w&(RAW.position_valid==1)
 if v.sum()>=5 and p.sum()>=5:TARGETS.append(dict(pts=lo+.5,time=lo+.5-f.RELEASE,v=float(RAW.com_speed_mm_s[v].mean()),r=float(RAW.r_mm[p].mean())))
TARGETS=np.array([[x[k] for k in ['pts','time','v','r']] for x in TARGETS])
(OUT/'targets.json').write_text(json.dumps(dict(scale_definition='true position=s*CV position; true velocity=s*CV velocity; r0=s*63.0921777 mm; no time scaling or time shift',release_pts_s=4.9,release_radius_CV_mm=RELEASE_RADIUS,target_period_s=f.TARGET_PERIOD,targets=TARGETS.tolist(),fixed=dict(h_mm=32,moment_Am2=4.222661,initial_speed=0,initial_spin=0,initial_JA_memory=0,initial_eddy_memory=0,held_turns=0,tilt=0,rolling_mu=.0049,disk_rpm=200)),indent=2))

def evaluate(x,tag='',seconds=72.,dt=.0005,save_best=False):
 # sigma in 1e7 S/m, mu in baseline units, C, K in baseline units, s, release phase.
 sigma,mu,c,k,s,phase=x[:6]
 cfg=dict(f.BASE,sigma=sigma*1e7,mu=mu*f.BASE['mu'],c=c,K=k*f.BASE['K'],r=s*RELEASE_RADIUS,phase=phase,seconds=seconds,dt=dt,wz=0.,hold=0.,vt=0.,vr=0.,qz=0.)
 if len(x)>6:cfg['alpha']=f.BASE['alpha']*x[6]
 native=f.run(cfg,tag=tag)
 key=hashlib.sha256(('scaleObjectiveV2'+json.dumps(dict(x=list(map(float,x)),seconds=seconds,dt=dt),sort_keys=True)).encode()).hexdigest()[:16]
 dest=OUT/('scale_'+key+'.json')
 if dest.exists():return json.loads(dest.read_text())
 a=pd.read_csv(OUT/(native['key']+'.csv'));t=a.time_s.to_numpy();phi=np.unwrap(a.phi_rad.to_numpy());res=[];vm=[];rm=[]
 for pts,tt,v,r in TARGETS:
  w=(t>=tt-.5)&(t<tt+.5)
  vm.append(float(a.speed_mm_s[w].mean())/s if w.any() else 1e4)
  rm.append(float(a.radius_mm[w].mean())/s if w.any() else 1e4)
 vm=np.array(vm);rm=np.array(rm);early=TARGETS[:,0]<35;quiet=(TARGETS[:,0]>=9)&(TARGETS[:,0]<18)
 # Startup gets priority; radial shape and overshoot are now explicit constraints.
 res.extend(((vm-TARGETS[:,2])/25*np.where(quiet,1.5,np.where(early,1.,.45))).tolist())
 res.extend(((rm-TARGETS[:,3])/1.2*.7).tolist())
 w=(t>=35-f.RELEASE)&(t<=72-f.RELEASE)
 period=2*np.pi/abs(np.polyfit(t[w],phi[w],1)[0]) if w.sum()>10 else 1e3
 n=round(1/np.median(np.diff(t)));smooth_r=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid')/s
 late=(t>=55-f.RELEASE)&(t<=72-f.RELEASE);tailr=float(a.radius_mm[late].mean())/s if late.any() else 1e3
 overshoot=float(smooth_r.max()-tailr)
 res.extend([(period-f.TARGET_PERIOD)/.035,max(0,overshoot-1.5)/.5])
 if t[-1]<seconds-.01 or native['status'].split('status=')[-1]!='0':res=[1e3]*len(res)
 res=np.nan_to_num(res,nan=1e3,posinf=1e3,neginf=-1e3)
 rec=dict(key=key,native_key=native['key'],x=list(map(float,x)),config=cfg,scale=s,tag=tag,residual=res.tolist(),score=float(np.mean(res**2)),period_s=float(period),period_error_percent=float((period/f.TARGET_PERIOD-1)*100),startup_speed_rmse_CV_mm_s=float(np.sqrt(np.mean((vm[early]-TARGETS[early,2])**2))),radius_rmse_CV_mm=float(np.sqrt(np.mean((rm-TARGETS[:,3])**2))),quiet_speed_CV_mm_s=float(vm[quiet].mean()),quiet_target_CV_mm_s=float(TARGETS[quiet,2].mean()),overshoot_CV_mm=overshoot,model_speed_CV=vm.tolist(),model_radius_CV=rm.tolist(),status=native['status'])
 dest.write_text(json.dumps(rec,indent=2));print('SCALE',tag,'score',round(rec['score'],3),'s',round(s,4),'period',round(period,4),'quiet',round(rec['quiet_speed_CV_mm_s'],1),'over',round(overshoot,2),flush=True)
 return rec

def batch(cases):
 with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(lambda a:evaluate(a[1],a[0]),cases))

if __name__=='__main__':
 base_mu=f.BASE['mu'];cases=[
 ('startup',[.84280685,554.30314/base_mu,.23816504,.00581625713/f.BASE['K'],1.03,.58]),
 ('smallover',[1.2,1,.4,.2,1.03,.58]),
 ('threshold7',[.7,1,f.BASE['c'],1,1.03,.58]),
 ('threshold6',[.6,1,f.BASE['c'],1,1.03,.58]),
 ('highmu_wait',[1.2,1200/base_mu,f.BASE['c'],.8,1.03,.58]),
 ('highmu_gain',[1.2,1200/base_mu,f.BASE['c'],.6,1.03,.58]),
 ('threshold75',[.75,1,f.BASE['c'],1,1.03,.58]),
 ('medium_c',[1.2,1,.25,1,1.03,.58])]
 r=batch(cases);(OUT/'seeds.json').write_text(json.dumps(r,indent=2))
