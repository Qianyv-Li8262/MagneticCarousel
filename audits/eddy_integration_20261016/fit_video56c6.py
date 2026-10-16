"""Mechanism-directed fitting of low-height branch to 56c6: isolated material snapshot."""
from pathlib import Path
import numpy as np,json,subprocess,hashlib,time,re,threading
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent
SOURCE=ROOT/'scratch/video56c6-fit-20261016-source';OUT=ROOT/'runs/video56c6-fit-20261016';OUT.mkdir(parents=True,exist_ok=True)
VIDEO=PROJECT/'cv/kinematics_latest6/56c6fd6092d21301a3c7808b624152c1'
DATA=np.genfromtxt(VIDEO/'kinematics.csv',names=True,delimiter=',');LAPS=np.genfromtxt(VIDEO/'orbit_periods.csv',names=True,delimiter=',')
RELEASE=4.9;BASE=dict(sigma=9e6,alpha=.0021323083849413885,c=.22415154978863677,K=.005375927782155882,r=70.,vt=0.,vr=0.,wz=0.,hold=0.,phase=.58,seconds=72.,dt=.0005,mu=605.8974337495617,qz=0.)
KEYS=list(BASE)
EDDY_LOCK=threading.Lock()
def eddy_table(mu):
 with EDDY_LOCK:
  key=format(mu,'.12g');binary=SOURCE/f'eddy_{key}.bin'
  if not binary.exists():
   header=SOURCE/f'eddy_{key}.hpp';report=SOURCE/f'eddy_{key}.json'
   subprocess.run([str(SOURCE/'generate_eddy_fit.exe'),str(header),str(report),str(mu)],check=True,capture_output=True)
   text=header.read_text();arrays=[]
   for name in ['aj','inv_tau_ref_star']:
    body=re.search(name+r'\[mode_count\] = \{(.*?)\};',text,re.S).group(1)
    arrays.extend(float(x) for x in body.split(',') if x.strip())
   np.asarray(arrays,dtype='<f8').tofile(binary)
  return binary
BINS=[]
for lo in np.arange(7.,35.,1.):
 w=(DATA['source_pts_s']>=lo)&(DATA['source_pts_s']<lo+1)&(DATA['derivative_valid']==1)
 if w.sum()>=5:BINS.append(dict(pts=lo+.5,time=lo+.5-RELEASE,speed=float(np.mean(DATA['com_speed_mm_s'][w])),r=float(np.mean(DATA['r_mm'][w])),vt=float(np.mean(DATA['signed_tangential_velocity_mm_s'][w]))))
w=(LAPS['start_pts_s']>=35)&(LAPS['end_pts_s']<=72)
TARGET_PERIOD=float(LAPS['period_s'][w].mean())
def run(config=None,tag='',**kwargs):
 cfg=BASE.copy();cfg.update(config or {});cfg.update(kwargs)
 key=hashlib.sha256(('v2'+json.dumps(cfg,sort_keys=True)).encode()).hexdigest()[:16];path=OUT/(key+'.csv');meta=OUT/(key+'.json')
 if not meta.exists():
  cmd=[str(SOURCE/'fit_video56c6.exe'),str(SOURCE/'field_lut_fp64.bin'),str(path)]+[str(cfg[k]) for k in KEYS]+[str(eddy_table(cfg['mu']))]
  start=time.perf_counter();result=subprocess.run(cmd,capture_output=True,text=True,check=True)
  a=np.genfromtxt(path,names=True,delimiter=',');t=a['time_s'];phi=np.unwrap(a['phi_rad']);res=[]
  means=[]
  for b in BINS:
   w=(t>=b['time']-.5)&(t<b['time']+.5)
   v=float(a['speed_mm_s'][w].mean()) if w.any() else 1e4
   means.append(v);res.append((v-b['speed'])/25.)
  def window(lo,hi):
   w=(t>=lo-RELEASE)&(t<=hi-RELEASE)
   rate=np.polyfit(t[w],phi[w],1)[0] if w.sum()>5 else np.nan
   return dict(period=2*np.pi/abs(rate),speed=float(np.mean(a['speed_mm_s'][w])),radius=float(np.mean(a['radius_mm'][w])),wz=float(np.mean(a['wz_rad_s'][w])),vt=float(np.mean(a['vt_mm_s'][w])))
  train=window(35,55);test=window(55,72)
  res += [(train['period']-TARGET_PERIOD)/.025,(test['period']-TARGET_PERIOD)/.025,(train['radius']-75.7)/2]
  if t[-1]<65:res=[1e3]*len(res)
  rec=dict(key=key,config=cfg,tag=tag,status=result.stdout.strip(),wall_seconds=time.perf_counter()-start,train=train,holdout=test,curve_means=means,residual=res,score=float(np.mean(np.square(res))),rmse_curve_mm_s=float(np.sqrt(np.mean((np.array(means)-[b['speed'] for b in BINS])**2))))
  meta.write_text(json.dumps(rec,indent=2))
 else:rec=json.loads(meta.read_text())
 print(tag,key,'score',round(rec['score'],3),'period',round(rec['train']['period'],4),'curveRMSE',round(rec['rmse_curve_mm_s'],2),'wz',round(rec['train']['wz'],1),flush=True)
 return rec
def batch(cases):
 with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(lambda c:run(c[1],c[0]),cases))
if __name__=='__main__':
 (OUT/'targets.json').write_text(json.dumps(dict(video=VIDEO.name,release_pts_s=RELEASE,train_period=TARGET_PERIOD,bins=BINS,fixed=dict(h_mm=32,magnet_moment_Am2=4.222661,Mu_r=605.8974337495617,rolling_mu=.0049,disk_rpm=200,tilt=0),fit_parameters=['sigma','alpha','c','K','r','vt','vr','wz','hold','phase'],note='Ignore third harmonic; disk RPM is nominal, not measured in existing video extraction.'),indent=2))
 cases=[('baseline9',{}),('baseline12',dict(sigma=12e6)),('release_r67',dict(sigma=12e6,r=67,vt=-15,vr=70)),('held_r67',dict(sigma=12e6,r=67,vt=-15,vr=70,hold=2))]
 results=batch(cases);(OUT/'stage1.json').write_text(json.dumps(results,indent=2))
