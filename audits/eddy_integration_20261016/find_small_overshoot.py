from pathlib import Path
import numpy as np,pandas as pd,json,re
ROOT=Path('src2d');outs=[ROOT/'runs/video56c6-fit-20261016',ROOT/'runs/video56c6-rest-release-20261017'];rows=[]
for out in outs:
 for p in out.glob('*.json'):
  if not re.fullmatch('[0-9a-f]{16}',p.stem):continue
  r=json.loads(p.read_text())
  if r.get('status','').split('status=')[-1]!='0':continue
  cfg=r['config']
  if cfg['seconds']!=72 or cfg['dt']!=.0005:continue
  a=pd.read_csv(out/(p.stem+'.csv'),usecols=['time_s','speed_mm_s','radius_mm','wz_rad_s','phi_rad'])
  t=a.time_s.to_numpy();n=round(1/np.median(np.diff(t)));radius=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid');ts=np.convolve(t,np.ones(n)/n,mode='valid');w=t>60
  tailr=float(a.radius_mm[w].mean());tailv=float(a.speed_mm_s[w].mean());tailw=float(a.wz_rad_s[w].mean());stdw=float(a.wz_rad_s[w].std());stdr=float(a.radius_mm[w].std());ph=np.unwrap(a.phi_rad.to_numpy());period=2*np.pi/abs(np.polyfit(t[w],ph[w],1)[0])
  rows.append(dict(key=p.stem,directory=str(out),tag=r['tag'],over_mm=float(radius.max()-tailr),peak_r_mm=float(radius.max()),tail_r_mm=tailr,tail_v_mm_s=tailv,tail_wz_rad_s=tailw,tail_wz_std=stdw,tail_r_std=stdr,period_s=float(period),peak_pts=float(ts[np.argmax(radius)]+4.9),startup_rmse=r['rmse_curve_mm_s'],config=cfg))
valid=[x for x in rows if x['tail_v_mm_s']>250 and x['tail_wz_rad_s']<-60 and x['tail_wz_std']<5 and x['tail_r_std']<1]
valid.sort(key=lambda x:x['over_mm'])
for x in valid[:15]:print(json.dumps(x,ensure_ascii=False))
print('counts',len(rows),len(valid),'small',sum(x['over_mm']<2 for x in valid))
(outs[1]/'overshoot_scan.json').write_text(json.dumps(dict(all_runs=rows,stable_highspeed_negative_spin=valid),indent=2))
