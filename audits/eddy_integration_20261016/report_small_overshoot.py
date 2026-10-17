import json,numpy as np,pandas as pd
from pathlib import Path
p=Path('src2d/runs/video56c6-rest-release-20261017');rs=json.loads((p/'small_overshoot_zero_checks.json').read_text());report=[]
for r in rs:
 a=pd.read_csv(p/(r['key']+'.csv'));t=a.time_s.to_numpy();n=round(1/np.median(np.diff(t)));sm=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid');tail=t>60;ph=np.unwrap(a.phi_rad.to_numpy());period=2*np.pi/abs(np.polyfit(t[tail],ph[tail],1)[0]);row=dict(key=r['key'],c=r['config']['c'],K=r['config']['K'],peak_mean_radius_mm=float(sm.max()),tail_radius_mm=float(a.radius_mm[tail].mean()),overshoot_mm=float(sm.max()-a.radius_mm[tail].mean()),period_s=float(period),speed_mm_s=float(a.speed_mm_s[tail].mean()),wz_rad_s=float(a.wz_rad_s[tail].mean()),wz_std=float(a.wz_rad_s[tail].std()),startup_rmse_mm_s=r['rmse_curve_mm_s']);report.append(row)
 print(row)
(p/'small_overshoot_summary.json').write_text(json.dumps(report,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
for key,label,color in [('dd0d31720fc0215c','启动优先：过冲约8 mm','#ea580c'),(rs[0]['key'],'C=0.40，K=0.2×原值','#2563eb'),(rs[1]['key'],'C=0.35，K=0.3×原值','#059669')]:
 a=pd.read_csv(p/(key+'.csv'));t=a.time_s.to_numpy();n=round(1/np.median(np.diff(t)));sm=lambda v:np.convolve(v,np.ones(n)/n,mode='valid');tt=sm(t)+4.9
 axs[0].plot(tt,sm(a.radius_mm),label=label,color=color);axs[1].plot(tt,sm(a.speed_mm_s),label=label,color=color)
cv=pd.read_csv('cv/kinematics_latest6/56c6fd6092d21301a3c7808b624152c1/kinematics.csv');valid=cv.derivative_valid==1
for lo in np.arange(7,72):
 w=valid&(cv.source_pts_s>=lo)&(cv.source_pts_s<lo+1)
 if w.sum()>5:
  axs[0].plot(lo+.5,cv.r_mm[w].mean(),'k.',ms=4);axs[1].plot(lo+.5,cv.com_speed_mm_s[w].mean(),'k.',ms=4)
for ax in axs:ax.grid(alpha=.25);ax.set_xlim(5,72)
axs[0].set(ylabel='1秒平均轨道半径 (mm)',title='已有小过冲路线：零初态补跑后仍只有约1–2 mm过冲');axs[0].legend(fontsize=9)
axs[1].set(ylabel='1秒平均质心速率 (mm/s)',xlabel='视频时间 PTS (s)',title='代价是启动更早、更平缓；黑点为视频平均值')
fig.savefig(p/'small_overshoot_comparison.png',dpi=150)
