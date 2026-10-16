"""Compare stationary, unmagnetized release with the earlier free-spin B fit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_video56c6 as f
OLD=f.OUT;OUT=f.ROOT/'runs/video56c6-rest-release-20261017'
rec=json.loads((OUT/'best_rest.json').read_text());orig=json.loads((OLD/'startup_candidate.json').read_text())
a=np.genfromtxt(OUT/(rec['key']+'.csv'),names=True,delimiter=',');b=np.genfromtxt(OLD/(orig['key']+'.csv'),names=True,delimiter=',');direct=np.genfromtxt(OUT/'6b0f0fd50be1ab88.csv',names=True,delimiter=',')
valid=f.DATA['derivative_valid']==1;pts=f.DATA['source_pts_s'];target=np.array([x['speed'] for x in f.BINS]);quiet=np.array([9<=x['pts']<18 for x in f.BINS])
def lap_table(x):
 w=x['time_s']>15;ph=-np.unwrap(x['phi_rad'])[w];tt=x['time_s'][w]+f.RELEASE
 levels=2*np.pi*np.arange(int(np.floor(ph[0]/(2*np.pi)))+1,int(np.floor(ph[-1]/(2*np.pi)))+1)
 cross=np.interp(levels,np.maximum.accumulate(ph),tt)
 return np.column_stack(((cross[1:]+cross[:-1])/2,np.diff(cross),cross[:-1],cross[1:]))
pp=lap_table(a);pm=(pp[:,2]>=35)&(pp[:,3]<=72)
half=next(r for r in json.loads((OUT/'validation.json').read_text()) if r['tag']=='rest_half_dt');long=next(r for r in json.loads((OUT/'validation.json').read_text()) if r['tag']=='rest_long')
h=np.genfromtxt(OUT/(half['key']+'.csv'),names=True,delimiter=',');z=np.genfromtxt(OUT/(long['key']+'.csv'),names=True,delimiter=',');tail=z['time_s']>=220
v=np.array(rec['curve_means'])
report=dict(parameters=rec['config'],initial_conditions='Exactly zero COM speed, angular velocity, JA memory and eddy memory; no held stage.',startup_rmse_mm_s=rec['rmse_curve_mm_s'],old_B_rmse_mm_s=orig['startup_1s_curve_rmse_mm_s'],direct_zero_initial_rmse_mm_s=next(r['rmse_curve_mm_s'] for r in json.loads((OUT/'initial_checks.json').read_text()) if r['tag']=='rest_r68.5'),period_mean_s=float(pp[pm,1].mean()),target_period_s=f.TARGET_PERIOD,period_error_percent=float((pp[pm,1].mean()/f.TARGET_PERIOD-1)*100),quiet_window_pts_s=[9,18],quiet_mean_model_speed_mm_s=float(v[quiet].mean()),quiet_mean_target_speed_mm_s=float(target[quiet].mean()),late_metrics=rec['holdout'],verification=dict(half_dt_max_speed_difference_mm_s=float(abs(a['speed_mm_s']-h['speed_mm_s']).max()),half_dt_max_radius_difference_mm=float(abs(a['radius_mm']-h['radius_mm']).max()),completed_seconds=float(z['time_s'][-1]),last20s_mean_speed_mm_s=float(z['speed_mm_s'][tail].mean()),last20s_mean_radius_mm=float(z['radius_mm'][tail].mean()),status=long['status']))
(OUT/'result.json').write_text(json.dumps(report,indent=2));(OUT/'candidate.json').write_text(json.dumps(rec['config'],indent=2))
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11})
fig,axs=plt.subplots(3,1,figsize=(11.5,9),layout='constrained')
axs[0].plot(pts[valid],f.DATA['com_speed_mm_s'][valid],color='#94a3b8',alpha=.4,lw=.6,label='视频有效速度')
axs[0].plot([x['pts'] for x in f.BINS],target,'k.-',lw=1.4,label='视频 1秒平均')
def smooth(x,key):
 n=round(1/np.median(np.diff(x['time_s'])));k=np.ones(n)/n
 return np.convolve(x['time_s']+f.RELEASE,k,mode='valid'),np.convolve(x[key],k,mode='valid')
for x,color,ls,label in [(b,'#ea580c','--','原 B：预磁化2圈，自旋−20 rad/s'),(direct,'#94a3b8',':','B材料直接改为零初态'),(a,'#2563eb','-','新候选：零初态，材料微调')]:
 xx,yy=smooth(x,'speed_mm_s');axs[0].plot(xx,yy,color=color,ls=ls,lw=1.7,label=label)
axs[0].axvspan(9,18,color='#fde68a',alpha=.2);axs[0].set(xlim=(5,40),ylim=(-5,380),ylabel='质心速率 (mm/s)',title='移除预磁化和初始自旋后，可恢复原 B 的曲线误差水平');axs[0].legend(fontsize=8.5)
pl=f.LAPS;w=(pl['start_pts_s']>=20)&(pl['end_pts_s']<=72)
axs[1].plot((pl['start_pts_s'][w]+pl['end_pts_s'][w])/2,pl['period_s'][w],'ko',ms=4,label='视频逐圈周期')
for x,color,ls,label in [(b,'#ea580c','--','原 B'),(a,'#2563eb','-','零初态新候选')]:
 p=lap_table(x);axs[1].plot(p[:,0],p[:,1],color=color,ls=ls,label=label)
axs[1].axhline(f.TARGET_PERIOD,color='gray',ls=':',label=f'视频平均 {f.TARGET_PERIOD:.3f} s');axs[1].set(xlim=(20,72),ylim=(1.35,2.4),ylabel='公转周期 (s)',title=f'启动优先：新候选平均周期 {report["period_mean_s"]:.3f} s');axs[1].legend(fontsize=9)
for lo in np.arange(6,72):
 w=(pts>=lo)&(pts<lo+1)&(f.DATA['position_valid']==1)
 if w.sum()>5:axs[2].plot(lo+.5,np.nanmean(f.DATA['r_mm'][w]),'k.',ms=5)
for x,color,ls,label in [(b,'#ea580c','--','原 B'),(a,'#2563eb','-','零初态新候选')]:
 xx,yy=smooth(x,'radius_mm');axs[2].plot(xx,yy,color=color,ls=ls,label=label)
axs[2].set(xlim=(5,72),ylim=(63,85),ylabel='平均轨道半径 (mm)',title='径向超调和早段平衡半径偏差仍在');axs[2].legend(fontsize=9)
for ax in axs:ax.grid(alpha=.25);ax.set_xlabel('视频时间 PTS (s)；模拟在视频 4.9 s 释放，无额外时间平移')
fig.suptitle('无预磁化、零自旋、质心静止释放：低高度分支再拟合',fontsize=17)
fig.savefig(OUT/'comparison.png',dpi=155)
print(json.dumps(report,indent=2))
