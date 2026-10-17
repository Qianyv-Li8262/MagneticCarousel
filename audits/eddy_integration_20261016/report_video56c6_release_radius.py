from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/video56c6-release-radius-20261017';OLD=ROOT/'runs/video56c6-rest-release-20261017'
records=json.loads((OUT/'runs.json').read_text());meta=json.loads((OUT/'release_measurement.json').read_text());cv=pd.read_csv(ROOT.parent/'cv/kinematics_latest6/56c6fd6092d21301a3c7808b624152c1/kinematics.csv')
cur=pd.read_csv(OUT/(records[0]['key']+'.csv'));oldrec=json.loads((OLD/'best_rest.json').read_text());old=pd.read_csv(OLD/(oldrec['key']+'.csv'));small=pd.read_csv(OUT/(records[2]['key']+'.csv'))
def sm(a,key,width=1):
 n=round(width/np.median(np.diff(a.time_s)));k=np.ones(n)/n
 return np.convolve(a.time_s+4.9,k,mode='valid'),np.convolve(a[key],k,mode='valid')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
fig,axs=plt.subplots(3,1,figsize=(11,9),layout='constrained')
w=(cv.position_valid==1)&(cv.source_pts_s>=4.9)&(cv.source_pts_s<=8)
axs[0].plot(cv.source_pts_s[w],cv.r_mm[w],'k.-',label='视频有效球心位置')
for a,col,ls,label in [(old,'#94a3b8','--','原候选，r0=67.4 mm'),(cur,'#2563eb','-','改用释放帧，r0=63.09 mm')]:axs[0].plot(a.time_s+4.9,a.radius_mm,color=col,ls=ls,label=label,lw=1)
axs[0].plot(4.9,meta['radius_mm'],'ro',ms=5,label='释放帧位置');axs[0].set(xlim=(4.87,8),ylim=(59,76),ylabel='瞬时半径 (mm)',title='释放位置对齐；随后径向振动与平衡位置仍有差别');axs[0].legend(fontsize=9)
for lo in np.arange(5,72,1):
 w=(cv.source_pts_s>=lo)&(cv.source_pts_s<lo+1)&(cv.position_valid==1)
 if w.sum()>5:axs[1].plot(lo+.5,cv.r_mm[w].mean(),'k.',ms=5)
 w=(cv.source_pts_s>=lo)&(cv.source_pts_s<lo+1)&(cv.derivative_valid==1)
 if w.sum()>5:axs[2].plot(lo+.5,cv.com_speed_mm_s[w].mean(),'k.',ms=5)
for a,col,ls,label in [(old,'#94a3b8','--','原启动优先候选'),(cur,'#2563eb','-','启动优先材料，释放半径63.09 mm'),(small,'#059669','-','小过冲材料，释放半径63.09 mm')]:
 for ax,key in [(axs[1],'radius_mm'),(axs[2],'speed_mm_s')]:
  tt,yy=sm(a,key);ax.plot(tt,yy,color=col,ls=ls,label=label)
axs[1].set(xlim=(4.9,72),ylabel='1秒平均半径 (mm)',title='模型仍回到约68 mm的早段平衡半径，视频约66 mm');axs[1].legend(fontsize=8.5)
axs[2].set(xlim=(4.9,40),ylabel='1秒平均速率 (mm/s)',title='只改初始半径，没有改善延迟启动');axs[2].legend(fontsize=8.5)
for ax in axs:ax.grid(alpha=.25);ax.set_xlabel('视频时间 PTS (s)，释放时刻4.900 s')
fig.suptitle('使用视频释放帧的实测初始半径：63.09 mm',fontsize=16)
fig.savefig(OUT/'comparison.png',dpi=150)
summary=dict(measurement=meta,results=[])
for rec in records:
 a=pd.read_csv(OUT/(rec['key']+'.csv'));tt=a.time_s+4.9;w=(tt>=10)&(tt<17);tail=a.time_s>60;n=round(1/np.median(np.diff(a.time_s)));rs=np.convolve(a.radius_mm,np.ones(n)/n,mode='valid')
 summary['results'].append(dict(tag=rec['tag'],config=rec['config'],startup_rmse_mm_s=rec['rmse_curve_mm_s'],early_mean_radius_mm=float(a.radius_mm[w].mean()),early_mean_speed_mm_s=float(a.speed_mm_s[w].mean()),peak_mean_radius_mm=float(rs.max()),tail_radius_mm=float(a.radius_mm[tail].mean()),overshoot_mm=float(rs.max()-a.radius_mm[tail].mean()),tail_period_s=rec['holdout']['period']))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['results'][0],indent=2))
