"""Report scale-consistent fits and scale-invariant angular/period residuals."""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_video56c6_scale as g
OUT=g.OUT;wait=json.loads((OUT/'selected_wait_candidate.json').read_text());small=json.loads((OUT/'selected_smallover_candidate.json').read_text());validation=json.loads((OUT/'validation.json').read_text())
def load(rec):return pd.read_csv(OUT/(rec['native_key']+'.csv'))
def smooth(a,key,divide=1):
 n=round(1/np.median(np.diff(a.time_s)));k=np.ones(n)/n
 return np.convolve(a.time_s+g.f.RELEASE,k,mode='valid'),np.convolve(a[key],k,mode='valid')/divide

def laps(a):
 w=a.time_s>15;ph=-np.unwrap(a.phi_rad.to_numpy())[w];tt=a.time_s.to_numpy()[w]+4.9
 levels=2*np.pi*np.arange(int(np.floor(ph[0]/(2*np.pi)))+1,int(np.floor(ph[-1]/(2*np.pi)))+1);cross=np.interp(levels,np.maximum.accumulate(ph),tt)
 return np.column_stack(((cross[1:]+cross[:-1])/2,np.diff(cross),cross[:-1],cross[1:]))
def threshold(v,level):
 x=g.TARGETS[:,0];ix=np.where((x[:-1]>8)&(v[:-1]<level)&(v[1:]>=level))[0]
 if len(ix)==0:return None
 j=ix[0];return float(np.interp(level,v[j:j+2],x[j:j+2]))
report=dict(scale_definition='r_true=s*r_CV, v_true=s*v_CV; r0=s*63.0921777mm. Plot model/s against original CV. Time, angular speed, periods and relative radial shape do not scale.',target_period_s=g.f.TARGET_PERIOD,candidates={},conclusion='The waiting phase exists and can approximately align, but no satisfactory joint wait/acceleration/radial/period fit was found. The constant scale is conditional on a mismatched model, not an independent camera calibration.')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');fig.suptitle('统一空间尺度参与拟合：等待能接近，后续加速与周期仍有偏差',fontsize=17)
T=g.TARGETS;axs[0,0].plot(T[:,0],T[:,3],'k.',label='视频 1秒平均');axs[0,1].plot(T[:,0],T[:,2],'k.-',lw=1,label='视频 1秒平均')
raw=g.RAW;valid=raw.derivative_valid==1
for lo in np.arange(7,72):
 w=valid&(raw.source_pts_s>=lo)&(raw.source_pts_s<lo+1)
 if w.sum()>5:axs[1,0].plot(lo+.5,abs(raw.signed_tangential_velocity_mm_s[w]/raw.r_mm[w]).mean(),'k.',ms=5)
pl=g.f.LAPS;w=(pl['start_pts_s']>=20)&(pl['end_pts_s']<=72)
axs[1,1].plot((pl['start_pts_s'][w]+pl['end_pts_s'][w])/2,pl['period_s'][w],'ko',ms=4,label='视频逐圈周期');axs[1,1].axhline(g.f.TARGET_PERIOD,color='gray',ls=':',label='视频平均1.481 s')
for rec,color,label in [(wait,'#2563eb','等待优先'),(small,'#059669','过冲优先')]:
 a=load(rec);s=rec['scale'];desc=label+f'，s={s:.4f}'
 for ax,key in [(axs[0,0],'radius_mm'),(axs[0,1],'speed_mm_s')]:
  xx,yy=smooth(a,key,s);ax.plot(xx,yy,color=color,label=desc,lw=1.7)
 n=round(1/np.median(np.diff(a.time_s)));xx=np.convolve(a.time_s+4.9,np.ones(n)/n,mode='valid');yy=np.convolve(abs(a.vt_mm_s/a.radius_mm),np.ones(n)/n,mode='valid');axs[1,0].plot(xx,yy,color=color,label=desc,lw=1.7)
 p=laps(a);axs[1,1].plot(p[:,0],p[:,1],color=color,label=desc,lw=1.7)
 w=(p[:,2]>=35)&(p[:,3]<=72)
 rec_report={k:rec[k] for k in ['native_key','scale','config','quiet_speed_CV_mm_s','quiet_target_CV_mm_s','startup_speed_rmse_CV_mm_s','radius_rmse_CV_mm','overshoot_CV_mm','period_s']}
 rec_report.update(mean_complete_lap_period_s=float(p[w,1].mean()),period_error_percent=float((p[w,1].mean()/g.f.TARGET_PERIOD-1)*100),thresholds_pts_s={str(v):dict(video=threshold(T[:,2],v),model=threshold(np.array(rec['model_speed_CV']),v)) for v in [30,50,150,250]})
 half=next(x['record'] for x in validation if x['tag']==('wait_half' if label=='等待优先' else 'small_half'));long=next(x['record'] for x in validation if x['tag']==('wait_long' if label=='等待优先' else 'small_long'));ha=load(half);la=load(long);n=min(len(a),len(ha))-1;tail=la.time_s>=220
 rec_report['validation']=dict(half_dt_max_speed_difference_mm_s=float(abs(a.speed_mm_s.iloc[:n].to_numpy()-ha.speed_mm_s.iloc[:n].to_numpy()).max()),half_dt_max_radius_difference_mm=float(abs(a.radius_mm.iloc[:n].to_numpy()-ha.radius_mm.iloc[:n].to_numpy()).max()),completed_seconds=float(la.time_s.iloc[-1]),last20s_true_speed_mm_s=float(la.speed_mm_s[tail].mean()),last20s_true_radius_mm=float(la.radius_mm[tail].mean()),last20s_period_s=float(2*np.pi/abs(np.polyfit(la.time_s[tail],np.unwrap(la.phi_rad.to_numpy())[tail],1)[0])),rolling_fraction=float(1-la.sliding.mean()),status=long['status'])
 report['candidates'][label]=rec_report
axs[0,0].set(xlim=(5,72),ylabel='半径：映回原CV标度 (mm)',title='绝对半径允许统一放缩，相对轨迹形状仍受约束')
axs[0,1].set(xlim=(5,40),ylabel='速率：映回原CV标度 (mm/s)',title='等待后加速偏慢，未平移或缩放时间');axs[0,1].axvspan(9,18,color='#fde68a',alpha=.2)
axs[1,0].set(xlim=(7,40),ylabel='公转角速率 (rad/s)',title='尺度无关检验：速度/半径仍未完全对上')
axs[1,1].set(xlim=(20,72),ylim=(1.35,2.4),ylabel='逐圈周期 (s)',title='周期保持原始秒数，不参与空间放缩')
for ax in axs.flat:ax.grid(alpha=.25);ax.set_xlabel('视频时间 PTS (s)，4.9 s释放');ax.legend(fontsize=8.5)
fig.savefig(OUT/'comparison.png',dpi=155)
(OUT/'fit_report.json').write_text(json.dumps(report,indent=2))
(OUT/'candidate_parameters.json').write_text(json.dumps(dict(scale=wait['scale'],simulation=wait['config'],assessment=report['conclusion']),indent=2))
print(json.dumps(report,indent=2))
