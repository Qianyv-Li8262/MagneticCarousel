"""Report successful checks and remaining mismatch without fitting a time shift."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_video56c6 as f
OUT=f.OUT
records=json.loads((OUT/'validation.json').read_text());A=next(r for r in records if r['key']=='a13139fee901d312');B=next(r for r in records if r['key']=='53952c4e7b3a674a')
cols=['#2563eb','#ea580c'];labels=['A：周期优先','B：启动曲线优先']
def load(rec):return np.genfromtxt(OUT/(rec['key']+'.csv'),names=True,delimiter=',')
def periods(a):
 phi=-np.unwrap(a['phi_rad']);levels=2*np.pi*np.arange(int(np.floor(phi.min()/(2*np.pi)))+1,int(np.floor(phi.max()/(2*np.pi)))+1)
 # All late trajectories circulate clockwise; interpolation is confined to this monotone stage.
 mask=a['time_s']>15;ph=phi[mask];t=a['time_s'][mask]+f.RELEASE
 ph=np.maximum.accumulate(ph);levels=levels[(levels>ph[0])&(levels<ph[-1])]
 cross=np.interp(levels,ph,t)
 return np.column_stack(((cross[1:]+cross[:-1])/2,np.diff(cross),cross[:-1],cross[1:]))
report=dict(video=f.VIDEO.name,release_pts_s=f.RELEASE,target_period_s=f.TARGET_PERIOD,period_window_pts_s=[35,72],candidates={},verification={})
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11})
fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
valid=f.DATA['derivative_valid']==1;pt=f.DATA['source_pts_s']
axs[0].plot(pt[valid],f.DATA['com_speed_mm_s'][valid],color='#94a3b8',lw=.6,alpha=.45,label='视频：有效速度样本')
axs[0].plot([b['pts'] for b in f.BINS],[b['speed'] for b in f.BINS],'k.-',lw=1.5,label='视频：1秒平均')
validp=(f.LAPS['start_pts_s']>=20)&(f.LAPS['end_pts_s']<=72)
axs[1].plot((f.LAPS['start_pts_s'][validp]+f.LAPS['end_pts_s'][validp])/2,f.LAPS['period_s'][validp],'ko',ms=4,label='视频：逐圈周期')
axs[1].axhline(f.TARGET_PERIOD,color='#64748b',ls=':',label=f'视频平均 {f.TARGET_PERIOD:.3f} s')
rv=f.DATA['position_valid']==1
for lo in np.arange(6,72,1):
 w=rv&(pt>=lo)&(pt<lo+1)
 if w.sum()>5:axs[2].plot(lo+.5,np.nanmean(f.DATA['r_mm'][w]),'k.',ms=5)
for rec,color,label in zip([A,B],cols,labels):
 a=load(rec);tt=a['time_s']+f.RELEASE;n=round(1/np.median(np.diff(tt)));kernel=np.ones(n)/n
 st=np.convolve(tt,kernel,mode='valid');sv=np.convolve(a['speed_mm_s'],kernel,mode='valid');sr=np.convolve(a['radius_mm'],kernel,mode='valid')
 axs[0].plot(st,sv,color=color,lw=1.7,label=label+'（1秒平均）')
 pp=periods(a);axs[1].plot(pp[:,0],pp[:,1],color=color,lw=1.6,label=label)
 axs[2].plot(st,sr,color=color,lw=1.6,label=label)
 w=(pp[:,2]>=35)&(pp[:,3]<=72);pl=pp[w,1]
 report['candidates'][label]=dict(key=rec['key'],parameters=rec['config'],startup_1s_curve_rmse_mm_s=rec['rmse_curve_mm_s'],period_mean_s=float(pl.mean()),period_error_percent=float((pl.mean()/f.TARGET_PERIOD-1)*100),late_metrics=rec['holdout'])
for ax in axs:ax.grid(alpha=.25);ax.set_xlim(5,72)
axs[0].axvspan(17,19,color='#fbbf24',alpha=.12);axs[0].set(ylabel='质心速率 (mm/s)',ylim=(-5,385),title='启动的真实时间关系：没有额外平移模拟曲线');axs[0].legend(fontsize=9,loc='lower right')
axs[1].set(ylabel='公转周期 (s)',ylim=(1.35,2.4),title='周期可以对上，但启动拟合更好的候选最终偏慢');axs[1].legend(fontsize=9)
axs[2].set(ylabel='轨道半径 (mm)',xlabel='视频时间 PTS (s)；模拟从视频 4.9 s 释放时开始',ylim=(63,86),title='平均径向轨迹：模型早段平衡半径仍偏大');axs[2].legend(fontsize=9)
fig.suptitle('低高度 h=32 mm：56C6 视频的启发式拟合与剩余偏差',fontsize=17)
fig.savefig(OUT/'comparison.png',dpi=155)
for tag,key in [('A','a13139fee901d312'),('B','53952c4e7b3a674a')]:
 base=next(r for r in records if r['key']==key);half=next(r for r in records if r['tag']==tag+'_half');long=next(r for r in records if r['tag']==tag+'_long')
 a,b,z=map(load,[base,half,long]);w=z['time_s']>=220
 report['verification'][tag]=dict(half_dt_max_speed_difference_mm_s=float(abs(a['speed_mm_s']-b['speed_mm_s']).max()),half_dt_max_radius_difference_mm=float(abs(a['radius_mm']-b['radius_mm']).max()),completed_seconds=float(z['time_s'][-1]),last20s_speed_mm_s=float(z['speed_mm_s'][w].mean()),last20s_radius_mm=float(z['radius_mm'][w].mean()),status=long['status'])
base=load(next(r for r in records if r['tag']=='baseline_check'));old=np.genfromtxt(f.ROOT/'scratch/eddy_integration_20261016/restored_moment_sigma9_h32/moment_restored_sigma9.csv',names=True,delimiter=',')[::2];n=min(len(base),len(old))-1
report['verification']['baseline_common_sample_max_speed_difference_mm_s']=float(abs(base['speed_mm_s'][:n]-old['speed_mm_s'][:n]).max())
report['conclusion']='No joint fit meeting both startup and mean-period targets found in this heuristic search. Candidate A fits period; candidate B improves startup but retains early creep, overshoot and a late period bias. Not a proof that all allowed parameter combinations fail.'
report['search_completed_runs']=len(list(OUT.glob('????????????????.json')))
(OUT/'fit_report.json').write_text(json.dumps(report,indent=2))
(OUT/'targets.json').write_text(json.dumps(dict(video=f.VIDEO.name,release_pts_s=f.RELEASE,period_window_pts_s=[35,72],target_period_s=f.TARGET_PERIOD,bins=f.BINS,ignored='third harmonic; no tilt, no RPM fit, no friction fit, no arbitrary time shift'),indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src=f.SOURCE
(src/'manifest.json').write_text(json.dumps(dict(h_mm=32,beta=.0569993,eddy_modes=12,production_untouched=True,source_hashes={str(p.relative_to(src)):sha(p) for p in (src/'kernels').rglob('*') if p.is_file()},harness_sha256=sha(f.ROOT/'tests/native/fit_video56c6.cpp'),binary_sha256=sha(src/'fit_video56c6.exe'),lut_sha256=sha(src/'field_lut_fp64.bin')),indent=2))
print(json.dumps(report,indent=2))
