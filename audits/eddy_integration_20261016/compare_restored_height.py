from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('scratch/eddy_integration_20261016');out=root/'restored_moment_sigma9_h39p62'
a=np.genfromtxt(out/'trajectory.csv',delimiter=',',names=True)
b=np.genfromtxt(root/'restored_moment_sigma9_h32/moment_restored_sigma9.csv',delimiter=',',names=True)
r={'h_mm':39.62,'beta':.0569993,'sigma_S_m':9e6,'end_t2d':float(a['time'][-1]),'peak_speed_mm_s':float(a['speed_mm_s'].max()),'windows':[]}
for lo,hi in [(500,1000),(1500,2000),(3500,4000),(4500,5000)]:
 s=a[(a['time']>=lo)&(a['time']<=hi)]
 d={'interval':[lo,hi]}
 for k in a.dtype.names:
  if k!='time':d[k]={'mean':float(s[k].mean()),'min':float(s[k].min()),'max':float(s[k].max())}
 d['net_drive_mN']=float(np.mean(s['JA_drive_mN']+s['eddy_drive_mN']-s['rolling_loss_mN']-s['air_loss_mN']))
 d['net_Mz_uNm']=float(np.mean(s['JA_Mz_uNm']+s['eddy_Mz_uNm']))
 r['windows'].append(d)
(out/'summary.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11})
fig,axs=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
for ax,k,y in zip(axs,['speed_mm_s','wz_rad_s','radius_mm'],['质心速率 (mm/s)','竖直自旋 (rad/s)','轨道半径 (mm)']):
 ax.plot(a['time'],a[k],label='原较大高度 h=39.62 mm',lw=1,color='#2563eb')
 ax.plot(b['time'],b[k],label='当前高度 h=32 mm（对照至 t=2000）',lw=1,color='#ea580c',alpha=.8)
 ax.set_ylabel(y);ax.grid(alpha=.25)
axs[0].legend();axs[1].axhline(0,color='gray',lw=.7)
axs[-1].set_xlabel('无量纲时间 t2D（5000 对应约 239 秒）')
fig.suptitle('正常磁矩、相同电导率：恢复较大高度后的长期运行',fontsize=16)
fig.savefig(out/'height_comparison.png',dpi=150)
