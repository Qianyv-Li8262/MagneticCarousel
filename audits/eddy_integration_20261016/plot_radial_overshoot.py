from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'runs/video56c6-rest-release-20261017/overshoot'
a=np.genfromtxt(out/'trajectory.csv.radial.csv',names=True,delimiter=',');report=json.loads((out/'analysis.json').read_text());t=a['time_s']+4.9
n=round(1/np.median(np.diff(t)));sm=lambda v:np.convolve(v,np.ones(n)/n,mode='valid');tt=sm(t)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained');fig.suptitle('半径过冲：切向超速、向外滚动力矩与有限磁约束共同作用',fontsize=17)
ax=axs[0,0];ax.plot(tt,sm(a['r_mm']),color='#2563eb',label='半径');ax.set(ylabel='平均半径 (mm)',title='半径先回落，自旋约4秒后才反向')
bx=ax.twinx();bx.plot(tt,sm(a['wz_rad_s']),color='#ea580c',label='轴向自旋');bx.set_ylabel('轴向自旋 (rad/s)',color='#ea580c');bx.axhline(0,color='gray',lw=.7)
ax.axvline(report['mean_radius_peak']['pts_s'],color='#2563eb',ls=':');ax.axvline(report['spin_zero_pts_s'],color='#ea580c',ls=':');ax.legend(loc='upper left');bx.legend(loc='upper right')
ax=axs[0,1]
for key,color,label in [('memory_Qclock_mN','#ea580c','磁滞记忆：顺时针广义驱动'),('eddy_Qclock_mN','#2563eb','涡流：顺时针广义驱动')]:ax.plot(tt,sm(a[key]),color=color,label=label)
ax.plot(tt,-sm(a['roll_Qclock_mN']),color='#64748b',ls='--',label='滚阻幅值');ax.set(ylabel='切向广义力 (mN)',title='正自旋阶段先出现驱动峰值，随后驱动衰减');ax.legend(fontsize=8)
ax=axs[1,0]
for key,color,label in [('memory_Tr_mN','#ea580c','磁滞水平力矩 / 球半径'),('memory_Fr_mN','#059669','磁滞记忆的直接径向力'),('eddy_Tr_mN','#2563eb','涡流水平力矩 / 球半径')]:ax.plot(tt,sm(a[key]),color=color,label=label)
ax.axhline(0,color='gray',lw=.7);ax.set(ylabel='径向广义力 (mN)，向外为正',title='磁滞力矩由向外约+1.2 mN，最终转为向内约−1.25 mN');ax.legend(fontsize=8)
for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set(xlim=(15,48),xlabel='视频时间 PTS (s)');ax.axvspan(28,30,color='#fde68a',alpha=.2);ax.grid(alpha=.25)
f=np.load(out/'field_radial.npz');r=f['radius_mm'];mass=7850*4*np.pi*.006**3/3;meff=1.4*mass
ax=axs[1,1];ax.plot(r,-f['rev_force_mN'],color='#059669',lw=2,label='可逆磁场：平均向内拉力')
for v,color in [(303,'#2563eb'),(348,'#ea580c')]:ax.plot(r,meff*(v*.001)**2/(r*.001)*1000,color=color,ls='--',label=f'滚动所需向心项，v={v} mm/s')
ax.axvline(74.6,color='#2563eb',ls=':',label='最终半径约74.6 mm');ax.axvline(82.6,color='#ea580c',ls=':',label='过冲半径约82.6 mm');ax.set(xlim=(68,96),ylim=(0,20),xlabel='轨道半径 (mm)',ylabel='力 (mN)',title='平均磁拉力在约81 mm附近达到峰值，外侧开始减弱');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(out/'mechanism.png',dpi=155)
print(out/'mechanism.png')
