"""Plot field helicity, memory phase and a physical radial-guide diagnostic."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"scratch/eddy_integration_20261016/sigma12_start_rest_no_blow_long"
phase=np.load(OUT/"capture_phase.npz");trace=np.load(OUT/"trace.npz")
t=phase["time_2d"];s=trace["state"]
a=np.genfromtxt(OUT/"drive.csv",delimiter=",",names=True)
guide=np.genfromtxt(OUT/"radial_guide_2.csv",delimiter=",",names=True)
n=201; smooth=lambda x:np.convolve(x,np.ones(n)/n,mode="valid");tx=t[n//2:-(n//2)]
weighted=phase["demodulated_memory"]*abs(phase["positive_h"])
z=smooth(weighted);scale=float(phase["scale"])
A=scale*abs(z);lag=-np.angle(z)*180/np.pi
other=a["JA_torque_z_uNm"]-phase["plus_torque"]
load=-(a["eddy_torque_z_uNm"]+other)
Bunit=4*np.pi*1e-7*1.1e6*1000
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8.2),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("从维持到崩落：轨道外移改变旋转场，响应相位耗尽补偿余量",fontsize=18,fontweight="bold")
fig.supxlabel("无吹气、σ = 1.2×10⁷ S/m。相位指磁滞记忆分量相对同向旋转场的相位；不把 90°直接认作严格稳定性边界。",fontsize=9)
blue,orange,green="#2563eb","#ea580c","#059669"
for ax in axs.flat:
    ax.grid(color="#e2e8f0",linewidth=.7)
    ax.axvspan(830,860,color="#fed7aa",alpha=.5)
    ax.set(xlim=(400,1000),xlabel="无量纲时间 t2D")
ax=axs[0,0]
ax.plot(t,abs(phase["positive_h"])*Bunit,color=blue,label="与正向自旋同向的主谐波分量")
ax.plot(t,abs(phase["negative_h"])*Bunit,color=green,label="反向旋转的主谐波分量")
ax.set(title="向外移动：同向场减弱，反向场增强",ylabel="旋转场分量幅值 (mT)",ylim=(0,9))
ax.legend(fontsize=9,frameon=False)
ax=axs[0,1]
ax.plot(tx,lag,color=blue,linewidth=1)
ax.axhline(90,color="#64748b",linestyle="--",linewidth=1)
ax.set(title="先增大相位滞后来补偿，之后越过正交相位",ylabel="磁滞响应相位滞后 (°)",ylim=(20,190))
ax=axs[1,0]
ax.plot(tx,A,color=blue,label="观测相干分量的力矩幅值尺度 A")
ax.plot(tx,smooth(phase["plus_torque"]),color=green,label="该分量实际提供的竖直力矩")
ax.plot(tx,smooth(load),color=orange,label="抵消其余竖直力矩所需的支撑")
ax.set(title="场变弱后，单靠调相位也难补足负载",ylabel="竖直轴力矩 (μN·m)",ylim=(0,2.3))
ax.legend(fontsize=9,frameon=False)
ax=axs[1,1]
ax.plot(t,s[:,11]*(200*2*np.pi/60),color=blue,label="原轨迹：允许径向漂移")
ax.plot(guide["time_2d"],guide["wz_rad_s"],color=orange,label="t=500 后加理想径向约束，r=69.8 mm")
ax.set(title="物理对照：固定半径后，本时段自旋崩落消失",ylabel="竖直自旋 (rad/s)",ylim=(-5,100))
ax.legend(fontsize=9,frameon=False)
fig.savefig(OUT/"capture_physics.png",dpi=165,facecolor=fig.get_facecolor())
print(OUT/"capture_physics.png")
