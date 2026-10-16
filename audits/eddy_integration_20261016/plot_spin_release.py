"""Inspect loss of near-synchronism around t2D=800--900 in the unforced run."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"scratch/eddy_integration_20261016"
OUT=DATA/"sigma12_start_rest_no_blow_long"
d=np.load(OUT/"trace.npz");t=d["time_2d"];s=d["state"]
a=np.genfromtxt(OUT/"drive.csv",delimiter=",",names=True)
O=200*2*np.pi/60;r=np.linalg.norm(s[:,1:3],axis=1)
pd=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/r**2
wr=(s[:,1]*s[:,9]+s[:,2]*s[:,10])/r
wp=(-s[:,2]*s[:,9]+s[:,1]*s[:,10])/r
wz=s[:,11]*O
W=np.sqrt(wr**2+wp**2+(s[:,11]-pd)**2)*O
H=4*(1-pd)*O
required=(pd+np.sqrt(np.maximum(0,16*(1-pd)**2-wr**2-wp**2)))*O
I=.4*(7850*4*np.pi*.006**3/3)*.006**2
records=[]
for low,high in [(750,775),(800,810),(830,840),(850,860),(860,875),(875,900),(900,925)]:
    m=(t>=low-1e-8)&(t<high)
    mean=lambda name:float(a[name][m].mean())
    rec={"interval_t2d":[low,high],"wz_rad_s":float(wz[m].mean()),
        "near_sync_required_wz_rad_s":float(required[m].mean()),"detuning_rad_s":float((W-H)[m].mean()),
        **{key:mean(key) for key in ["speed_mm_s","radius_mm","hysteresis_drive_mN","eddy_drive_mN",
                                    "JA_torque_z_uNm","eddy_torque_z_uNm","h_norm"]}}
    rec["net_torque_z_uNm"]=mean("JA_torque_z_uNm")+mean("eddy_torque_z_uNm")
    rec["torque_predicted_domega_z_dt_rad_s2"]=rec["net_torque_z_uNm"]*1e-6/I
    ix=np.flatnonzero(m)
    rec["observed_domega_z_dt_rad_s2"]=float((wz[ix[-1]]-wz[ix[0]])/((t[ix[-1]]-t[ix[0]])/O))
    records.append(rec)
half=np.load(DATA/"sigma12_start_rest_no_blow_halfstep_1000/trace.npz")
n=len(half["time_2d"]);mask=(t[:n]>=750)&(t[:n]<=950)
report={"windows":records,"halfstep_max_wz_difference_rad_s":float(np.max(abs(wz[:n][mask]-half["state"][:n,11][mask]*O))),
    "halfstep_max_COM_speed_difference_mm_s":float(np.max(abs(np.linalg.norm(s[:n,3:5][mask],axis=1)-np.linalg.norm(half["state"][:n,3:5][mask],axis=1)))*17.5*O)}
(OUT/"spin_release_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("800–900：近同步力矩平衡失效，竖直自旋快速下降",fontsize=18,fontweight="bold")
fig.supxlabel("当前无吹气配置，σ = 1.2×10⁷ S/m；阴影 850–875。频率匹配指标只用于诊断，不等同于严格锁相证明。",fontsize=10)
blue,orange,green="#2563eb","#ea580c","#059669"
window=101
smooth=lambda x:np.convolve(x,np.ones(window)/window,mode="valid")
tx=t[window//2:-(window//2)]
for ax in axs.flat:
    ax.grid(color="#e2e8f0",linewidth=.7)
    ax.axvspan(850,875,color="#fed7aa",alpha=.5)
    ax.set(xlim=(700,950),xlabel="无量纲时间 t2D")
ax=axs[0,0]
ax.plot(tx,smooth(wz),color=blue,label="实际竖直自旋")
ax.plot(tx,smooth(required),color=orange,linestyle="--",label="计入滚动自旋后的近同步所需分量")
ax.set(title="初期小幅下降含几何因素；后期明显脱离",ylabel="竖直自旋 (rad/s)",ylim=(20,92))
ax.legend(fontsize=9,frameon=False)
ax=axs[0,1]
ax.plot(tx,smooth(W-H),color=blue)
ax.axhline(0,color="#64748b",linestyle="--",linewidth=1)
ax.set(title="完整自旋矢量与主谐波的失谐持续增大",ylabel="相对自旋速率 − 主谐波频率 (rad/s)",ylim=(-50,5))
ax=axs[1,0]
for key,color,label in [("JA_torque_z_uNm",blue,"磁滞竖直驱动力矩"),("eddy_torque_z_uNm",green,"涡流竖直力矩")]:
    ax.plot(tx,smooth(a[key]),color=color,label=label)
ax.plot(tx,smooth(a["JA_torque_z_uNm"]+a["eddy_torque_z_uNm"]),color=orange,label="净竖直力矩")
ax.axhline(0,color="#64748b",linewidth=1)
ax.set(title="磁滞支撑力矩塌落，涡流制动仍然存在",ylabel="竖直轴力矩 (μN·m)",ylim=(-1.7,1.8))
ax.legend(fontsize=9,frameon=False)
ax=axs[1,1]
for key,color,label in [("hysteresis_drive_mN",blue,"磁滞广义驱动"),("eddy_drive_mN",green,"涡流广义驱动"),("roll_loss_mN","#64748b","滚阻大小")]:
    ax.plot(tx,smooth(a[key]),color=color,label=label)
ax.set(title="随后磁滞平动驱动也下降，高速运动开始衰减",ylabel="沿速度方向的广义力 (mN)",ylim=(0,.7))
ax.legend(fontsize=9,frameon=False)
fig.savefig(OUT/"spin_release_800_900.png",dpi=165,facecolor=fig.get_facecolor())
print(OUT/"spin_release_800_900.png")
