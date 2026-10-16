"""Compare the corrected 35 mm plate spacing (h=32 mm) to the old h=39.62 mm."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"scratch/eddy_integration_20261016"
OLD=DATA/"sigma12_start_rest_no_blow_long"
NEW=DATA/"height35_h32_rest_no_blow"
old=np.load(OLD/"trace.npz");new=np.load(NEW/"trace.npz")
half=np.load(DATA/"height35_h32_rest_no_blow_halfstep/trace.npz")
O=200*2*np.pi/60;vunit=17.5*O;Bunit=.0854985*4*np.pi*1e-7*1.1e6*1000
radius=np.linspace(35.01,104.99,600)
def field(path):
    nodes=np.fromfile(path,dtype="<f8").reshape(5,204,4)
    rho=radius/17.5;dr=4/203;x=(rho-2)/dr;j=np.clip(x.astype(int),0,202);u=x-j
    rms=np.zeros(len(radius));plus=minus=None
    for k in range(5):
        A,MA,Z,MZ=nodes[k,j].T;Ar,MAr,Zr,MZr=nodes[k,j+1].T
        b=Ar-A-dr*dr*(2*MA+MAr)/6;c=dr*dr*MA/2;d=dr*dr*(MAr-MA)/6
        Ap=(b+u*(2*c+3*d*u))/dr;A=A+u*(b+u*(c+u*d))
        bz=Zr-Z-dr*dr*(2*MZ+MZr)/6;cz=dr*dr*MZ/2;dz=dr*dr*(MZr-MZ)/6
        Z=Z+u*(bz+u*(cz+u*dz));m=4*(2*k+1)
        rms+=(Ap*Ap+(m*A/rho)**2+Z*Z)/2
        if k==0:plus=np.abs((-Ap-m*A/rho)/2)*Bunit;minus=np.abs((-Ap+m*A/rho)/2)*Bunit
    return np.sqrt(rms)*Bunit,plus,minus
oldfield=field(DATA/"height35_backup/field_lut_h39p62_fp64.bin")
newfield=field(ROOT/"build/release/field-lut/field_lut_fp64.bin")
n=min(len(new["time_2d"]),len(half["time_2d"]))
stats={"surface_spacing_mm":35,"ball_center_z_mm":41,"magnet_center_z_mm":9,"h_mm":32,
       "dimensionless_H":32/17.5,"sigma_S_m":1.2e7,
       "ring_old_B_rms_mT":float(np.interp(70,radius,oldfield[0])),
       "ring_new_B_rms_mT":float(np.interp(70,radius,newfield[0])),
       "halfstep_max_speed_difference_mm_s":float(np.max(abs(np.linalg.norm(new["state"][:n,3:5],axis=1)-np.linalg.norm(half["state"][:n,3:5],axis=1)))*vunit),
       "halfstep_max_radius_difference_mm":float(np.max(abs(np.linalg.norm(new["state"][:n,1:3],axis=1)-np.linalg.norm(half["state"][:n,1:3],axis=1)))*17.5)}
(NEW/"height_comparison.json").write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("表面间距改为 35 mm：球心—磁铁中心高度 h = 32 mm",fontsize=18,fontweight="bold")
fig.supxlabel("当前静止初态、无预磁化、无吹气；σ = 1.2×10⁷ S/m，200 rpm。新轨迹在 t2D ≈ 111.8 越出磁场表后终止。",fontsize=10)
blue,orange="#2563eb","#ea580c"
for ax in axs.flat:ax.grid(color="#e2e8f0",linewidth=.7)
for d,color,label in [(old,blue,"旧高度 h=39.62 mm"),(new,orange,"新高度 h=32 mm")]:
    ts=d["time_2d"];s=d["state"]
    axs[0,0].plot(ts,np.linalg.norm(s[:,3:5],axis=1)*vunit,color=color,label=label,linewidth=1)
    axs[0,1].plot(ts,np.linalg.norm(s[:,1:3],axis=1)*17.5,color=color,linewidth=1)
    axs[1,0].plot(ts,s[:,11]*O,color=color,linewidth=1)
axs[0,0].set(title="新高度下更早、更强地加速",xlabel="无量纲时间 t2D",ylabel="质心速率 (mm/s)",xlim=(0,130),ylim=(0,580))
axs[0,0].legend(fontsize=9,frameon=False)
axs[0,1].set(title="半径向外增大，触及 105 mm 表边界",xlabel="无量纲时间 t2D",ylabel="半径 (mm)",xlim=(0,130),ylim=(60,110))
axs[0,1].axhline(105,color="#64748b",linestyle="--",linewidth=1)
axs[1,0].set(title="竖直自旋也经历快速变化",xlabel="无量纲时间 t2D",ylabel="竖直自旋 (rad/s)",xlim=(0,130),ylim=(-15,100))
axs[1,1].plot(radius,oldfield[0],color=blue,label="旧高度")
axs[1,1].plot(radius,newfield[0],color=orange,label="新高度")
axs[1,1].axvline(70,color="#64748b",linestyle=":",linewidth=1)
axs[1,1].set(title="源磁场 RMS 的径向分布",xlabel="轨道半径 (mm)",ylabel="磁感应强度 RMS (mT)",xlim=(35,105))
axs[1,1].legend(fontsize=9,frameon=False)
fig.savefig(NEW/"height35_comparison.png",dpi=165,facecolor=fig.get_facecolor())
print(NEW/"height35_comparison.png")
