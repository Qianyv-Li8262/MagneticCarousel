"""Comparison figure for the restored-moment stable negative-spin branch."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"scratch/eddy_integration_20261016/restored_moment_sigma9_h32"
d=np.load(OUT/"trace.npz");p=np.load(OUT/"capture_phase.npz")
analysis=json.loads((OUT/"restored_analysis.json").read_text())
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8.3),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("恢复磁矩后：转入可恢复的负向自旋支路",fontsize=19,fontweight="bold")
fig.supxlabel("h=32 mm，静止释放、无预磁化、无吹气。当前 β=0.0569993，σ=9×10^6 S/m；原生产参数未作修改。",fontsize=10)
colors={"moment_restored_sigma9":"#2563eb","moment_restored_sigma12":"#059669", "moment_high_sigma9":"#ea580c","moment_high_sigma12":"#dc2626"}
labels={"moment_restored_sigma9":"恢复磁矩，σ=9×10^6", "moment_restored_sigma12":"恢复磁矩，σ=1.2×10^7",
        "moment_high_sigma9":"高磁矩，σ=9×10^6（到达表边界）", "moment_high_sigma12":"高磁矩，σ=1.2×10^7（到达表边界）"}
for ax in axs.flat:ax.grid(color="#e2e8f0",linewidth=.7)
for name in colors:
    table=np.genfromtxt(OUT/f"{name}.csv",names=True,delimiter=",")
    for ax,key in [(axs[0,0],"speed_mm_s"),(axs[0,1],"wz_rad_s"),(axs[1,0],"radius_mm")]:
        ax.plot(table["time"],table[key],color=colors[name],linewidth=1,label=labels[name])
        ax.set(xlim=(0,2000),xlabel="无量纲时间 t2D")
axs[0,0].set(title="低磁矩两组均保持高速；高磁矩两组先越界",ylabel="质心速率 (mm/s)",ylim=(0,570))
axs[0,0].legend(fontsize=8,frameon=False)
axs[0,1].axhline(0,color="#64748b",linewidth=1)
axs[0,1].set(title="正自旋平台结束后，转入负自旋平台",ylabel="竖直自旋 (rad/s)",ylim=(-110,110))
axs[1,0].axhline(105,color="#64748b",linestyle="--",linewidth=1)
axs[1,0].set(title="当前半径约 74.9 mm，轨道不再持续外移",ylabel="轨道半径 (mm)",ylim=(65,108))
n=201;smooth=lambda x:np.convolve(x,np.ones(n)/n,mode="valid")
tx=p["time_2d"][n//2:-(n//2)]
moment=p["demodulated_memory"]*p["positive_h"]/abs(p["positive_h"])
positive=smooth(moment*np.conj(p["positive_h"]))
negative=smooth(moment*np.conj(p["negative_h"]))
ax=axs[1,1]
ax.plot(tx,-np.angle(positive)*180/np.pi,color="#ea580c",label="相对正旋场的响应相位")
ax.plot(tx,np.angle(negative)*180/np.pi,color="#2563eb",label="相对反旋场的响应相位")
ax.axhline(90,color="#64748b",linestyle=":",linewidth=1)
ax.set(title="反旋分量的磁滞响应相位保持约 31°",xlabel="无量纲时间 t2D",ylabel="按各自旋向定义的相位滞后 (°)",xlim=(200,2000),ylim=(0,185))
ax.legend(fontsize=8,frameon=False)
ax.text(.98,.42,"相位指磁滞记忆分量\n局部扰动后也能回到同一运行范围",ha="right",transform=ax.transAxes,fontsize=10)
fig.savefig(OUT/"restored_moment_comparison.png",dpi=165,facecolor=fig.get_facecolor())
print(OUT/"restored_moment_comparison.png")

coef=4*np.pi*1.1e6/(4/.07)**3
steady=analysis["current_windows"][-1]
report={"moment_old_Am2":.0854985*coef,"moment_current_Am2":.0569993*coef,
    "relative_moment":.0569993/.0854985,"relative_quadratic_field_scale":(.0569993/.0854985)**2,
    "steady_force_balance_mN":{k:steady[k] for k in ["hysteresis_drive_mN","reversible_drive_mN","eddy_drive_mN","roll_loss_mN","air_loss_mN","net_drive_mN"]},
    "steady_axial_torque_uNm":{k:steady[k] for k in ["JA_torque_z_uNm","eddy_torque_z_uNm","net_Mz_uNm"]}}
(OUT/"physical_balance.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
