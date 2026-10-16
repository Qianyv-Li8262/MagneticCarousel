"""The user's corrected blow interval: t2D=200 through 300, enabled model."""
from pathlib import Path
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"scratch/eddy_integration_20261016"
OUT=DATA/"eddy_on_blow_200_300_long"
trace=np.load(OUT/"trace.npz")
obs=np.genfromtxt(OUT/"observables.csv",delimiter=",",names=True)
rows=np.column_stack((trace["time_2d"],trace["state"],trace["blow_mN"]))
rows.astype("<f8").tofile(OUT/"drive_input.bin")
manifest=json.loads((ROOT/"build/release/extensions/fp64/build_config.json").read_text())
assert manifest["eddy_currents"]["enabled"]
subprocess.run([str(DATA/"diagnose_drive.exe"),str(OUT/"drive_input.bin"),manifest["lut"],str(OUT/"drive.csv")],check=True)
drive=np.genfromtxt(OUT/"drive.csv",delimiter=",",names=True)
coast=drive[(drive["t2d"]>=300-1e-8)&(drive["t2d"]<340)]
stats={key:float(np.mean(coast[key])) for key in drive.dtype.names}
stats["interval_t2d"]=[300,340]
stats["net_mN"]=stats["JA_drive_mN"]+stats["eddy_drive_mN"]-stats["roll_loss_mN"]-stats["air_loss_mN"]
(OUT/"coast_force_summary.json").write_text(json.dumps(stats,indent=2))
half=np.genfromtxt(DATA/"eddy_on_blow_200_300_halfstep/observables.csv",delimiter=",",names=True)
first=obs[:len(half)]
check={"max_speed_difference_mm_s":float(np.max(abs(first["speed_mm_s"]-half["speed_mm_s"]))),
       "max_radius_difference_mm":float(np.max(abs(first["radius_mm"]-half["radius_mm"])))}
(OUT/"halfstep_check.json").write_text(json.dumps(check,indent=2))
print(json.dumps({"coast":stats,"halfstep":check},indent=2))

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("涡流开启 · 吹气 200–300：停吹后仍回到慢速运动",fontsize=19,fontweight="bold")
fig.supxlabel("初态 v0 = −0.05，S0 = 0，相位 0.58，预磁化 2 圈；sr = 1，sω = 2；吹气 0.7 mN，实际时间 9.55–14.32 s",fontsize=10)
blue,orange,green="#2563eb","#ea580c","#059669"
for ax in axs.flat:
    ax.grid(color="#e2e8f0",linewidth=.7)
for ax in (axs[0,0],axs[0,1],axs[1,0]):
    ax.axvspan(200,300,color="#fed7aa",alpha=.6)
    ax.set_xlabel("无量纲时间 t2D")
ax=axs[0,0]
ax.plot(obs["time_2d"],obs["speed_mm_s"],color=blue,linewidth=1)
ax.set(title="速度：停吹瞬间 243.85 mm/s",ylabel="质心速率 (mm/s)",xlim=(0,650))
off=np.argmin(abs(obs["time_2d"]-300))
ax.annotate("停吹",xy=(300,obs["speed_mm_s"][off]),xytext=(390,210),arrowprops={"arrowstyle":"->"})
ax.text(.98,.45,"峰值 249.95 mm/s\n橙色区域：吹气开启",transform=ax.transAxes,ha="right",fontsize=10)
ax=axs[0,1]
ax.plot(obs["time_2d"],obs["radius_mm"],color=blue,linewidth=1)
ax.set(title="轨道半径：本次没有越出磁场表",ylabel="半径 (mm)",xlim=(0,650))
ax.axhline(70,color="#64748b",linestyle=":",linewidth=1)
ax=axs[1,0]
ax.plot(obs["time_2d"],obs["speed_mm_s"],color=green,linewidth=.8)
ax.set(title="延长至 95.49 s：末段均速 5.17 mm/s",ylabel="质心速率 (mm/s)",xlim=(0,2000))
ax.text(.98,.7,"t2D = 1950–2000 的平均值\n全程纯滚动，无求解失败",transform=ax.transAxes,ha="right",fontsize=10)
ax=axs[1,1]
vals=[stats["JA_drive_mN"],stats["eddy_drive_mN"],-stats["roll_loss_mN"],stats["net_mN"]]
bars=ax.bar(["原磁模型","涡流增量","滚阻","净值"],vals,color=[blue,green,"#64748b",orange],width=.6)
ax.axhline(0,color="#334155",linewidth=1)
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,val+(.012 if val>=0 else -.012),f"{val:+.3f}",ha="center",va="bottom" if val>=0 else "top")
ax.set(title="停吹初段：磁驱动仍不足以抵消滚阻",ylabel="沿速度方向的等效驱动力 (mN)",ylim=(-.65,.4))
ax.text(.98,.97,f"t2D = 300–340；均速 {stats['speed_mm_s']:.1f} mm/s\n计入力和滚动力矩；净值含空气阻力",transform=ax.transAxes,ha="right",va="top",fontsize=9)
fig.savefig(OUT/"blow_200_300.png",dpi=165,facecolor=fig.get_facecolor())
print(OUT/"blow_200_300.png")
