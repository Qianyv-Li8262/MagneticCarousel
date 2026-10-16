"""Diagnose the unforced sigma=12 MS/m startup transition around t2D=400."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import animate_one_trajectory as animation
DATA=ROOT/"scratch/eddy_integration_20261016"
OUT=DATA/"sigma12_start_rest_no_blow_long"
trace=np.load(OUT/"trace.npz")
t=trace["time_2d"]; s=trace["state"]
drive=np.genfromtxt(OUT/"drive.csv",delimiter=",",names=True)
O=200*2*np.pi/60
r=np.linalg.norm(s[:,1:3],axis=1)
speed=np.linalg.norm(s[:,3:5],axis=1)*.0175*O*1000
wr=(s[:,1]*s[:,9]+s[:,2]*s[:,10])/r
wp=(-s[:,2]*s[:,9]+s[:,1]*s[:,10])/r
phidot=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/r**2
# Relative spin magnitude in an orbit-attached frame; the fundamental field
# frequency in that frame is m*(1-phidot), m=4. This is a local detuning
# diagnostic, not a proof of phase locking on the fully nonuniform trajectory.
relative_spin=np.sqrt(wr*wr+wp*wp+(s[:,11]-phidot)**2)*O
field_frequency=4*(1-phidot)*O

module=animation.load_hysteresis()
params=module.SolverParams(sr=animation.SR,somega=animation.ANIMATION_SOMEGA)
assert animation.ANIMATION_V0==0 and animation.MAGNETIZE_TURNS==0
state=animation.prepare_initial_state(module,params,.0005,v0=0,s0=0,phi0=0,
    magnetize_turns=0,disk_phase=animation.DISK_PHASE)
states=[state.copy()]; control_t=[0.]
mode=direction=None
for step in range(16000):
    state,mode,direction=module.evolve_trajectory(state,.0005,100,params,mode,direction,step*100)
    # Artificial external constraint for diagnosis only; production is untouched.
    state[11]=0
    states.append(state.copy()); control_t.append((step+1)*.05)
states=np.asarray(states); control_t=np.asarray(control_t)
control_speed=np.linalg.norm(states[:,3:5],axis=1)*.0175*O*1000
np.savez_compressed(OUT/"fixed_wz_control.npz",time_2d=control_t,state=states)

half=np.load(DATA/"sigma12_start_rest_no_blow_halfstep/trace.npz")
n=len(half["time_2d"])
half_speed=np.linalg.norm(half["state"][:,3:5],axis=1)*.0175*O*1000
records=[]
for low,high in [(300,325),(350,375),(375,390),(390,400),(400,410),(410,425),(425,450),(750,800),(1950,2000)]:
    mask=(t>=low-1e-8)&(t<high)
    mean=lambda key:float(np.mean(drive[key][mask]))
    record={"interval_t2d":[low,high],"speed_mm_s":float(speed[mask].mean()),
        "wz_rad_s":float(np.mean(s[mask,11])*O),"q_norm":float(np.linalg.norm(s[mask,12:15],axis=1).mean()),
        "relative_spin_rad_s":float(relative_spin[mask].mean()),"field_frequency_rad_s":float(field_frequency[mask].mean()),
        **{key:mean(key) for key in ["hysteresis_drive_mN","eddy_drive_mN","reversible_drive_mN",
            "roll_loss_mN","JA_torque_z_uNm","eddy_torque_z_uNm"]}}
    record["net_mN"]=mean("JA_drive_mN")+mean("eddy_drive_mN")-mean("roll_loss_mN")-mean("air_loss_mN")
    records.append(record)
report={"configuration":{"sigma_S_m":1.2e7,"v0":0,"magnetize_turns":0,"blowing":False,
    "sr":1,"somega":2,"phase":animation.DISK_PHASE},"windows":records,
    "control_fixed_wz_tail_mean_speed_mm_s":float(control_speed[control_t>=750].mean()),
    "control_fixed_wz_max_speed_mm_s":float(control_speed.max()),
    "halfstep_max_speed_difference_mm_s":float(np.max(abs(speed[:n]-half_speed))),
    "all_accepted_samples_rolling":bool(not np.any(trace["sliding"]))}
(OUT/"spin_transition_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig,axs=plt.subplots(2,2,figsize=(13,8.4),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("t ≈ 400 的加速：竖直自旋积累，磁滞驱动增强",fontsize=19,fontweight="bold")
fig.supxlabel("当前配置：σ = 1.2×10⁷ S/m；静止释放、无预磁化、无吹气；全程纯滚动。阴影标出 t2D = 390–410。",fontsize=10)
blue,orange,green="#2563eb","#ea580c","#059669"
for ax in axs.flat:
    ax.grid(color="#e2e8f0",linewidth=.7)
    ax.axvspan(390,410,color="#fed7aa",alpha=.55)
    ax.set_xlabel("无量纲时间 t2D")
    ax.set_xlim(0,800)
ax=axs[0,0]
ax.plot(t,speed,color=blue,linewidth=1,label="当前模型")
ax.plot(control_t,control_speed,color="#64748b",linewidth=.8,label="诊断对照：人为固定竖直自旋为 0")
ax.set(title="相同初态：固定竖直自旋后不再突破低速状态",ylabel="质心速率 (mm/s)")
ax.legend(fontsize=9,frameon=False)
ax=axs[0,1]
def smooth(x,window=101):
    return np.convolve(x,np.ones(window)/window,mode="valid")
tx=t[50:-50]
ax.plot(tx,smooth(relative_spin),color=blue,label="球相对轨道坐标系的自旋速率")
ax.plot(tx,smooth(field_frequency),color=orange,linestyle="--",label="磁场主谐波频率")
ax.set(title="400 附近：自旋频率接近并穿过主谐波频率",ylabel="角频率 (rad/s)")
ax.legend(fontsize=9,frameon=False)
ax=axs[1,0]
for key,color,label in [("hysteresis_drive_mN",blue,"磁滞记忆贡献"),("eddy_drive_mN",green,"涡流贡献"),
                         ("roll_loss_mN","#64748b","滚阻大小")]:
    ax.plot(tx,smooth(drive[key]),color=color,label=label,linewidth=1.2)
ax.set(title="驱动收支：突增主要来自磁滞记忆项",ylabel="沿速度方向的广义力 (mN)",xlim=(280,480),ylim=(-.1,.8))
ax.legend(fontsize=9,frameon=False)
ax=axs[1,1]
ax.plot(t,speed,color=blue,linewidth=.8)
ax.set(title="延长至 t2D = 2000：后面仍有较慢的演化",ylabel="质心速率 (mm/s)",xlim=(0,2000))
ax.text(.98,.94,"峰值约 279 mm/s\n末段均速约 123 mm/s\n尚不能认定已达最终稳态",ha="right",va="top",transform=ax.transAxes,fontsize=10)
fig.savefig(OUT/"spin_transition.png",dpi=165,facecolor=fig.get_facecolor())
print(OUT/"spin_transition.png")
