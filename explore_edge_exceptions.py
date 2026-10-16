"""Track four exceptional pixels on the v0=0.3 edge of the T=7000 scan."""
import importlib.util
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project_paths import RUNS_DIR, SCRATCH_DIR, extension_dir

scan=np.load(RUNS_DIR/"phase-behavior"/"initial_2d_cpu_fp64.npz")
if int(scan["steps"]) != 7_000_000: raise ValueError("Expected the user's T=7000 scan")
directory=extension_dir("fp64")
manifest=json.loads((directory/"build_config.json").read_text())
spec=importlib.util.spec_from_file_location("hysteresis",directory/manifest["binary"])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sr,somega=float(scan["sr"]),float(scan["somega"])
params=module.SolverParams(sr=sr, somega=somega)
m=module.magnet_number/2
rows=(1,2,60,63)
labels=[f"S0={float(scan['s0'][i]):.2f}" for i in rows]
colors=("tab:blue","tab:orange","tab:green","tab:red")
out=SCRATCH_DIR/"phase-behavior"/"edge-exceptions"
out.mkdir(parents=True,exist_ok=True)

# Parameter map: use the actual numerical array, not the screenshot colors.
fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
for ax, index, title in zip(axes,(0,3),("mean tangential speed / m","mean phase rate")):
    im=ax.pcolormesh(scan["v0"],scan["s0"],scan["metrics"][:,:,index],
        shading="nearest",cmap="viridis")
    fig.colorbar(im,ax=ax)
    for i,label,color in zip(rows,labels,colors):
        ax.scatter([scan["v0"][-1]],[scan["s0"][i]],s=95,facecolors="none",
            edgecolors=color,linewidths=2,label=label,zorder=5)
    ax.set(xlabel="initial v0",ylabel="initial S0",title=title,xlim=(-.4,.34))
    ax.legend(fontsize=8,loc="center left")
fig.savefig(out/"parameter_map.png",dpi=180)
plt.close(fig)

dt=float(scan["dt"])
sample_steps=5000
total_steps=48_000_000
times=np.arange(total_steps//sample_steps+1)*sample_steps*dt
history=[]
start=time.perf_counter()
for i,label in zip(rows,labels):
    v0=float(scan["v0"][-1]); s0=float(scan["s0"][i])
    state=np.zeros(15,dtype=np.float64)
    state[1],state[4],state[5],state[9]=m,m*v0,1,m*(s0-v0)/params.alpha
    states=[state.copy()]
    sliding=direction=None
    for step in range(0,total_steps,sample_steps):
        state,sliding,direction=module.evolve_trajectory(
            state,dt,sample_steps,params,sliding,direction,step)
        states.append(state.copy())
    states=np.asarray(states)
    r=np.hypot(states[:,1],states[:,2])
    v=(states[:,1]*states[:,4]-states[:,2]*states[:,3])/r/m
    omega=(states[:,1]*states[:,9]+states[:,2]*states[:,10])/r/m
    phase=1-m*v/r-omega
    history.append((v,r,phase,states))
    print(label,"finished",round(time.perf_counter()-start,1),"s",flush=True)

np.savez_compressed(out/"trajectories.npz",time=times,rows=rows,
    s0=scan["s0"][list(rows)],v=np.asarray([h[0] for h in history]),
    radius=np.asarray([h[1] for h in history]),phase_rate=np.asarray([h[2] for h in history]),
    states=np.asarray([h[3] for h in history]))

fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout="constrained")
for (v,r,phase,_),label,color in zip(history,labels,colors):
    axes[0].plot(times,v,lw=.9,label=label,color=color)
    axes[1].plot(times,r,lw=.9,label=label,color=color)
    axes[2].plot(times,phase,lw=.9,label=label,color=color)
axes[0].set_ylabel("tangential speed / m")
axes[1].set_ylabel("radius")
axes[2].set_ylabel("phase rate")
axes[2].set_xlabel("time")
axes[0].legend(ncol=4,fontsize=9)
for ax in axes:
    ax.grid(alpha=.25)
    ax.axvline(7000,color="black",lw=.8,ls="--")
fig.savefig(out/"trajectories.png",dpi=180)
plt.close(fig)

# Display late behavior separately to retain useful vertical resolution.
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
late=times>=6000
for (v,r,phase,_),label,color in zip(history,labels,colors):
    axes[0].plot(times[late],v[late],lw=.9,label=label,color=color)
    axes[1].plot(times[late],phase[late],lw=.9,label=label,color=color)
axes[0].set_ylabel("tangential speed / m")
axes[1].set_ylabel("phase rate")
axes[1].set_xlabel("time")
axes[0].legend(ncol=4,fontsize=9)
for ax in axes: ax.grid(alpha=.25)
fig.savefig(out/"trajectories_late.png",dpi=180)
plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout="constrained")
kernel=np.ones(101)/101  # 500 time units at five units per sample.
smoothed_time=times[50:-50]
for (v,r,_,_),label,color in zip(history,labels,colors):
    axes[0].plot(smoothed_time,np.convolve(v,kernel,"valid"),label=label,color=color)
    axes[1].plot(smoothed_time,np.convolve(r,kernel,"valid"),label=label,color=color)
axes[0].set_ylabel("500-time moving mean speed / m")
axes[1].set_ylabel("500-time moving mean radius")
axes[1].set_xlabel("time")
axes[0].legend(ncol=4,fontsize=9)
for ax in axes: ax.grid(alpha=.25)
fig.savefig(out/"trajectories_smoothed.png",dpi=180)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(9,9),layout="constrained")
late_orbit=times>=46000
for ax,(_,_,_,states),label,color in zip(axes.flat,history,labels,colors):
    ax.plot(states[late_orbit,1],states[late_orbit,2],lw=.5,color=color)
    ax.scatter([states[-1,1]],[states[-1,2]],s=25,color=color)
    ax.set(xlim=(-4.5,4.5),ylim=(-4.5,4.5),aspect="equal",
           title=f"{label}, t=46000-48000")
    ax.grid(alpha=.2)
fig.savefig(out/"xy_trajectories_late.png",dpi=180)
plt.close(fig)

# Sampled 500-unit window summaries for the full continuous history.
with (out/"window_summary.csv").open("w",encoding="utf-8") as file:
    file.write("row,s0,window_end,mean_v,mean_phase,mean_radius,delta_v,finite,quaternion_norm\n")
    for i,(v,r,phase,states) in zip(rows,history):
        for end in (7000,12000,24000,36000,48000):
            selection=(times>end-500)&(times<=end)
            halfway=end-250
            first=selection&(times<=halfway)
            second=selection&(times>halfway)
            record=(i,float(scan["s0"][i]),end,np.mean(v[selection]),
                np.mean(phase[selection]),np.mean(r[selection]),
                np.mean(v[second])-np.mean(v[first]),
                bool(np.isfinite(states).all()),np.linalg.norm(states[end//5,5:9]))
            file.write(",".join(map(str,record))+"\n")
            print(record,flush=True)
