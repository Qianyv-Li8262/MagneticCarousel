"""Decompose actual enabled trajectories into JA, eddy and rolling contributions."""
from pathlib import Path
import json
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scratch/eddy_integration_20261016"
manifest = json.loads((ROOT / "build/release/extensions/fp64/build_config.json").read_text())
assert manifest["eddy_currents"]["enabled"]
labels = ["after_eddy_on", "after_eddy_on_no_blow", "after_eddy_on_halfstep", "after_eddy_on_no_blow_halfstep"]
tables = {}
report = {}
for label in labels:
    folder = DATA / label
    trace = np.load(folder / "trace.npz")
    rows = np.column_stack((trace["time_2d"], trace["state"], trace["blow_mN"]))
    rows.astype("<f8").tofile(folder / "drive_input.bin")
    subprocess.run([str(DATA / "diagnose_drive.exe"), str(folder / "drive_input.bin"),
                    manifest["lut"], str(folder / "drive.csv")], check=True)
    table = np.genfromtxt(folder / "drive.csv", names=True, delimiter=",")
    tables[label] = table
    windows = [(100, 200), (140, 190), (600, 650)]
    stats = []
    for lower, upper in windows:
        selected = table[(table["t2d"] >= lower-1e-8) & (table["t2d"] < upper)]
        if not len(selected):
            continue
        mean = {key: float(np.mean(selected[key])) for key in table.dtype.names if key not in ("t2d", "time_s")}
        mean["interval_t2d"] = [lower, upper]
        mean["net_drive_mN"] = mean["JA_drive_mN"] + mean["eddy_drive_mN"] - mean["roll_loss_mN"] - mean["air_loss_mN"]
        stats.append(mean)
    report[label] = stats

before = np.load(DATA / "after_eddy_off/trace.npz")
on = np.load(DATA / "after_eddy_on/trace.npz")
half = np.load(DATA / "after_eddy_on_halfstep/trace.npz")
n = min(len(on["time_2d"]),len(half["time_2d"]))
report["halfstep_max_radius_difference_mm"] = float(np.max(abs(
    np.linalg.norm(on["state"][:n,1:3],axis=1)-np.linalg.norm(half["state"][:n,1:3],axis=1)))*17.5)
report["halfstep_max_speed_difference_mm_s"] = float(np.max(abs(
    np.linalg.norm(on["state"][:n,3:5],axis=1)-np.linalg.norm(half["state"][:n,3:5],axis=1)))*17.5*(2*np.pi*200/60))
report["state_of_analysis"] = "Production enabled; force decomposition on actual coupled states. No parameters changed."
(DATA / "enabled_drive_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
fig, axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("开启涡流后：驱动增加，但仍不足以维持高速运动",fontsize=19,fontweight="bold")
fig.supxlabel("同一初态、原模型参数；吹气 0.7 mN，t2D = 75–275。开启曲线在越出磁场表时终止，未到停吹时刻。",fontsize=10)
orange, blue, green = "#ea580c", "#2563eb", "#059669"
units = 17.5*(2*np.pi*200/60)
for ax in axes.flat:
    ax.grid(color="#e2e8f0",linewidth=.7)
ax = axes[0,0]
ax.plot(before["time_2d"],np.linalg.norm(before["state"][:,3:5],axis=1)*units,color=blue,label="涡流关闭")
ax.plot(on["time_2d"],np.linalg.norm(on["state"][:,3:5],axis=1)*units,color=orange,label="涡流开启")
ax.axvspan(75,275,color="#fed7aa",alpha=.4)
ax.set(title="相同吹气：开启后更快，但向外逃逸",xlabel="无量纲时间 t2D",ylabel="质心速率 (mm/s)",xlim=(0,650))
ax.legend(frameon=False)
ax.text(.98,.28,"开启后的峰值约 283 mm/s\n不能从这条轨迹判断停吹后的行为",transform=ax.transAxes,ha="right",fontsize=10)
ax = axes[0,1]
ax.plot(before["time_2d"],np.linalg.norm(before["state"][:,1:3],axis=1)*17.5,color=blue)
ax.plot(on["time_2d"],np.linalg.norm(on["state"][:,1:3],axis=1)*17.5,color=orange)
ax.axhline(105,color="#b91c1c",linestyle="--",linewidth=1)
ax.axvspan(75,275,color="#fed7aa",alpha=.4)
ax.set(title="半径：约 9.80 s 时触及 105 mm 表边界",xlabel="无量纲时间 t2D",ylabel="半径 (mm)",xlim=(0,650),ylim=(65,110))
ax.annotate("t2D ≈ 205.2：停止计算",xy=(205.2,104.82),xytext=(290,98),
    arrowprops={"arrowstyle":"->","color":"#b91c1c"},fontsize=10)
ax = axes[1,0]
nb=tables["after_eddy_on_no_blow"]
ax.plot(nb["t2d"],nb["speed_mm_s"],color=green,label="开启涡流，无吹气")
ax.set(title="无吹气：末段均速仍只有 5.16 mm/s",xlabel="无量纲时间 t2D",ylabel="质心速率 (mm/s)",xlim=(0,650))
ax.legend(frameon=False)
ax.text(.98,.7,"涡流参与了运动\n但没有形成高速自持轨道",transform=ax.transAxes,ha="right",fontsize=11)
ax = axes[1,1]
fast = report["after_eddy_on"][1]
values = [fast["JA_drive_mN"],fast["eddy_drive_mN"],-fast["roll_loss_mN"],fast["net_drive_mN"]]
bars=ax.bar(["原磁模型","涡流增量","滚阻","净值"],values,color=[blue,green,"#64748b",orange],width=.6)
ax.axhline(0,color="#334155",linewidth=1)
for bar,value in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,value+(.012 if value>=0 else -.015),f"{value:+.3f}",ha="center",va="bottom" if value>=0 else "top",fontsize=11)
ax.set(title="高速段力收支：去掉吹气后还缺 0.319 mN",ylabel="沿实际速度方向的等效驱动力 (mN)",ylim=(-.62,.32))
ax.text(.98,.97,"t2D = 140–190；均速 238 mm/s\n已计入力和滚动力矩；净值含空气阻力",transform=ax.transAxes,ha="right",va="top",fontsize=9)
fig.savefig(DATA/"eddy_enabled_drive.png",dpi=165,facecolor=fig.get_facecolor())
print(DATA/"eddy_enabled_drive.png")
