"""Plot the production animation replay against the preserved pre-integration run."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scratch/eddy_integration_20261016"
before = np.load(DATA / "before_integration/trace.npz")
after = np.load(DATA / "after_eddy_off/trace.npz")
summary = json.loads((DATA / "after_eddy_off/summary.json").read_text())
old = json.loads((ROOT / "audits/pure_rolling_20261015/current_blow/summary.json").read_text())
assert np.array_equal(before["initial"], after["initial"])
assert np.array_equal(before["time_2d"], after["time_2d"])
assert np.array_equal(before["state"], after["state"])
assert np.array_equal(before["sliding"], after["sliding"])
observables = np.genfromtxt(DATA / "after_eddy_off/observables.csv", delimiter=",", names=True)
time = after["time_2d"]
speed = observables["speed_mm_s"]
radius = observables["radius_mm"]
error = np.max(np.abs(after["state"] - before["state"]), axis=1)
length_mm = radius[0] / np.linalg.norm(before["state"][0,1:3])
omega = time[-1] / observables["time_s"][-1]
record = {
    "initial_identical": True, "all_states_bitwise_identical": True,
    "contact_modes_identical": True, "sample_count": len(time),
    "max_state_absolute_difference": float(error.max()),
    "historical_peak_speed_difference_mm_s": summary["max_speed_mm_s"] - 1000*old["max_speed_m_s"],
    "historical_tail_speed_difference_mm_s": summary["tail_mean_speed_mm_s"] - 1000*old["speed_tail_mean_m_s"],
}
(DATA / "regression.json").write_text(json.dumps(record, indent=2))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#334155", "text.color": "#1e293b", "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), layout="constrained")
fig.set_facecolor("#f8fafc")
fig.suptitle("涡流已接入，编译开关关闭：复现原模型的吹气响应", fontsize=19, fontweight="bold")
fig.supxlabel("同一初态：v0 = −0.05，S0 = 0，磁盘相位 0.58，预磁化 2 圈；sr = 1，sω = 2，Δt = 0.0005", fontsize=10)
for ax in axes.flat:
    ax.set_facecolor("white")
    ax.grid(color="#e2e8f0", linewidth=.7)

blue, orange = "#2563eb", "#ea580c"
ax = axes[0, 0]
old_speed = np.linalg.norm(before["state"][:, 3:5], axis=1)
old_speed *= length_mm * omega
ax.plot(time, old_speed, color="#94a3b8", linewidth=3.3, label="接入前基线")
ax.plot(time, speed, color=blue, linewidth=1.3, label="接入后 · 涡流关闭")
ax.axvspan(75, 275, color="#fed7aa", alpha=.6, label="吹气 0.7 mN")
ax.set(title="速度：停吹后回到慢速运动", xlabel="无量纲时间 t2D", ylabel="质心速率 (mm/s)", xlim=(0,650))
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.annotate("停吹 248.4 mm/s", xy=(275, speed[np.argmin(abs(time-275))]),
    xytext=(330,185), arrowprops={"arrowstyle":"->", "color":"#475569"}, fontsize=10)
ax.text(.98,.15,"末段均速 3.09 mm/s\n(t2D = 600–650)", ha="right", transform=ax.transAxes, fontsize=10)

ax = axes[0, 1]
ax.plot(time, np.linalg.norm(before["state"][:,1:3],axis=1)*length_mm, color="#94a3b8", linewidth=3.3)
ax.plot(time, radius, color=blue, linewidth=1.3)
ax.axvspan(75,275,color="#fed7aa",alpha=.6)
ax.set(title="轨道半径：末段均值 68.40 mm", xlabel="无量纲时间 t2D", ylabel="半径 (mm)",xlim=(0,650))
ax.axhline(70,color="#64748b",linestyle=":",linewidth=1)

ax = axes[1, 0]
ax.step(time, after["blow_mN"],where="post",color=orange,linewidth=2)
ax.fill_between(time,after["blow_mN"],step="post",color="#fed7aa",alpha=.6)
ax.set(title="吹气时序：75 开始，持续 200",xlabel="无量纲时间 t2D",ylabel="吹气力 (mN)",xlim=(0,650),ylim=(-.03,.85))
ax.set_xticks([0,75,275,450,650])
ax.text(.98,.86,"实际时间 3.58–13.13 s\n仿真至 31.04 s",ha="right",va="top",transform=ax.transAxes,fontsize=10)

ax = axes[1, 1]
ax.plot(time,error,color="#059669",linewidth=2)
ax.set(title="与接入前逐点比较：差值严格为 0",xlabel="无量纲时间 t2D",ylabel=r"$\max_i |\mathrm{state}_i - \mathrm{baseline}_i|$",xlim=(0,650),ylim=(-.05,.15))
ax.set_yticks([0,.05,.1])
ax.text(.5,.70,f"{len(time):,} 个采样点 × 15 个状态量\n全部逐位一致；全程纯滚动",ha="center",transform=ax.transAxes,fontsize=11,
        bbox={"facecolor":"#ecfdf5","edgecolor":"#a7f3d0","boxstyle":"round,pad=.7"})
fig.savefig(DATA / "eddy_off_regression.png",dpi=165,facecolor=fig.get_facecolor())
print(json.dumps(record,indent=2))
print(DATA / "eddy_off_regression.png")
