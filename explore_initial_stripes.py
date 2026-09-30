"""Animate representative points from the initial-state FP64 stripe scan.

Run from the repository root: python src2d/explore_initial_stripes.py
The script saves a GIF, a history plot, and trajectory samples beside the scan.
"""

from pathlib import Path
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from project_paths import extension_dir, RUNS_DIR, SCRATCH_DIR
from scan_sr_somega_2d_gpu import REFERENCE_ALPHA, REFERENCE_DELTA, REFERENCE_GAMMA_G, REFERENCE_GAMMA, REFERENCE_GAMMA_M


POINT_ROWS = (38, 39, 43, 44, 48, 51, 56, 61)
POINT_COL = 36
SAMPLE_STEPS = 10000
TOTAL_STEPS = 1_500_000


def main():
    source = np.load(RUNS_DIR / "phase-behavior" / "initial_2d_cpu_fp64.npz")
    dt = float(source["dt"])
    sr, somega = float(source["sr"]), float(source["somega"])
    count = int(source["steps"])
    if count != TOTAL_STEPS or dt != 0.001:
        raise ValueError("Scan time settings changed; review sample spacing")
    manifest = json.loads((extension_dir("fp64") / "build_config.json").read_text())
    binary = extension_dir("fp64") / manifest["binary"]
    spec = importlib.util.spec_from_file_location("hysteresis", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m = module.magnet_number / 2
    scale = (module.magnet_number / 8) ** 2 / somega**2
    alpha = REFERENCE_ALPHA * sr
    params = module.SolverParams(alpha, REFERENCE_DELTA / sr,
        REFERENCE_GAMMA_G * scale, REFERENCE_GAMMA * scale,
        REFERENCE_GAMMA_M * scale / sr**2)
    histories = []
    rows = []
    for row in POINT_ROWS:
        v0, s0 = float(source["v0"][POINT_COL]), float(source["s0"][row])
        state = np.zeros(15, dtype=np.float64)
        state[1], state[4], state[5], state[9] = m, m*v0, 1, m*(s0-v0)/alpha
        states = [state.copy()]
        sliding = direction = None
        for step in range(0, count, SAMPLE_STEPS):
            state, sliding, direction = module.evolve_trajectory(
                state, dt, min(SAMPLE_STEPS, count-step), params, sliding, direction, step)
            states.append(state.copy())
        states = np.asarray(states)
        radius = np.hypot(states[:,1], states[:,2])
        v = (states[:,1]*states[:,4]-states[:,2]*states[:,3]) / radius / m
        psi_rate = 1 - m*v/radius - (states[:,1]*states[:,9]+states[:,2]*states[:,10])/radius/m
        histories.append((states, v, psi_rate))
        metric = source["metrics"][row, POINT_COL]
        rows.append((row, v0, s0, metric[0], metric[3], metric[12], metric[13],
                     np.mean(v[-50:]), np.mean(psi_rate[-50:]), np.min(radius), np.max(radius)))
        print(f"row={row} s0={s0:.3f} mean_v={metric[0]:.5f} drift_v={metric[12]:.5f}", flush=True)

    out = SCRATCH_DIR / "phase-behavior" / "stripe-exploration"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "histories.npz", rows=POINT_ROWS, v0=source["v0"][POINT_COL],
        s0=source["s0"][list(POINT_ROWS)], states=np.asarray([h[0] for h in histories]),
        v=np.asarray([h[1] for h in histories]), psi_rate=np.asarray([h[2] for h in histories]),
        time=np.arange(count//SAMPLE_STEPS+1)*SAMPLE_STEPS*dt)
    with (out / "summary.csv").open("w", encoding="utf-8") as file:
        file.write("row,v0,s0,scan_mean_v,scan_mean_phase,scan_delta_v,scan_delta_phase,sampled_mean_v,sampled_mean_phase,min_r,max_r\n")
        for row in rows:
            file.write(",".join(str(x) for x in row)+"\n")

    time = np.arange(count//SAMPLE_STEPS+1)*SAMPLE_STEPS*dt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for row, (_, v, phase) in zip(POINT_ROWS, histories):
        label = f"S0={float(source['s0'][row]):.2f}"
        axes[0].plot(time, v, lw=1, label=label)
        axes[1].plot(time, phase, lw=1, label=label)
    axes[0].set_ylabel("tangential speed / m")
    axes[1].set_ylabel("phase rate")
    axes[1].set_xlabel("time")
    axes[0].legend(ncol=4, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
        ax.axvline(1000, color="black", ls="--", lw=.8)
    fig.savefig(out / "histories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    lines, dots, texts = [], [], []
    for ax, row in zip(axes.flat, POINT_ROWS):
        ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect="equal", title=f"S0={float(source['s0'][row]):.2f}")
        ax.add_patch(plt.Circle((0,0), m, fill=False, color="0.7"))
        line, = ax.plot([], [], lw=.8)
        dot, = ax.plot([], [], "o", ms=4)
        label = ax.text(.03,.97,"",transform=ax.transAxes,va="top",fontsize=8)
        lines.append(line); dots.append(dot); texts.append(label)
    def update(frame):
        for i, (states, v, _) in enumerate(histories):
            lines[i].set_data(states[:frame+1,1],states[:frame+1,2])
            dots[i].set_data([states[frame,1]],[states[frame,2]])
            texts[i].set_text(f"t={time[frame]:.0f}  v={v[frame]:.3f}")
        return [*lines,*dots,*texts]
    animation = FuncAnimation(fig, update, frames=len(time), interval=90, blit=True)
    animation.save(out / "trajectories.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
