# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Plot the 1D reference trajectory over the physical time of the 2D view."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from animate_one_trajectory import load_hysteresis
from project_paths import SCRATCH_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-end", type=float, default=835.0)
    args = parser.parse_args()
    sr, somega = 0.428571, 1.0
    t2_end, ratio = args.t2_end, 4.0
    dt, chunk = 0.001, 1000
    module = load_hysteresis(False)
    alpha = 0.33333333 * sr
    params = module.SolverParams(
        alpha, 1.7212357e-4 / sr,
        0.16560356 / somega**2, 489.39639 / somega**2,
        11011.419 / (sr * sr * somega**2),
    )
    state = np.zeros(8, dtype=module.real_dtype)
    sliding = slip_sign = None
    records = [state.copy()]
    for _ in range(round(t2_end * ratio / (dt * chunk))):
        state, sliding, slip_sign = module.evolve_with_discrete_mode(
            state, dt, chunk, params, sliding, slip_sign
        )
        records.append(state.copy())
    a = np.asarray(records, dtype=float)
    t1 = np.arange(len(a)) * dt * chunk
    t2 = t1 / ratio
    phase_rate = 1 - a[:, 2] - a[:, 4]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, values, ylabel, color in zip(
        axes,
        (a[:, 1], a[:, 2], a[:, 3], phase_rate),
        ("x (1D units)", "v = dx/dt₁", "θ (rad)", "phase rate"),
        ("tab:blue", "tab:orange", "tab:purple", "tab:green"),
    ):
        ax.plot(t2, values, color=color, lw=1.35)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Equivalent 2D time t₂ = t₁ / 4")
    axes[-1].set_xlim(0, t2_end)
    top = axes[0].secondary_xaxis("top", functions=(lambda x: 4*x, lambda x: x/4))
    top.set_xlabel("1D simulation time t₁")
    fig.suptitle(f"Equivalent 1D trajectory · sr=0.428571, sω=1 · t₁={t2_end*ratio:g} ↔ t₂={t2_end:g}")
    out = SCRATCH_DIR / f"equivalent_1d_sr0.428571_somega1_t2_{t2_end:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"output={out}\nfinal_x={a[-1,1]:.9g} final_v={a[-1,2]:.9g} final_phase_rate={phase_rate[-1]:.9g}")


if __name__ == "__main__":
    main()
