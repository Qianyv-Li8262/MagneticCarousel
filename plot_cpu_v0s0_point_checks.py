# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Plot CPU center-of-mass and phase velocities for two selected initial states."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CASES = (
    ("red-grid initial state", "v0s0-red-current", 4.285714, 2.698413, "#b40426"),
    ("flesh-grid initial state", "v0s0-flesh-current", -4.920635, 4.285714, "#e9a88e"),
)
OUTPUT = ROOT / "scratch" / "trajectories" / "cpu_selected_points_v_and_phase_rate.png"


def load(run_name: str):
    path = ROOT / "runs" / "trajectories" / run_name / "trajectory_overview.csv"
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    for column, (label, run_name, v0, s0, color) in enumerate(CASES):
        data = load(run_name)
        late = data["time"] >= 46000.0
        mean_v = float(np.mean(data["xdot"][late]))
        mean_phase = float(np.mean(data["phase_rate"][late]))
        axes[0, column].plot(data["time"], data["xdot"], color=color, lw=0.8)
        axes[1, column].plot(data["time"], data["phase_rate"], color=color, lw=0.8)
        axes[0, column].axhline(mean_v, color="0.2", ls="--", lw=1.0)
        axes[1, column].axhline(mean_phase, color="0.2", ls="--", lw=1.0)
        axes[0, column].set_title(
            rf"{label}: $v_0={v0:g},\ S_0={s0:g}$" "\n" rf"final-window $\langle v\rangle={mean_v:.6f}$"
        )
        axes[1, column].set_title(rf"final-window $\langle\dot\phi\rangle={mean_phase:.6f}$")
        axes[0, column].set_ylabel(r"center-of-mass velocity $v$")
        axes[1, column].set_ylabel(r"phase velocity $\dot\phi=1-v-\dot\theta$")
        axes[1, column].set_xlabel(r"time $t$")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    figure.suptitle(r"CPU point checks at $s_r=2.5$, $s_\omega=2.25447$, $\Delta t=10^{-3}$")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=190)
    plt.close(figure)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
