"""Compare CPU FP32/FP64 phase-rate evolution for one bright phase-map cell."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "trajectories"
RESULT = ROOT / "scratch" / "trajectories" / "bright_scatter_phidot_cpu_comparison.png"
STEM = "bright-scatter-sr3.9761906-sw1.7111111-"


def load(kind: str) -> np.ndarray:
    return np.genfromtxt(
        RUN_ROOT / f"{STEM}{kind}" / "trajectory_overview.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def main() -> None:
    fp64 = load("fp64")
    fp32 = load("fp32")
    result = RESULT.resolve()
    result.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.2), constrained_layout=True)
    colors = {"FP64": "#1976d2", "FP32": "#ef6c00"}
    for label, data in (("FP64", fp64), ("FP32", fp32)):
        axes[0].plot(data["time"], data["phase_rate"], lw=1.05, color=colors[label], label=label)
        late = data["time"] >= 46000.0
        mean = float(np.mean(data["phase_rate"][late]))
        axes[2].plot(data["time"][late], data["phase_rate"][late], lw=1.1, color=colors[label], label=f"{label}: mean={mean:.6f}")

    difference = np.abs(fp32["phase_rate"].astype(float) - fp64["phase_rate"].astype(float))
    axes[1].semilogy(fp64["time"], np.maximum(difference, 1.0e-10), color="#6a1b9a", lw=1.0)
    axes[1].axhline(0.05, color="0.35", ls="--", lw=1.0, label="locking threshold 0.05")

    axes[0].set(ylabel=r"$\dot\phi=1-v-\dot\theta$", title="Full evolution")
    axes[0].legend()
    axes[1].set(ylabel=r"$|\dot\phi_{32}-\dot\phi_{64}|$", title="Precision-induced trajectory separation")
    axes[1].legend()
    axes[2].set(xlabel=r"time $t$", ylabel=r"$\dot\phi=1-v-\dot\theta$", title="Final observation window")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(r"Bright cell: $s_r=3.9761906$, $s_\omega=1.7111111$; CPU, $\Delta t=10^{-3}$")
    figure.savefig(result, dpi=200)
    plt.close(figure)
    print(result)


if __name__ == "__main__":
    main()
