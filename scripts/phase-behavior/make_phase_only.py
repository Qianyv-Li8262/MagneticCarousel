from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import RUNS_DIR, SCRATCH_DIR
from scan_sr_somega_gpu import _cell_edges
from summarize_phase_results import ALPHA_REF, binned_collapse, locking_limit


ROOT = RUNS_DIR / "phase-behavior" / "legacy-analysis"
OUT = SCRATCH_DIR / "phase-behavior" / "phase_diagram.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(path):
    data = np.load(path)
    names = list(map(str, data["metric_names"]))
    return data, data["metrics"][..., names.index("mean_phase_rate")]


def observed_edges(sr, phase_rate, threshold=0.05):
    left = np.full(phase_rate.shape[0], np.nan)
    right = np.full(phase_rate.shape[0], np.nan)
    for j, row in enumerate(phase_rate):
        selected = np.flatnonzero(row > threshold)
        if selected.size >= 2:
            left[j], right[j] = sr[selected[0]], sr[selected[-1]]
    return left, right


long, phase = load(ROOT / "resonance_projected_128_T16000.npz")
short, phase_short = load(ROOT / "resonance_projected_192_T4000.npz")
sr, sw = long["x"], long["y"]

fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.35), constrained_layout=True)

image = axes[0].pcolormesh(
    _cell_edges(sr), _cell_edges(sw), np.clip(phase, 0.0, 1.0),
    shading="flat", cmap="magma", vmin=0.0, vmax=1.0,
)
left_sr = np.linspace(sr.min(), 1.0 / ALPHA_REF - 1.0e-4, 1000)
right_sr = np.linspace(1.0 / ALPHA_REF + 1.0e-4, sr.max(), 1000)
axes[0].plot(left_sr, locking_limit(left_sr), "w--", lw=2.0, label="analytic torque limit")
axes[0].plot(right_sr, locking_limit(right_sr), "w--", lw=2.0)
axes[0].set(xlabel=r"$s_r$", ylabel=r"$s_\omega$", title=r"CUDA order parameter $\langle\dot\phi\rangle$")
axes[0].set_ylim(sw.min(), sw.max())
axes[0].legend(loc="upper left")
fig.colorbar(image, ax=axes[0], label=r"$\langle\dot\phi\rangle$ (clipped)")

ratio = sw[:, None] / locking_limit(sr[None, :])
centers, quantiles = binned_collapse(ratio.ravel(), phase.ravel())
valid = np.all(np.isfinite(quantiles), axis=1)
axes[1].fill_between(centers[valid], quantiles[valid, 0], quantiles[valid, 2], alpha=0.25, label="10--90%")
axes[1].plot(centers[valid], quantiles[valid, 1], "o-", ms=4, label="median")
axes[1].axvline(1.0, color="0.2", ls="--", label="torque threshold")
axes[1].set(
    xscale="log", xlabel=r"$\Lambda=s_\omega/s_\omega^{\max}(s_r)$",
    ylabel=r"$\langle\dot\phi\rangle$", title="Collapse onto one control ratio", ylim=(-0.03, 1.03),
)
axes[1].legend()

long_left, long_right = observed_edges(sr, phase)
short_left, short_right = observed_edges(short["x"], phase_short)
theory_left = np.interp(sw, locking_limit(left_sr)[::-1], left_sr[::-1])
theory_right = np.interp(sw, locking_limit(right_sr), right_sr)
axes[2].plot(theory_left, sw, "k--", lw=2, label="analytic")
axes[2].plot(theory_right, sw, "k--", lw=2)
axes[2].plot(short_left, short["y"], ".", ms=3, alpha=0.45, label="CUDA T=4000")
axes[2].plot(short_right, short["y"], ".", ms=3, alpha=0.45)
axes[2].plot(long_left, sw, "o", ms=2.5, label="CUDA T=16000")
axes[2].plot(long_right, sw, "o", ms=2.5)
axes[2].set(xlabel=r"boundary $s_r$", ylabel=r"$s_\omega$", title="Finite-time boundary converges to theory")
axes[2].set(xlim=(sr.min(), sr.max()), ylim=(sw.min(), sw.max()))
axes[2].legend(loc="upper left")

for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle("Self-consistent hysteresis model: locked phase and running wedge", fontsize=16)
fig.savefig(OUT, dpi=220)
print(OUT.resolve())
