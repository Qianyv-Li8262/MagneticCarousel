from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import RUNS_DIR, SCRATCH_DIR
from scan_sr_somega_gpu import _cell_edges
from summarize_phase_results import ALPHA_REF, BOUNDARY_SCALE


ROOT = RUNS_DIR / "phase-behavior" / "legacy-analysis"
CASES = (
    (ROOT / "resonance_projected_128_T16000.npz", 1.0, r"$H/H_0=1$"),
    (ROOT / "field_0p1_T8000.npz", 0.1, r"$H/H_0=0.1$"),
    (ROOT / "field_0p01_T8000.npz", 0.01, r"$H/H_0=0.01$"),
)


fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), sharex=True, sharey=True, constrained_layout=True)
last_image = None
for axis, (path, field_scale, title) in zip(axes, CASES):
    data = np.load(path)
    names = list(map(str, data["metric_names"]))
    phase = data["metrics"][..., names.index("mean_phase_rate")]
    q = data["metrics"][..., names.index("mean_abs_q")]
    sr, sw = data["x"], data["y"]
    last_image = axis.pcolormesh(
        _cell_edges(sr), _cell_edges(sw), np.clip(phase, 0.0, 1.0),
        shading="flat", cmap="magma", vmin=0.0, vmax=1.0,
    )
    q_typical = float(np.nanmedian(q))
    if field_scale == 1.0:
        left = np.linspace(sr.min(), 1.0 / ALPHA_REF - 1.0e-4, 1000)
        right = np.linspace(1.0 / ALPHA_REF + 1.0e-4, sr.max(), 1000)
        boundary = lambda x: BOUNDARY_SCALE * np.abs(1.0 - ALPHA_REF * x) ** 1.5 / x
        axis.plot(left, boundary(left), "w--", lw=1.8)
        axis.plot(right, boundary(right), "w--", lw=1.8, label="saturated torque limit")
    locked = int(np.count_nonzero(np.abs(phase) < 0.05))
    axis.set_title(f"{title}\nlocked cells: {locked}/{phase.size}; median |q|={q_typical:.4g}")
    axis.set(xlabel=r"$s_r$", xlim=(2.0, 4.5), ylim=(0.05, 5.0))
    axis.grid(alpha=0.18)
    if field_scale == 1.0:
        axis.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel(r"$s_\omega$")
fig.colorbar(last_image, ax=axes, label=r"$\langle\dot\phi\rangle$ (clipped)")
fig.suptitle("Lower external field removes the locked region")
output = SCRATCH_DIR / "phase-behavior" / "field_strength_comparison.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220)
print(output.resolve())
