from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from scan_sr_somega_gpu import _cell_edges
from project_paths import RESULTS_DIR, RUNS_DIR


source = RUNS_DIR / "initial-conditions" / "t50000" / "sr2.5_sw2.254472.npz"
output = RESULTS_DIR / "initial-conditions" / "threshold_sensitivity.png"
output.parent.mkdir(parents=True, exist_ok=True)
data = np.load(source)
v0 = data["v0"]
s0 = data["s0"]
phase_rate = np.abs(data["metrics"][..., 3])

thresholds = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.065)
cmap = ListedColormap(("white", "black"))
norm = BoundaryNorm((-0.5, 0.5, 1.5), cmap.N)
figure, axes = plt.subplots(2, 4, figsize=(15.6, 8.0), sharex=True, sharey=True, constrained_layout=True)

for axis, threshold in zip(axes.ravel(), thresholds):
    locked = phase_rate < threshold
    axis.pcolormesh(
        _cell_edges(v0),
        _cell_edges(s0),
        np.where(locked, 0, 1),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    if np.any(locked) and np.any(~locked):
        axis.contour(v0, s0, locked.astype(float), levels=(0.5,), colors=("#e66101",), linewidths=1.5)
    axis.set_title(f"threshold = {threshold:g}; locked = {np.mean(locked):.1%}")
    axis.grid(alpha=0.15)
    print(threshold, int(np.count_nonzero(locked)), locked.size, float(np.mean(locked)))

for axis in axes[-1, :]:
    axis.set_xlabel(r"initial $v_0$")
for axis in axes[:, 0]:
    axis.set_ylabel(r"initial slip $S_0$")
figure.suptitle(
    r"Ordinary FP32 threshold sensitivity at $s_r=2.5$, $s_\omega=2.25447$, $T=50000$"
    "\n"
    r"white: $|\langle\dot\phi\rangle|<$ threshold; black: above threshold"
)
figure.savefig(output, dpi=200)
print(output)
