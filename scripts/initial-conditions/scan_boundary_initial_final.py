from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from scan_sr_somega_gpu import locking_limit
from scan_v0_s0_gpu import _cell_edges, phase_locked_mask, run_sweep
from project_paths import RESULTS_DIR, RUNS_DIR, SCRATCH_DIR


OUT = RUNS_DIR / "initial-conditions" / "boundary-final"
RESULTS = RESULTS_DIR / "initial-conditions" / "boundary-final"
FIGURES = SCRATCH_DIR / "initial-conditions" / "boundary-final"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

POINT_SPECS = (
    (2.5, 0.95),
    (2.5, 1.05),
    (2.7, 1.05),
    (3.3, 1.05),
    (3.5, 0.95),
    (3.5, 1.05),
)
K = 80
LIMIT = 12.0
TOTAL_TIME = 16000.0
WINDOW = 4000.0

figure, axes = plt.subplots(2, 3, figsize=(14.6, 9.2), constrained_layout=True, sharex=True, sharey=True)
cmap = ListedColormap(("white", "black"))
norm = BoundaryNorm((-0.5, 0.5, 1.5), cmap.N)
summary = []

for axis, (sr, factor) in zip(axes.ravel(), POINT_SPECS):
    somega = float(locking_limit(sr)) * factor
    data_path = OUT / f"sr{sr:g}_sw{somega:.7g}.npz"
    if data_path.exists():
        saved = np.load(data_path)
        v0, s0 = saved["v0"], saved["s0"]
        status, metrics, locked = saved["status"], saved["metrics"], saved["locked"]
    else:
        v0, s0, status, metrics = run_sweep(
            sr,
            somega,
            -LIMIT,
            LIMIT,
            -LIMIT,
            LIMIT,
            K,
            1.0e-3,
            TOTAL_TIME,
            observation_window=WINDOW,
            sample_time=0.2,
        )
        locked = phase_locked_mask(status, metrics)
    locked_fraction = float(np.mean(locked))
    summary.append((sr, somega, factor, locked_fraction))
    np.savez_compressed(
        data_path,
        sr=sr,
        somega=somega,
        factor=factor,
        v0=v0,
        s0=s0,
        status=status,
        metrics=metrics,
        locked=locked,
    )

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
    axis.set_title(
        rf"$s_r={sr:g}$, $s_\omega={somega:.4g}$ "
        rf"({factor:.2f}$\times$ limit); locked={locked_fraction:.1%}"
    )
    axis.grid(alpha=0.15)
    print(
        f"SUMMARY sr={sr:g} somega={somega:.9g} factor={factor:g} "
        f"locked_fraction={locked_fraction:.8f}",
        flush=True,
    )

for axis in axes[-1, :]:
    axis.set_xlabel(r"initial $v_0$")
for axis in axes[:, 0]:
    axis.set_ylabel(r"initial slip $S_0=v_0+\alpha\omega_0$")
figure.suptitle(
    rf"Wide initial-condition sweeps near the phase-lock boundary; "
    rf"$v_0,S_0\in[-{LIMIT:g},{LIMIT:g}]$, $k={K}$, $T={TOTAL_TIME:g}$"
    "\n"
    r"white: phase-locked; black: running; orange: finite-time classification front"
)
figure.savefig(FIGURES / "initial_condition_boundaries.png", dpi=220)
np.savetxt(
    RESULTS / "summary.csv",
    np.asarray(summary),
    delimiter=",",
    header="sr,somega,boundary_factor,locked_fraction",
    comments="",
)
print(f"Saved final results to {OUT}", flush=True)
