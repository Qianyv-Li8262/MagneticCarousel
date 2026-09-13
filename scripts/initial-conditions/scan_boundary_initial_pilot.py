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
from project_paths import RESULTS_DIR, RUNS_DIR


OUT = RUNS_DIR / "initial-conditions" / "boundary-pilot"
RESULTS = RESULTS_DIR / "initial-conditions" / "boundary-pilot"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SRS = (2.5, 2.7, 2.85, 3.15, 3.3, 3.5)
FACTORS = (0.95, 1.05)
K = 40
LIMIT = 10.0
TOTAL_TIME = 8000.0
WINDOW = 2000.0

points = [(sr, float(locking_limit(sr)) * factor, factor) for sr in SRS for factor in FACTORS]
figure, axes = plt.subplots(3, 4, figsize=(14, 10.2), constrained_layout=True, sharex=True, sharey=True)
cmap = ListedColormap(("white", "black"))
norm = BoundaryNorm((-0.5, 0.5, 1.5), cmap.N)
summary = []

for axis, (sr, somega, factor) in zip(axes.ravel(), points):
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
        OUT / f"sr{sr:g}_sw{somega:.6g}.npz",
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
    axis.set_title(
        rf"$s_r={sr:g}$, $s_\omega={somega:.3g}$ ({factor:.2f}$\times$); "
        rf"locked={locked_fraction:.1%}"
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
    axis.set_ylabel(r"initial slip $S_0$")
figure.suptitle(
    rf"Pilot initial-condition sweep near locking boundary; "
    rf"$v_0,S_0\in[-{LIMIT:g},{LIMIT:g}]$, $k={K}$, $T={TOTAL_TIME:g}$\n"
    r"white: phase-locked; black: running"
)
figure.savefig(RESULTS / "pilot_grid.png", dpi=180)
np.savetxt(
    RESULTS / "summary.csv",
    np.asarray(summary),
    delimiter=",",
    header="sr,somega,boundary_factor,locked_fraction",
    comments="",
)
print(f"Saved pilot results to {OUT}", flush=True)
