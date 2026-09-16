"""Run the C++ threaded boundary scan and plot its bisection brackets."""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTS_DIR, RUNS_DIR, SCRATCH_DIR, fp64_module_dir

REFERENCE = (0.33333333, 1.7212357e-4, 0.16560356, 489.39639, 11011.419)


def load_module(path: Path):
    sys.path.insert(0, str(path.resolve()))
    try:
        return importlib.import_module("hysteresis")
    finally:
        sys.path.pop(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-lower", type=float, required=True)
    parser.add_argument("--sr-upper", type=float, required=True)
    parser.add_argument("--plot-points", type=int, required=True)
    parser.add_argument("--somega-lower", type=float, default=0.1)
    parser.add_argument("--somega-upper", type=float, default=3.0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--total-time", type=float, default=50_000.0)
    parser.add_argument("--observation-window", type=float, default=4_000.0)
    parser.add_argument("--sample-time", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--threads", type=int, default=0, help="0 uses all logical CPU cores")
    parser.add_argument("--module-dir", type=Path, default=fp64_module_dir())
    parser.add_argument("--data", type=Path, default=RUNS_DIR / "phase-behavior" / "cpp_phase_boundary.npz")
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "phase-behavior" / "cpp_phase_boundary.csv")
    parser.add_argument("--figure", type=Path, default=SCRATCH_DIR / "phase-behavior" / "cpp_phase_boundary.png")
    args = parser.parse_args()

    module = load_module(args.module_dir)
    if not hasattr(module, "bisect_phase_boundaries"):
        parser.error("FP64 extension lacks bisect_phase_boundaries; rebuild with: python src/build.py --fp64")
    reference = module.SolverParams(*REFERENCE)
    result = module.bisect_phase_boundaries(
        args.sr_lower, args.sr_upper, args.plot_points,
        args.somega_lower, args.somega_upper, args.tolerance, args.threshold,
        args.dt, args.total_time, args.observation_window, args.sample_time,
        args.max_iterations, reference, args.threads,
    )

    args.data.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.data, **result)
    columns = ("sr", "lower", "upper", "midpoint", "lower_mean_phase_rate", "upper_mean_phase_rate", "iterations", "status")
    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(zip(*(result[name] for name in columns), strict=True))

    success = result["status"] == 0
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    if np.any(success):
        sr = result["sr"][success]
        lower = result["lower"][success]
        upper = result["upper"][success]
        axis.fill_between(sr, lower, upper, color="tab:blue", alpha=0.25, label="bisection bracket")
        axis.plot(sr, result["midpoint"][success], "o-", ms=3, lw=1.2, color="tab:blue", label="bracket midpoint")
    failed = ~success
    if np.any(failed):
        axis.scatter(result["sr"][failed], np.full(np.count_nonzero(failed), args.somega_lower),
                     marker="x", color="tab:red", label="unbracketed/error")
    axis.set(xlabel=r"$s_r$", ylabel=r"critical $s_\omega$",
             title=rf"FP64 C++ parallel phase boundary ($N={args.plot_points}$, $T={args.total_time:g}$)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.figure, dpi=220)
    plt.close(figure)
    print(f"successful points: {np.count_nonzero(success)}/{args.plot_points}")
    print(f"saved data: {args.data.resolve()}")
    print(f"saved csv: {args.csv.resolve()}")
    print(f"saved figure: {args.figure.resolve()}")


if __name__ == "__main__":
    main()
