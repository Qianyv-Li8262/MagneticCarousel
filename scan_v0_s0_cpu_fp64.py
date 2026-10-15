# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Scan the 1D initial-state plane with the threaded CPU FP64 model.

Example (from the workspace root)::

    python src/scan_v0_s0_cpu_fp64.py --threads 0 --total-time 112000

The output NPZ is checkpointed after every row batch. Pass --resume after an
interruption; --overwrite is required to replace an existing scan.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from project_paths import RUNS_DIR, SCRATCH_DIR, extension_dir
from scan_sr_somega_gpu import DEFAULT_REFERENCE, METRIC_NAMES
from scan_v0_s0_gpu import plot_results


DEFAULT_SR = 2.5
DEFAULT_SOMEGA = 2.25447
DEFAULT_DT = 1.0e-3
DEFAULT_TOTAL_TIME = 112000.0
DEFAULT_WINDOW = 4000.0
DEFAULT_SAMPLE_TIME = 0.2


def load_fp64_extension():
    sys.path.insert(0, str(extension_dir("fp64")))
    import hysteresis

    if not hasattr(hysteresis, "scan_initial_grid_cpu"):
        raise RuntimeError("CPU FP64 extension lacks scan_initial_grid_cpu; rebuild hysteresis_fp64")
    return hysteresis


def steps_for(duration: float, dt: float, name: str) -> int:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{name} must be finite and positive")
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{name} must be an integer number of dt steps")
    return steps


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **data)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=64, help="grid points per axis")
    parser.add_argument("--v0-lower", type=float, default=-1.0)
    parser.add_argument("--v0-upper", type=float, default=1.0)
    parser.add_argument("--s0-lower", type=float, default=-14.0)
    parser.add_argument("--s0-upper", type=float, default=14.0)
    parser.add_argument("--sr", type=float, default=DEFAULT_SR)
    parser.add_argument("--somega", type=float, default=DEFAULT_SOMEGA)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--total-time", type=float, default=DEFAULT_TOTAL_TIME)
    parser.add_argument("--observation-window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument("--sample-time", type=float, default=DEFAULT_SAMPLE_TIME)
    parser.add_argument("--threads", type=int, default=0, help="0 uses all logical CPU cores")
    parser.add_argument("--batch-rows", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.k < 1 or args.batch_rows < 1 or args.threads < 0:
        parser.error("k and batch-rows must be positive; threads must be nonnegative")
    if args.resume and args.overwrite:
        parser.error("choose either --resume or --overwrite")
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("dt must be finite and positive")
    if not all(np.isfinite(x) for x in (args.v0_lower, args.v0_upper, args.s0_lower, args.s0_upper)):
        parser.error("grid bounds must be finite")
    if args.v0_lower > args.v0_upper or args.s0_lower > args.s0_upper:
        parser.error("grid bounds must be ordered")
    steps = steps_for(args.total_time, args.dt, "total-time")
    window_steps = steps_for(args.observation_window, args.dt, "observation-window")
    sample_stride = steps_for(args.sample_time, args.dt, "sample-time")
    if window_steps > steps or window_steps < 2 or window_steps // sample_stride < 2:
        parser.error("observation window must fit within the run and contain at least two samples")

    output = args.output or RUNS_DIR / "initial-conditions" / f"cpu_fp64_k{args.k}_T{args.total_time:g}.npz"
    plot_path = args.plot or SCRATCH_DIR / "cmp_init_state" / f"cpu_fp64_k{args.k}_T{args.total_time:g}.png"
    v0 = np.linspace(args.v0_lower, args.v0_upper, args.k, dtype=np.float64)
    s0 = np.linspace(args.s0_lower, args.s0_upper, args.k, dtype=np.float64)
    extension = load_fp64_extension()
    reference = extension.SolverParams(
        DEFAULT_REFERENCE.alpha, DEFAULT_REFERENCE.delta,
        DEFAULT_REFERENCE.gamma_g, DEFAULT_REFERENCE.gamma,
        DEFAULT_REFERENCE.gamma_m,
    )
    signature = dict(
        sr=np.asarray(args.sr), somega=np.asarray(args.somega),
        dt=np.asarray(args.dt), steps=np.asarray(steps),
        window_steps=np.asarray(window_steps), sample_stride=np.asarray(sample_stride),
        reference_params=np.asarray([
            reference.alpha, reference.delta, reference.gamma_g,
            reference.gamma, reference.gamma_m,
        ]),
        h_ext=np.asarray(extension.scan_h_ext),
        field_z0=np.asarray(extension.scan_field_z0),
        field_fourier_mode_count=np.asarray(extension.scan_field_fourier_mode_count),
        field_legacy_first_harmonic=np.asarray(extension.scan_field_legacy_first_harmonic),
    )

    if args.resume:
        if not output.is_file():
            parser.error(f"checkpoint not found: {output}")
        with np.load(output) as saved:
            if not np.array_equal(saved["v0"], v0) or not np.array_equal(saved["s0"], s0):
                parser.error("checkpoint grid does not match this run")
            for key, expected in signature.items():
                if key not in saved or not np.array_equal(saved[key], expected):
                    parser.error(f"checkpoint {key} does not match this build/run")
            status = saved["status"].copy()
            metrics = saved["metrics"].copy()
            completed_rows = int(saved["completed_rows"])
        if not (0 <= completed_rows <= args.k):
            parser.error("checkpoint completed_rows is invalid")
    else:
        if output.exists() and not args.overwrite:
            parser.error(f"output exists: {output}; pass --resume or --overwrite")
        status = np.full((args.k, args.k), 4, dtype=np.int32)
        metrics = np.full((args.k, args.k, len(METRIC_NAMES)), np.nan, dtype=np.float64)
        completed_rows = 0

    started = time.monotonic()
    print(
        f"CPU FP64: {args.k}x{args.k}, T={args.total_time:g}, dt={args.dt:g}, "
        f"threads={'all logical' if args.threads == 0 else args.threads}, "
        f"H_EXT={extension.scan_h_ext:g}, Z0={extension.scan_field_z0:g}, "
        f"modes={extension.scan_field_fourier_mode_count}",
        flush=True,
    )
    for row in range(completed_rows, args.k, args.batch_rows):
        end = min(row + args.batch_rows, args.k)
        batch_status, batch_metrics = extension.scan_initial_grid_cpu(
            v0, s0[row:end], args.sr, args.somega, args.dt,
            steps, window_steps, sample_stride, reference, args.threads,
        )
        status[row:end] = batch_status
        metrics[row:end] = batch_metrics
        data = dict(
            v0=v0, s0=s0, status=status, metrics=metrics,
            metric_names=np.asarray(METRIC_NAMES), completed_rows=np.asarray(end),
            phase_locked=(status == 0) & (np.abs(metrics[..., 3]) < 0.05),
            lock_threshold=np.asarray(0.05),
            kahan=np.asarray(False), fp64=np.asarray(True), backend=np.asarray("cpu"),
            **signature,
        )
        save_checkpoint(output, data)
        print(
            f"rows {end}/{args.k}; completed {end*args.k}/{args.k*args.k}; "
            f"elapsed {time.monotonic()-started:.1f} s; "
            f"failed {np.count_nonzero(status[:end] != 0)}; saved {output}",
            flush=True,
        )

    if not args.no_plot:
        fig = plot_results(v0, s0, status, metrics, args.sr, args.somega,
                           total_time=args.total_time,
                           observation_window=args.observation_window,
                           use_fp64=True)
        fig.suptitle(
            f"1D CPU FP64 initial-state scan: sr={args.sr:g}, "
            f"somega={args.somega:g}, K={args.k}, T={args.total_time:g}"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180)
        print(f"Saved plot: {plot_path}", flush=True)
    print(f"Done. Data: {output}", flush=True)


if __name__ == "__main__":
    main()
