# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Bracket the horizontal s_omega boundary at a given s_r by bisection.

Classification matches the phase-diagram GPU scan: sample
``phase_rate = 1 - v - omega`` over the final observation window and call a
trajectory locked when ``abs(mean_phase_rate) < threshold``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from compare_boundary_trajectories import initial_state, load_fp64, parameters
from project_paths import RESULTS_DIR, fp64_module_dir


def positive(parser: argparse.ArgumentParser, option: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{option} must be finite and positive")


def aligned_steps(parser: argparse.ArgumentParser, option: str, duration: float, dt: float) -> int:
    steps = round(duration / dt)
    if steps < 1 or not math.isclose(steps * dt, duration, rel_tol=0.0, abs_tol=max(1e-12, 64 * np.finfo(float).eps * duration)):
        parser.error(f"{option} must be an integer multiple of --dt")
    if steps > np.iinfo(np.int32).max:
        parser.error(f"{option} produces more than int32 integration steps")
    return steps


def mean_phase_rate(hysteresis, sr, somega, *, dt, total_time, observation_window, sample_time):
    """Use the same terminal-window sampling convention as phase_behavior.cu."""
    alpha, params = parameters(hysteresis, sr, somega)
    state = initial_state(0.0, 0.0, alpha)
    total_steps = round(total_time / dt)
    window_steps = min(total_steps, round(observation_window / dt))
    sample_steps = round(sample_time / dt)
    window_start = total_steps - window_steps

    # The GPU samples at completed=window_start and then every sample_stride.
    # There is no t=0 sample when the observation window covers the whole run.
    pre_steps = window_start if window_start > 0 else sample_steps
    state, _ = hysteresis.evolve_with_mode(state, dt, pre_steps, params)
    completed = pre_steps
    values = [float(1.0 - state[2] - state[4])]
    while completed + sample_steps <= total_steps:
        state, _ = hysteresis.evolve_with_mode(state, dt, sample_steps, params)
        completed += sample_steps
        values.append(float(1.0 - state[2] - state[4]))
    if len(values) < 2:
        raise RuntimeError("observation window produced fewer than two samples")
    array = np.asarray(values)
    return float(np.mean(array)), float(np.std(array, ddof=1)), len(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr", type=float, required=True)
    parser.add_argument("--lower", type=float, default=0.1, help="lower initial s_omega bound (default: GPU scan minimum)")
    parser.add_argument("--upper", type=float, default=3.0, help="upper initial s_omega bound (default: GPU scan maximum)")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="stop when bracket width is no larger")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--total-time", type=float, default=50_000.0)
    parser.add_argument("--observation-window", type=float, default=4_000.0)
    parser.add_argument("--sample-time", type=float, default=0.2)
    parser.add_argument("--module-dir", type=Path, default=fp64_module_dir())
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "phase-behavior" / "somega_bisection.csv")
    args = parser.parse_args()

    for name in ("sr", "lower", "upper", "tolerance", "threshold", "dt", "total_time", "observation_window", "sample_time"):
        positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.lower >= args.upper:
        parser.error("--lower must be less than --upper")
    if args.observation_window > args.total_time:
        parser.error("--observation-window cannot exceed --total-time")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    aligned_steps(parser, "--total-time", args.total_time, args.dt)
    window_steps = aligned_steps(parser, "--observation-window", args.observation_window, args.dt)
    sample_steps = aligned_steps(parser, "--sample-time", args.sample_time, args.dt)
    if window_steps < 2 * sample_steps:
        parser.error("observation window must contain at least two sample intervals")

    hysteresis = load_fp64(args.module_dir)
    common = dict(dt=args.dt, total_time=args.total_time, observation_window=args.observation_window, sample_time=args.sample_time)
    rows = []

    def evaluate(somega, label, iteration):
        mean, std, count = mean_phase_rate(hysteresis, args.sr, somega, **common)
        locked = abs(mean) < args.threshold
        rows.append((iteration, label, somega, mean, std, count, int(locked)))
        print(f"{label:>8}  s_omega={somega:.12g}  mean={mean:+.9g}  std={std:.6g}  {'LOCKED' if locked else 'RUNNING'}", flush=True)
        return locked

    lower, upper = args.lower, args.upper
    lower_locked = evaluate(lower, "lower", 0)
    upper_locked = evaluate(upper, "upper", 0)
    if lower_locked == upper_locked:
        parser.error("initial endpoints have the same phase classification; provide a bracket spanning the boundary")

    iteration = 0
    while upper - lower > args.tolerance and iteration < args.max_iterations:
        iteration += 1
        midpoint = (lower + upper) / 2.0
        midpoint_locked = evaluate(midpoint, "midpoint", iteration)
        if midpoint_locked == lower_locked:
            lower = midpoint
            lower_locked = midpoint_locked
        else:
            upper = midpoint
            upper_locked = midpoint_locked
        print(f"bracket  [{lower:.12g}, {upper:.12g}]  width={upper-lower:.6g}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("iteration", "role", "somega", "mean_phase_rate", "std_phase_rate", "sample_count", "locked"))
        writer.writerows(rows)
    print(f"FINAL sr={args.sr:g}: {lower:.12g} < s_omega,c < {upper:.12g}; midpoint={(lower+upper)/2:.12g}; width={upper-lower:.6g}")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
