"""Compare two long FP64 CPU trajectories above and below a horizontal boundary."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RUNS_DIR, fp64_module_dir, image_output_path

ALPHA_REF = 0.33333333
DELTA_REF = 1.7212357e-4
GAMMA_G_REF = 0.16560356
GAMMA_REF = 489.39639
GAMMA_M_REF = 11011.419


def load_fp64(module_dir: Path):
    sys.path.insert(0, str(module_dir.resolve()))
    hysteresis = importlib.import_module("hysteresis")
    if np.dtype(hysteresis.real_dtype) != np.dtype(np.float64):
        raise RuntimeError(f"Expected FP64, loaded {hysteresis.real_dtype} from {hysteresis.__file__}")
    return hysteresis


def parameters(hysteresis, sr: float, somega: float):
    alpha = ALPHA_REF * sr
    inv_sw2 = 1.0 / somega**2
    return alpha, hysteresis.SolverParams(
        alpha, DELTA_REF / sr, GAMMA_G_REF * inv_sw2,
        GAMMA_REF * inv_sw2, GAMMA_M_REF / sr**2 * inv_sw2,
    )


def initial_state(v0: float, s0: float, alpha: float):
    return np.asarray((0.0, 0.0, v0, 0.0, (s0 - v0) / alpha, 0.0, 0.0, 0.0), dtype=np.float64)


def trace(hysteresis, sr, somega, v0, s0, total_time, sample_time, dt):
    alpha, params = parameters(hysteresis, sr, somega)
    state = initial_state(v0, s0, alpha)
    steps = round(sample_time / dt)
    rows = []
    previous_x = float(state[1])
    for _ in range(round(total_time / sample_time)):
        state, sliding = hysteresis.evolve_with_mode(state, dt, steps, params)
        block_velocity = (float(state[1]) - previous_x) / sample_time
        previous_x = float(state[1])
        rows.append((
            float(state[0]), float(state[1]), float(state[2]), block_velocity,
            float(state[4]), float(1.0 - state[2] - state[4]),
            float(state[2] + alpha * state[4]), int(sliding),
        ))
    return np.asarray(rows)


def save_csv(path: Path, data: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("t", "x", "instantaneous_v", "block_mean_v", "instantaneous_omega", "instantaneous_phase_rate", "slip", "sliding"))
        writer.writerows(data)


def positive(parser, option, value):
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{option} must be finite and positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=fp64_module_dir())
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "phase-behavior" / "horizontal-boundary-trajectories")
    parser.add_argument("--sr", type=float, default=2.0)
    parser.add_argument("--lower-somega", type=float, default=1.12890625, help="point below the horizontal feature")
    parser.add_argument("--upper-somega", type=float, default=1.130859375, help="point above the horizontal feature")
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--s0", type=float, default=0.0)
    parser.add_argument("--total-time", type=float, default=50_000.0)
    parser.add_argument("--sample-time", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    args = parser.parse_args()

    for name in ("sr", "lower_somega", "upper_somega", "total_time", "sample_time", "dt"):
        positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.lower_somega >= args.upper_somega:
        parser.error("--lower-somega must be less than --upper-somega")
    if args.sample_time > args.total_time:
        parser.error("--sample-time cannot exceed --total-time")
    if not all(math.isfinite(x) for x in (args.v0, args.s0)):
        parser.error("--v0 and --s0 must be finite")

    hysteresis = load_fp64(args.module_dir)
    cases = {
        "below": trace(hysteresis, args.sr, args.lower_somega, args.v0, args.s0, args.total_time, args.sample_time, args.dt),
        "above": trace(hysteresis, args.sr, args.upper_somega, args.v0, args.s0, args.total_time, args.sample_time, args.dt),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in cases.items():
        save_csv(args.output_dir / f"{name}.csv", data)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    labels = {"below": rf"below: $s_\omega={args.lower_somega:g}$", "above": rf"above: $s_\omega={args.upper_somega:g}$"}
    series = (
        (2, r"instantaneous $v$", "Translational velocity"),
        (4, r"instantaneous $\omega$", "Angular velocity"),
        (5, r"$\dot\phi=1-v-\omega$", "Phase drift"),
        (6, r"slip $v+\alpha\omega$", "Contact slip"),
    )
    for axis, (column, ylabel, title) in zip(axes.ravel(), series, strict=True):
        for name, data in cases.items():
            axis.plot(data[:, 0], data[:, column], label=labels[name], linewidth=1.05)
        axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.8)
        axis.set(xlabel="$t$", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(rf"Trajectories across the horizontal feature at $s_r={args.sr:g}$")
    figure_path = image_output_path(args.output_dir / "horizontal_boundary_comparison.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    print(f"FP64 module: {hysteresis.__file__}")
    print(f"Saved {figure_path.resolve()}")
    for name, data in cases.items():
        tail_count = max(1, round(min(4000.0, args.total_time) / args.sample_time))
        tail = data[-tail_count:]
        sw = args.lower_somega if name == "below" else args.upper_somega
        print(f"{name} somega={sw:g}: final v={data[-1,2]:+.9g}, omega={data[-1,4]:+.9g}, tail mean phase_rate={np.mean(tail[:,5]):+.9g}, tail std={np.std(tail[:,5]):.9g}")


if __name__ == "__main__":
    main()
