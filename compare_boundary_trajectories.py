"""Long FP64 CPU integrations on both sides of the true phase boundary."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SR = 0.8
SOMEGA = 0.6
ALPHA_REF = 0.33333333
DELTA_REF = 1.7212357e-4
GAMMA_G_REF = 0.16560356
GAMMA_REF = 489.39639
GAMMA_M_REF = 11011.419
DT = 1.0e-3


def load_fp64(module_dir: Path):
    sys.path.insert(0, str(module_dir.resolve()))
    hysteresis = importlib.import_module("hysteresis")
    if np.dtype(hysteresis.real_dtype) != np.dtype(np.float64):
        raise RuntimeError(f"Expected an FP64 module, loaded {hysteresis.real_dtype} from {hysteresis.__file__}")
    return hysteresis


def parameters(hysteresis):
    alpha = ALPHA_REF * SR
    params = hysteresis.SolverParams(
        alpha,
        DELTA_REF / SR,
        GAMMA_G_REF / SOMEGA**2,
        GAMMA_REF / SOMEGA**2,
        GAMMA_M_REF / (SR**2 * SOMEGA**2),
    )
    return alpha, params


def initial_state(v0: float, s0: float, alpha: float):
    return np.asarray((0.0, 0.0, v0, 0.0, (s0 - v0) / alpha, 0.0, 0.0, 0.0), dtype=np.float64)


def trace(hysteresis, v0: float, s0: float, total_time: float, sample_time: float):
    alpha, params = parameters(hysteresis)
    state = initial_state(v0, s0, alpha)
    steps = round(sample_time / DT)
    count = round(total_time / sample_time)
    rows = []
    previous_x = float(state[1])
    for _ in range(count):
        state, sliding = hysteresis.evolve_with_mode(state, DT, steps, params)
        block_velocity = (float(state[1]) - previous_x) / sample_time
        previous_x = float(state[1])
        phase_rate = float(1.0 - state[2] - state[4])
        rows.append(
            (
                float(state[0]),
                float(state[1]),
                float(state[2]),
                block_velocity,
                phase_rate,
                float(state[2] + alpha * state[4]),
                int(sliding),
            )
        )
    return np.asarray(rows)


def save_csv(path: Path, data: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("t", "x", "instantaneous_v", "block_mean_v", "instantaneous_phase_rate", "slip", "sliding"))
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=Path("tmp/fp64"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/true_boundary_fp64"))
    parser.add_argument("--near-time", type=float, default=100_000.0)
    parser.add_argument("--ultra-time", type=float, default=1_000_000.0)
    parser.add_argument("--left-v0", type=float, default=0.189186667, help="Initial velocity at the left analysis point")
    parser.add_argument("--right-v0", type=float, default=0.189213333, help="Initial velocity at the right analysis point")
    parser.add_argument("--long-v0", type=float, default=0.19000, help="Initial velocity for the long-time analysis")
    args = parser.parse_args()
    for name in ("left_v0", "right_v0", "long_v0"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name, sample_time in (("near_time", 1000.0), ("ultra_time", 10_000.0)):
        value = getattr(args, name)
        if not np.isfinite(value) or value < sample_time:
            parser.error(f"--{name.replace('_', '-')} must be finite and at least {sample_time:g}")

    hysteresis = load_fp64(args.module_dir)
    alpha, _params = parameters(hysteresis)
    locked_velocity = -alpha / (1.0 - alpha)

    cases = {
        f"left_v0_{args.left_v0}": trace(hysteresis, args.left_v0, 0.0, args.near_time, 1000.0),
        f"right_v0_{args.right_v0}": trace(hysteresis, args.right_v0, 0.0, args.near_time, 1000.0),
        f"ultralong_v0_{args.long_v0}": trace(hysteresis, args.long_v0, 0.0, args.ultra_time, 10_000.0),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in cases.items():
        save_csv(args.output_dir / f"{name}.csv", data)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), constrained_layout=True)
    left, right, ultra = cases.values()
    left_label = rf"Left: $v_0={args.left_v0}$"
    right_label = rf"Right: $v_0={args.right_v0}$"

    axes[0, 0].plot(left[:, 0], left[:, 3], label=left_label)
    axes[0, 0].plot(right[:, 0], right[:, 3], label=right_label)
    axes[0, 0].axhline(locked_velocity, color="0.25", ls="--", label=rf"$-\alpha/(1-\alpha)={locked_velocity:.6f}$")
    axes[0, 0].set(xlabel="$t$", ylabel="block-mean $v$", title=rf"Analysis points: $|\Delta v_0|={abs(args.right_v0 - args.left_v0):.6g}$")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(left[:, 0], left[:, 4], label=left_label)
    axes[0, 1].plot(right[:, 0], right[:, 4], label=right_label)
    axes[0, 1].axhline(0.0, color="0.25", ls="--")
    axes[0, 1].set(xlabel="$t$", ylabel=r"instantaneous $\dot\phi=1-v-\omega$", title="Phase-rate comparison")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].loglog(ultra[:, 0], np.abs(ultra[:, 2]), "o-", ms=3, label=rf"Long: $v_0={args.long_v0}$")
    reference = abs(ultra[-1, 2]) * np.sqrt(ultra[-1, 0] / ultra[:, 0])
    axes[1, 0].loglog(ultra[:, 0], reference, ls="--", color="0.3", label=r"$t^{-1/2}$ guide")
    axes[1, 0].set(xlabel="$t$", ylabel="$|v|$", title=rf"Long-time analysis to $t={ultra[-1, 0]:g}$")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    summary = (
        "FP64 CPU result at $S_0=0$\n\n"
        f"Left point: $v_0={args.left_v0}$\n"
        f"Right point: $v_0={args.right_v0}$\n"
        f"Long-time point: $v_0={args.long_v0}$\n\n"
        f"Left/right end time: {left[-1, 0]:g} / {right[-1, 0]:g}\n"
        f"Long-time end time: {ultra[-1, 0]:g}\n"
        f"Integration step: $\\Delta t={DT:g}$\n\n"
        "Phase-locked reference:\n"
        f"$v={locked_velocity:.6f}$,  $\\dot\\phi=0$"
    )
    axes[1, 1].text(0.03, 0.95, summary, va="top", fontsize=12)

    for axis in axes.ravel()[:3]:
        axis.grid(alpha=0.25)
    figure.suptitle(r"True initial-state boundary: locked transport versus unlocked decay ($s_r=0.8$, $s_\omega=0.6$)")
    figure_path = args.output_dir / "true_boundary_fp64.png"
    figure.savefig(figure_path, dpi=200)

    print(f"FP64 module: {hysteresis.__file__}")
    print(f"Saved {figure_path.resolve()}")
    for name, data in cases.items():
        print(
            f"{name}: t={data[-1,0]:.0f}, v={data[-1,2]:+.9f}, "
            f"block_mean_v={data[-1,3]:+.9f}, phase_rate={data[-1,4]:+.9f}, slip={data[-1,5]:+.3e}"
        )


if __name__ == "__main__":
    main()
