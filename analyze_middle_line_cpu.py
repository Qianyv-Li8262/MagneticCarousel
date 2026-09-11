"""Diagnose the middle ridge in the initial-state minimum-velocity map.

This uses the same compiled CPU solver and parameter definitions as
``animate_hysteresis_cpu.py``.  It compares nearby initial states across the
ridge and writes a compact figure showing their transient and long-time
behaviour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hysteresis
from animate_hysteresis_cpu import DT, make_initial_state, scaled_model_parameters


S0 = 0.0
POINTS = {
    "left of ridge": -0.0460,
    "on ridge": -0.0450,
    "right of ridge": -0.0440,
}


def solver_setup():
    alpha, delta, gamma_g, gamma, gamma_m = scaled_model_parameters()
    params = hysteresis.SolverParams(alpha, delta, gamma_g, gamma, gamma_m)
    return alpha, params


def sampled_evolution(v0: float, s0: float, total_time: float = 10_000.0):
    """Return a fine early trace and a coarse full trace."""
    alpha, params = solver_setup()
    state = make_initial_state(v0, s0, alpha)

    early_times = [0.0]
    early_v = [float(state[2])]
    early_s = [float(state[2] + alpha * state[4])]
    early_modes = [abs(early_s[-1]) > 1.0e-10]

    early_time = 150.0
    early_chunk_steps = 10
    early_count = round(early_time / (DT * early_chunk_steps))
    for _ in range(early_count):
        state, sliding = hysteresis.evolve_with_mode(state, DT, early_chunk_steps, params)
        early_times.append(float(state[0]))
        early_v.append(float(state[2]))
        early_s.append(float(state[2] + alpha * state[4]))
        early_modes.append(bool(sliding))

    coarse_times = [early_times[-1]]
    coarse_v = [early_v[-1]]
    coarse_chunk_steps = 1000
    coarse_count = round((total_time - early_time) / (DT * coarse_chunk_steps))
    for _ in range(coarse_count):
        state, _sliding = hysteresis.evolve_with_mode(state, DT, coarse_chunk_steps, params)
        coarse_times.append(float(state[0]))
        coarse_v.append(float(state[2]))

    return {
        "early_t": np.asarray(early_times),
        "early_v": np.asarray(early_v),
        "early_s": np.asarray(early_s),
        "early_modes": np.asarray(early_modes),
        "coarse_t": np.asarray(coarse_times),
        "coarse_v": np.asarray(coarse_v),
    }


def minimum_cross_section(s0: float, v_values: np.ndarray, total_time: float = 150.0):
    """Sample the signed velocity minimum and the time at which it occurs."""
    alpha, params = solver_setup()
    chunk_steps = 50
    chunk_count = round(total_time / (DT * chunk_steps))
    minimum_v = np.empty_like(v_values)
    minimum_t = np.empty_like(v_values)

    for index, v0 in enumerate(v_values):
        state = make_initial_state(float(v0), s0, alpha)
        best_v = float(v0)
        best_t = 0.0
        for _ in range(chunk_count):
            state, _sliding = hysteresis.evolve_with_mode(state, DT, chunk_steps, params)
            velocity = float(state[2])
            if velocity < best_v:
                best_v = velocity
                best_t = float(state[0])
        minimum_v[index] = best_v
        minimum_t[index] = best_t
    return minimum_v, minimum_t


def block_means(times: np.ndarray, values: np.ndarray, width: float = 500.0):
    edges = np.arange(0.0, times[-1] + width, width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(centers.shape, np.nan)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (times >= lower) & (times < upper)
        if np.any(selected):
            means[index] = np.mean(values[selected])
    return centers, means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/middle_line_cpu_comparison.png"),
        help="output figure",
    )
    args = parser.parse_args()

    traces = {label: sampled_evolution(v0, S0) for label, v0 in POINTS.items()}
    v_values = np.linspace(-0.052, -0.037, 61)
    minimum_v, minimum_t = minimum_cross_section(S0, v_values)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    colors = ("tab:blue", "tab:red", "tab:green")

    for (label, v0), color in zip(POINTS.items(), colors):
        trace = traces[label]
        axes[0].plot(trace["early_t"], trace["early_v"], color=color, label=f"{label}: $v_0={v0:g}$")
        index = int(np.argmin(trace["early_v"]))
        axes[0].plot(trace["early_t"][index], trace["early_v"][index], "o", color=color, ms=4)
    axes[0].set(xlabel="$t$", ylabel="$v(t)$", title="Early transient (CPU solver)", xlim=(0, 100))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(v_values, minimum_v, c=minimum_t, cmap="viridis", s=24)
    axes[1].axvline(POINTS["on ridge"], color="tab:red", ls="--", lw=1)
    axes[1].set(
        xlabel="$v_0$ at $S_0=0$",
        ylabel="minimum $v$ over $0\\leq t\\leq150$",
        title="The ridge switches the selected trough",
    )
    axes[1].grid(alpha=0.25)
    figure.colorbar(scatter, ax=axes[1], label="time of selected minimum")

    for (label, _v0), color in zip(POINTS.items(), colors):
        trace = traces[label]
        full_t = np.concatenate((trace["early_t"], trace["coarse_t"][1:]))
        full_v = np.concatenate((trace["early_v"], trace["coarse_v"][1:]))
        centers, means = block_means(full_t, full_v)
        axes[2].plot(centers, means, marker=".", ms=3, color=color, label=label)
    axes[2].axhline(-0.36361, color="0.25", ls="--", lw=1, label="common late mean")
    axes[2].set(
        xlabel="$t$",
        ylabel="500-time-unit mean of $v$",
        title="Both sides approach the same state",
        xlim=(0, 10_000),
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle(r"Middle line at $s_r=0.8$, $s_\omega=0.6$, $S_0=0$: transient ridge, not a phase boundary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)

    print(f"Saved {args.output.resolve()}")
    for label, v0 in POINTS.items():
        trace = traces[label]
        early_index = int(np.argmin(trace["early_v"]))
        tail = trace["coarse_v"][-1000:]
        print(
            f"{label:>14}  v0={v0:+.4f}  "
            f"early_min={trace['early_v'][early_index]:+.7f} at t={trace['early_t'][early_index]:.2f}  "
            f"tail_mean={np.mean(tail):+.7f} tail_range=[{np.min(tail):+.7f}, {np.max(tail):+.7f}]  "
            f"sliding_samples={np.count_nonzero(trace['early_modes'])}"
        )


if __name__ == "__main__":
    main()
