"""Simple real-time animation for the CPU hysteresis solver.

Run from this directory with::

    python src/animate_hysteresis_cpu.py

The compiled ``hysteresis`` extension must be next to this file.  Rebuild it
with ``python src/build.py`` if it is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import hysteresis


def require_current_extension() -> None:
    """Fail visibly when Python has loaded an extension built from older sources."""
    extension_path = Path(hysteresis.__file__).resolve()
    source_directory = Path(__file__).resolve().parent
    model_sources = (
        source_directory / "1dmodel-hysteresis.hpp",
        source_directory / "1dmodel-hysteresis.cpp",
        source_directory / "pybind.cpp",
    )
    newer_sources = [source.name for source in model_sources if source.stat().st_mtime > extension_path.stat().st_mtime]
    if newer_sources:
        names = ", ".join(newer_sources)
        raise RuntimeError(
            f"Loaded {extension_path.name}, but these model sources are newer: {names}. "
            "Rebuild with `python build.py --fp64`, then restart Python."
        )


require_current_extension()


# Simulation settings.  The normal path advances one solver chunk per frame;
# --fast batches several unchanged chunks into each display refresh.
DT = 1.0e-3
STEPS_PER_FRAME = 10000
FPS = 30

# Fast display settings.  The full sampled history is retained; only rendering
# is batched and downsampled.
FAST_BATCH_SIZE = 5
FAST_GRAPH_UPDATE_EVERY = 5
FAST_RESCALE_EVERY = 20
FAST_MAX_PLOT_POINTS = 2000

# GPU-sweep-compatible model parameters.  SR and SOMEGA select one point
# relative to the five reference values below, using exactly the same scaling
# as scaled_params() in 1dmodel-hysteresis.cu.
SR = 2.9
SOMEGA = 1
ALPHA = 0.33333333
DELTA = 1.7212357e-4
GAMMA_G = 0.16560356
GAMMA = 489.39639
GAMMA_M = 11011.419

# State layout: [t, x, x_dot, theta, theta_dot, phase, q_x, q_y].
# In particular, the initial hysteresis vector q is the zero vector.
DEFAULT_INITIAL_V0 = -50
DEFAULT_INITIAL_S0 = 0

def scaled_model_parameters() -> tuple[float, float, float, float, float]:
    """Return the effective CPU parameters for the selected GPU sweep point."""
    if not np.isfinite(SR) or SR <= 0.0:
        raise ValueError("SR must be finite and positive")
    if not np.isfinite(SOMEGA) or SOMEGA <= 0.0:
        raise ValueError("SOMEGA must be finite and positive")

    inverse_sr = 1.0 / SR
    inverse_somega_squared = 1.0 / (SOMEGA * SOMEGA)
    return (
        ALPHA * SR,
        DELTA * inverse_sr,
        GAMMA_G * inverse_somega_squared,
        GAMMA * inverse_somega_squared,
        GAMMA_M * inverse_sr * inverse_sr * inverse_somega_squared,
    )


def make_initial_state(v0: float, s0: float, alpha: float) -> np.ndarray:
    """Build the CPU state from ``v0`` and contact-point slip ``s0``."""
    if not np.isfinite(v0) or not np.isfinite(s0):
        raise ValueError("v0 and s0 must be finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")

    state = np.zeros(8, dtype=hysteresis.real_dtype)
    state[2] = v0
    state[4] = (s0 - v0) / alpha
    return state


def minmax_downsample(
    times: list[float], values: list[float], max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a trace while retaining each time bucket's extrema."""
    count = len(times)
    if count <= max_points:
        return np.asarray(times), np.asarray(values)

    values_array = np.asarray(values)
    bucket_count = max(1, max_points // 2)
    edges = np.linspace(0, count, bucket_count + 1, dtype=np.int64)
    indices = []
    for start, stop in zip(edges[:-1], edges[1:]):
        bucket = values_array[start:stop]
        if bucket.size == 0:
            continue
        minimum = int(start + np.argmin(bucket))
        maximum = int(start + np.argmax(bucket))
        if minimum == maximum:
            indices.append(minimum)
        else:
            indices.extend(sorted((minimum, maximum)))
    if indices[-1] != count - 1:
        indices.append(count - 1)

    return np.asarray(times)[indices], values_array[indices]


def animate(
    *,
    show_graphs: bool = True,
    show_velocity: bool = False,
    fast: bool = False,
    initial_v0: float = DEFAULT_INITIAL_V0,
    initial_s0: float = DEFAULT_INITIAL_S0,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    if not show_graphs and show_velocity:
        raise ValueError("show_velocity requires show_graphs")

    alpha, delta, gamma_g, gamma, gamma_m = scaled_model_parameters()
    params = hysteresis.SolverParams(alpha, delta, gamma_g, gamma, gamma_m)
    state = make_initial_state(initial_v0, initial_s0, alpha)

    times = [float(state[0])]
    positions = [float(state[1])]
    angles = [float(state[3])]
    velocities = [float(state[2])]
    phase_rates = [1.0 - float(state[2]) - float(state[4])]

    if show_graphs:
        graph_rows = 4 if show_velocity else 3
        figure_height = 8.2 if show_velocity else 6.8
        figure_options = {} if fast else {"layout": "constrained"}
        fig = plt.figure(figsize=(12.5, figure_height), **figure_options)
        grid = fig.add_gridspec(graph_rows, 2, width_ratios=(1.7, 1.0))
        ax_scene = fig.add_subplot(grid[:, 0])
        ax_x = fig.add_subplot(grid[0, 1])
        ax_theta = fig.add_subplot(grid[1, 1], sharex=ax_x)
        ax_phase_rate = fig.add_subplot(grid[2, 1], sharex=ax_x)
        ax_velocity = fig.add_subplot(grid[3, 1], sharex=ax_x) if show_velocity else None
    else:
        figure_options = {} if fast else {"layout": "constrained"}
        fig, ax_scene = plt.subplots(figsize=(8.5, 4.5), **figure_options)
        ax_x = None
        ax_theta = None
        ax_phase_rate = None
        ax_velocity = None

    # Left: a ball on the ground with a right-moving sinusoidal wave above it.
    view_half_width = 2.0 * np.pi
    wave_height = 1.25 * alpha
    wave_amplitude = 0.18 * alpha
    scene_x = np.linspace(-view_half_width, view_half_width, 700)

    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_ylim(-0.18 * alpha, 1.65 * alpha)
    ax_scene.set_xlim(-view_half_width, view_half_width)
    ax_scene.set_yticks([])
    ax_scene.set_xlabel("x")
    ax_scene.spines[["left", "right", "top"]].set_visible(False)
    ax_scene.axhline(0.0, color="0.2", linewidth=1.5)

    (wave,) = ax_scene.plot(
        scene_x,
        wave_height + wave_amplitude * np.sin(-scene_x),
        color="tab:cyan",
        linewidth=2.0,
    )
    ball = Circle((0.0, alpha), alpha, edgecolor="0.15", linewidth=1.6)
    ax_scene.add_patch(ball)
    (radius_line,) = ax_scene.plot([], [], color="0.12", linewidth=2.0)
    status = ax_scene.text(
        0.02,
        0.97,
        "",
        transform=ax_scene.transAxes,
        va="top",
        family="monospace",
    )
    ax_scene.text(
        0.98,
        0.97,
        r"$B\propto\sin(t-x)$  $\longrightarrow$",
        transform=ax_scene.transAxes,
        ha="right",
        va="top",
    )

    # Right: traces grow as the CPU solver advances.
    traces = []
    if show_graphs:
        (x_trace,) = ax_x.plot(times, positions, color="tab:blue", linewidth=1.7)
        (theta_trace,) = ax_theta.plot(times, angles, color="tab:purple", linewidth=1.7)
        (x_point,) = ax_x.plot(times, positions, "o", color="tab:blue", markersize=4)
        (theta_point,) = ax_theta.plot(times, angles, "o", color="tab:purple", markersize=4)
        traces.extend(((ax_x, x_trace, x_point, positions), (ax_theta, theta_trace, theta_point, angles)))

        (phase_rate_trace,) = ax_phase_rate.plot(
            times, phase_rates, color="tab:red", linewidth=1.7
        )
        (phase_rate_point,) = ax_phase_rate.plot(
            times, phase_rates, "o", color="tab:red", markersize=4
        )
        traces.append((ax_phase_rate, phase_rate_trace, phase_rate_point, phase_rates))

        ax_x.set_title("CPU evolution")
        ax_x.set_ylabel(r"$x(t)$")
        ax_theta.set_ylabel(r"$\theta(t)$")
        ax_phase_rate.set_ylabel(r"$\dot{\phi}=1-v-\omega$")
        ax_x.grid(alpha=0.25)
        ax_theta.grid(alpha=0.25)
        ax_phase_rate.grid(alpha=0.25)

        if show_velocity:
            (velocity_trace,) = ax_velocity.plot(times, velocities, color="tab:green", linewidth=1.7)
            (velocity_point,) = ax_velocity.plot(times, velocities, "o", color="tab:green", markersize=4)
            traces.append((ax_velocity, velocity_trace, velocity_point, velocities))
            ax_velocity.set_ylabel(r"$\dot{x}(t)$")
            ax_velocity.set_xlabel(r"$t$")
            ax_velocity.grid(alpha=0.25)
        else:
            ax_phase_rate.set_xlabel(r"$t$")

    fig.suptitle(rf"$v_0={initial_v0:g},\quad S_0={initial_s0:g}$")

    def rescale_history_axes() -> None:
        t_now = times[-1]
        ax_x.set_xlim(0.0, max(10.0, t_now * 1.05))
        for axis, _trace, _point, values in traces:
            lo = min(values)
            hi = max(values)
            span = hi - lo
            padding = 0.08 * span if span > 1.0e-12 else 0.1
            axis.set_ylim(lo - padding, hi + padding)

    def update(_frame_number: int):
        nonlocal state
        batch_size = FAST_BATCH_SIZE if fast else 1
        for _ in range(batch_size):
            state, sliding = hysteresis.evolve_with_mode(state, DT, STEPS_PER_FRAME, params)
            times.append(float(state[0]))
            positions.append(float(state[1]))
            angles.append(float(state[3]))
            velocities.append(float(state[2]))
            phase_rates.append(1.0 - float(state[2]) - float(state[4]))

        t = times[-1]
        x = positions[-1]
        theta = angles[-1]

        # Keep the ball centered while the wave travels past it.
        left = x - view_half_width
        right = x + view_half_width
        scene_x = np.linspace(left, right, 700)
        ax_scene.set_xlim(left, right)
        wave.set_data(
            scene_x,
            wave_height + wave_amplitude * np.sin(t - scene_x),
        )

        ball.center = (x, alpha)
        ball.set_facecolor("tab:orange" if sliding else "tab:blue")
        radius_line.set_data(
            [x, x + 0.78 * alpha * np.cos(theta)],
            [alpha, alpha + 0.78 * alpha * np.sin(theta)],
        )
        mode = "SLIDING" if sliding else "STICKING"
        status.set_text(f"t = {t:7.3f}\nx = {x:9.4f}\ntheta = {theta:9.4f}\n{mode}")

        update_graphs = not fast or _frame_number % FAST_GRAPH_UPDATE_EVERY == 0
        if update_graphs:
            for _axis, trace, point, values in traces:
                if fast:
                    display_times, display_values = minmax_downsample(times, values, FAST_MAX_PLOT_POINTS)
                else:
                    display_times, display_values = times, values
                trace.set_data(display_times, display_values)
                point.set_data([t], [values[-1]])
        if show_graphs and (not fast or _frame_number % FAST_RESCALE_EVERY == 0):
            rescale_history_axes()

        trace_artists = [artist for _axis, trace, point, _values in traces for artist in (trace, point)]
        return wave, ball, radius_line, status, *trace_artists

    if fast:
        fig.tight_layout()

    # interval=1000/FPS guarantees a requested display rate no higher than 30 FPS.
    animation = FuncAnimation(
        fig,
        update,
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    return fig, animation


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Animate the CPU hysteresis solver.")
    graph_options = parser.add_mutually_exclusive_group()
    graph_options.add_argument("--no-graph", action="store_true", help="hide the real-time history graphs")
    graph_options.add_argument(
        "--show-v",
        action="store_true",
        help="show the center-of-mass velocity graph in addition to x(t) and theta(t)",
    )
    parser.add_argument("--fast", action="store_true", help="reduce real-time rendering overhead")
    parser.add_argument("--v0", type=float, default=DEFAULT_INITIAL_V0, help="initial center-of-mass velocity")
    parser.add_argument(
        "--s0",
        type=float,
        default=DEFAULT_INITIAL_S0,
        help="initial contact-point slip velocity, S0 = v0 + alpha * omega0",
    )
    args = parser.parse_args()

    figure, animation = animate(
        show_graphs=not args.no_graph,
        show_velocity=args.show_v,
        fast=args.fast,
        initial_v0=args.v0,
        initial_s0=args.s0,
    )
    plt.show()
