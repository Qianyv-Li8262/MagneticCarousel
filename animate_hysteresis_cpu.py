"""Simple real-time animation for the CPU hysteresis solver.

Run from this directory with::

    python src/animate_hysteresis_cpu.py

The compiled ``hysteresis`` extension must be next to this file.  Rebuild it
with ``python src/build.py`` if it is missing.
"""

from __future__ import annotations

import argparse

import numpy as np

import hysteresis


# Simulation settings.  Each displayed frame advances exactly 1000 CPU steps.
DT = 1.0e-3
STEPS_PER_FRAME = 5000
FPS = 30

# Model parameters.
ALPHA = 0.33333333
DELTA = 1.7212357e-4
GAMMA_G = 0.16560356
GAMMA = 489.39639
GAMMA_M = 11011.419

# State layout: [t, x, x_dot, theta, theta_dot, phase, q_x, q_y].
# In particular, the initial hysteresis vector q is the zero vector.
INITIAL_STATE = np.zeros(8, dtype=np.float32)


def animate(*, show_graphs: bool = True, show_velocity: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    if not show_graphs and show_velocity:
        raise ValueError("show_velocity requires show_graphs")

    params = hysteresis.SolverParams(
        ALPHA,
        DELTA,
        GAMMA_G,
        GAMMA,
        GAMMA_M,
    )
    state = INITIAL_STATE.copy()

    times = [float(state[0])]
    positions = [float(state[1])]
    angles = [float(state[3])]
    velocities = [float(state[2])]

    if show_graphs:
        graph_rows = 3 if show_velocity else 2
        figure_height = 6.8 if show_velocity else 5.2
        fig = plt.figure(figsize=(12.5, figure_height), layout="constrained")
        grid = fig.add_gridspec(graph_rows, 2, width_ratios=(1.7, 1.0))
        ax_scene = fig.add_subplot(grid[:, 0])
        ax_x = fig.add_subplot(grid[0, 1])
        ax_theta = fig.add_subplot(grid[1, 1], sharex=ax_x)
        ax_velocity = fig.add_subplot(grid[2, 1], sharex=ax_x) if show_velocity else None
    else:
        fig, ax_scene = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
        ax_x = None
        ax_theta = None
        ax_velocity = None

    # Left: a ball on the ground with a right-moving sinusoidal wave above it.
    view_half_width = 2.0 * np.pi
    wave_height = 1.25 * ALPHA
    wave_amplitude = 0.18 * ALPHA
    scene_x = np.linspace(-view_half_width, view_half_width, 700)

    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_ylim(-0.18 * ALPHA, 1.65 * ALPHA)
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
    ball = Circle((0.0, ALPHA), ALPHA, edgecolor="0.15", linewidth=1.6)
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

        ax_x.set_title("CPU evolution")
        ax_x.set_ylabel(r"$x(t)$")
        ax_theta.set_ylabel(r"$\theta(t)$")
        ax_x.grid(alpha=0.25)
        ax_theta.grid(alpha=0.25)

        if show_velocity:
            (velocity_trace,) = ax_velocity.plot(times, velocities, color="tab:green", linewidth=1.7)
            (velocity_point,) = ax_velocity.plot(times, velocities, "o", color="tab:green", markersize=4)
            traces.append((ax_velocity, velocity_trace, velocity_point, velocities))
            ax_velocity.set_ylabel(r"$\dot{x}(t)$")
            ax_velocity.set_xlabel(r"$t$")
            ax_velocity.grid(alpha=0.25)
        else:
            ax_theta.set_xlabel(r"$t$")

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
        state, sliding = hysteresis.evolve_with_mode(state, DT, STEPS_PER_FRAME, params)

        t = float(state[0])
        x = float(state[1])
        theta = float(state[3])

        times.append(t)
        positions.append(x)
        angles.append(theta)
        velocities.append(float(state[2]))

        # Keep the ball centered while the wave travels past it.
        left = x - view_half_width
        right = x + view_half_width
        scene_x = np.linspace(left, right, 700)
        ax_scene.set_xlim(left, right)
        wave.set_data(
            scene_x,
            wave_height + wave_amplitude * np.sin(t - scene_x),
        )

        ball.center = (x, ALPHA)
        ball.set_facecolor("tab:orange" if sliding else "tab:blue")
        radius_line.set_data(
            [x, x + 0.78 * ALPHA * np.cos(theta)],
            [ALPHA, ALPHA + 0.78 * ALPHA * np.sin(theta)],
        )
        mode = "SLIDING" if sliding else "STICKING"
        status.set_text(f"t = {t:7.3f}\nx = {x:9.4f}\ntheta = {theta:9.4f}\n{mode}")

        for _axis, trace, point, values in traces:
            trace.set_data(times, values)
            point.set_data([t], [values[-1]])
        if show_graphs:
            rescale_history_axes()

        trace_artists = [artist for _axis, trace, point, _values in traces for artist in (trace, point)]
        return wave, ball, radius_line, status, *trace_artists

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
    args = parser.parse_args()

    figure, animation = animate(show_graphs=not args.no_graph, show_velocity=args.show_v)
    plt.show()
