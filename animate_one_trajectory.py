"""Animate one CPU trajectory of the two-dimensional hysteresis model.

From src2d, build with ``cmake --preset local-cpu`` and
``cmake --build --preset local-cpu-release``, then run with the daily Python.
The defaults use the 1D reference coefficients converted to disk-angular time
for the compiled magnet count; finite curvature still changes the field.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from project_paths import extension_dir


DT_AT_EIGHT_MAGNETS = 1.0e-3
STEPS_PER_FRAME = 1000
FPS = 30
FAST_BATCH_SIZE = 5
FAST_GRAPH_UPDATE_EVERY = 5
FAST_RESCALE_EVERY = 20
FAST_MAX_PLOT_POINTS = 2000

# Radius and angular-speed multipliers at the same selected point as the 1D animation.
SR = 0.428571
SOMEGA = 0.1

# These are the 8-magnet references. The field period rate is N/2 per
# revolution, so the three acceleration coefficients scale as (N/8)**2.
ALPHA = 0.33333333
DELTA = 1.7212357e-4
GAMMA_G = 2.64965696
GAMMA = 7830.34224
GAMMA_M = 176182.704


def scaled_model_parameters(magnet_count: int, *, sr=SR, somega=SOMEGA) -> tuple[float, float, float, float, float]:
    """Keep the same local 1D reference when the compiled magnet count changes."""
    if magnet_count < 2 or magnet_count % 2:
        raise ValueError("magnet count must be positive and even")
    if not np.isfinite(sr) or sr <= 0.0:
        raise ValueError("SR must be finite and positive")
    if not np.isfinite(somega) or somega <= 0.0:
        raise ValueError("SOMEGA must be finite and positive")

    inverse_sr = 1.0 / sr
    inverse_somega_squared = 1.0 / (somega * somega)
    magnet_time_scale_squared = (magnet_count / 8.0) ** 2
    return (
        ALPHA * sr,
        DELTA * inverse_sr,
        GAMMA_G * inverse_somega_squared * magnet_time_scale_squared,
        GAMMA * inverse_somega_squared * magnet_time_scale_squared,
        GAMMA_M * inverse_sr * inverse_sr * inverse_somega_squared * magnet_time_scale_squared,
    )


def timestep_for_magnets(magnet_count: int) -> float:
    """Use the same phase-time step as dt=0.001 with eight magnets."""
    if magnet_count < 2 or magnet_count % 2:
        raise ValueError("magnet count must be positive and even")
    return DT_AT_EIGHT_MAGNETS * 8.0 / magnet_count


def load_hysteresis():
    directory = extension_dir("fp64")
    modules = sorted(directory.glob("hysteresis*.pyd"))
    compatible = [path for path in modules if path.name.startswith(f"hysteresis.cp{sys.version_info.major}{sys.version_info.minor}-")]
    if len(compatible) != 1:
        raise RuntimeError(
            f"Expected one extension for Python {sys.version_info.major}.{sys.version_info.minor} "
            f"in {directory}. Build with CMake using the daily environment."
        )
    path = compatible[0]
    kernels = Path(__file__).resolve().parent / "kernels"
    sources = (
        kernels / "headers" / "2dmodel-hysteresis.hpp",
        kernels / "headers" / "hysteresis_types.hpp",
        kernels / "headers" / "hysteresis_config.hpp",
        kernels / "headers" / "hysteresis_config_field.hpp",
        kernels / "2dmodel-hysteresis.cpp",
        kernels / "bindings" / "pybind.cpp",
    )
    newer = [source.name for source in sources if source.stat().st_mtime > path.stat().st_mtime]
    if newer:
        raise RuntimeError(f"Rebuild the extension; newer sources: {', '.join(newer)}")
    spec = importlib.util.spec_from_file_location("hysteresis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.state_size != 15:
        raise RuntimeError("The extension does not expose the two-dimensional state")
    return module


def make_initial_state(module):
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    state[1] = module.magnet_number / 2  # magnet radius in dimensionless units
    state[5] = 1  # identity attitude quaternion, (w, x, y, z)
    return state


def minmax_downsample(times, values, max_points):
    """Keep each time bucket's extrema when limiting plotted history."""
    count = len(times)
    if count <= max_points:
        return times, values
    values_array = np.asarray(values)
    edges = np.linspace(0, count, max(1, max_points // 2) + 1, dtype=np.int64)
    indices = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if start == stop:
            continue
        bucket = values_array[start:stop]
        indices.extend(sorted({int(start + np.argmin(bucket)), int(start + np.argmax(bucket))}))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return np.asarray(times)[indices], values_array[indices]


def animate(*, dt=None, steps_per_frame=STEPS_PER_FRAME, frames=None, show_magnets=True, fast=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    module = load_hysteresis()
    magnet_count = module.magnet_number
    if dt is None:
        dt = timestep_for_magnets(magnet_count)
    if not np.isfinite(dt) or dt <= 0 or steps_per_frame <= 0:
        raise ValueError("dt and steps_per_frame must be positive")
    params = module.SolverParams(*scaled_model_parameters(magnet_count))
    state = make_initial_state(module)
    sliding = None
    sliding_dir = None
    time = 0.0
    completed_steps = 0
    magnet_radius = magnet_count / 2
    base_angles = 2 * np.pi * np.arange(magnet_count) / magnet_count if show_magnets else None

    fig = plt.figure(figsize=(14, 9), **({} if fast else {"layout": "constrained"}))
    grid = fig.add_gridspec(4, 2, width_ratios=(1.3, 1.0))
    scene = fig.add_subplot(grid[:, 0])
    radial_axis = fig.add_subplot(grid[0, 1])
    angular_axis = fig.add_subplot(grid[1, 1], sharex=radial_axis)
    displacement_axis = fig.add_subplot(grid[2, 1], sharex=radial_axis)
    velocity_axis = fig.add_subplot(grid[3, 1], sharex=radial_axis)

    extent = magnet_radius + 1.0
    scene.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y")
    scene.set_aspect("equal", adjustable="box")
    scene.grid(alpha=0.12)
    disk = Circle((0, 0), magnet_radius + 0.35, facecolor="0.94", edgecolor="0.48", linewidth=1.5, zorder=0)
    scene.add_patch(disk)
    magnets = (scene.scatter([], [], s=140, c="0.48", edgecolors="0.28", linewidths=0.8, zorder=2)
               if show_magnets else None)
    (path_line,) = scene.plot([], [], color="tab:blue", linewidth=1.6, alpha=0.85, zorder=3)
    (ball,) = scene.plot([], [], "o", color="tab:orange", markeredgecolor="white", markersize=12, zorder=4)
    status = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top", family="monospace",
                        fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    scene.set_title(f"Top view · {magnet_count} rotating magnets · {module.field_mode} field")

    times = [0.0]
    radial = [float(np.hypot(state[1], state[2]))]
    polar = [float(np.arctan2(state[2], state[1]))]
    initial_angle = polar[0]
    # Arithmetic mean of all displayed radius samples, including the initial state.
    radius_sum = radial[0]
    mean_radial = [radial[0]]
    equivalent_x = [0.0]
    speeds = [float(np.hypot(state[3], state[4]))]
    tangential_speeds = [float((state[1] * state[4] - state[2] * state[3]) / radial[0])]
    xs = [float(state[1])]
    ys = [float(state[2])]
    (radial_line,) = radial_axis.plot(times, radial, color="tab:blue")
    (polar_line,) = angular_axis.plot(times, polar, color="tab:purple")
    (displacement_line,) = displacement_axis.plot(times, equivalent_x, color="tab:green")
    (speed_line,) = velocity_axis.plot(times, speeds, color="tab:orange", label="COM speed |v|")
    (tangential_line,) = velocity_axis.plot(times, tangential_speeds, color="tab:red", label="signed tangential v")
    radial_axis.set_ylabel("r(t)")
    angular_axis.set_ylabel("φ(t) [rad]")
    displacement_axis.set_ylabel("x_eq = mean(r) Δφ")
    velocity_axis.set(xlabel="time (2D units)", ylabel="velocity (2D units)")
    velocity_axis.legend(loc="upper right", fontsize=8)
    velocity_axis.axhline(0, color="0.5", linewidth=0.7)
    for axis in (radial_axis, angular_axis, displacement_axis, velocity_axis):
        axis.grid(alpha=0.25)
    fig.suptitle(f"Single 2D trajectory · sr={SR:g}, sω={SOMEGA:g}")

    def render(*, update_graphs=True, rescale=True):
        if magnets is not None:
            angle = base_angles + float(state[0])
            magnets.set_offsets(np.column_stack((magnet_radius * np.cos(angle), magnet_radius * np.sin(angle))))
        ball.set_data([xs[-1]], [ys[-1]])
        if fast and len(xs) > FAST_MAX_PLOT_POINTS:
            indices = np.linspace(0, len(xs) - 1, FAST_MAX_PLOT_POINTS, dtype=np.int64)
            path_line.set_data(np.asarray(xs)[indices], np.asarray(ys)[indices])
        else:
            path_line.set_data(xs, ys)
        if update_graphs:
            for line, values in ((radial_line, radial), (polar_line, polar),
                                 (displacement_line, equivalent_x), (speed_line, speeds),
                                 (tangential_line, tangential_speeds)):
                display_times, display_values = (minmax_downsample(times, values, FAST_MAX_PLOT_POINTS)
                                                 if fast else (times, values))
                line.set_data(display_times, display_values)
        if rescale:
            radial_axis.set_xlim(0, max(1.0, time * 1.05))
            for axis, values in ((radial_axis, radial), (angular_axis, polar),
                                 (displacement_axis, equivalent_x), (velocity_axis, speeds + tangential_speeds)):
                lower, upper = min(values), max(values)
                padding = max(0.05, 0.1 * (upper - lower))
                axis.set_ylim(lower - padding, upper + padding)
        status.set_text(
            f"t = {time:.3f}  (1D t = {magnet_radius * time:.3f})\n"
            f"r = {radial[-1]:.6f}  mean(r) = {mean_radial[-1]:.6f}\n"
            f"φ = {polar[-1]:.6f}  x_eq = {equivalent_x[-1]:.6f}\n"
            f"v = ({float(state[3]):.6f}, {float(state[4]):.6f})\n"
            f"|v| = {speeds[-1]:.6f}  v_tan = {tangential_speeds[-1]:.6f}\n"
            f"v_tan / {magnet_radius:g} (1D units) = {tangential_speeds[-1] / magnet_radius:.6f}\n"
            f"{'sliding' if sliding else 'rolling'}"
        )
        artists = (ball, path_line, radial_line, polar_line, displacement_line,
                   speed_line, tangential_line, status)
        return (magnets, *artists) if magnets is not None else artists

    def update(_frame):
        nonlocal state, sliding, sliding_dir, time, completed_steps, radius_sum
        for _ in range(FAST_BATCH_SIZE if fast else 1):
            state, sliding, sliding_dir = module.evolve_trajectory(
                state, dt, steps_per_frame, params, sliding, sliding_dir, completed_steps
            )
            completed_steps += steps_per_frame
            time += dt * steps_per_frame
            xs.append(float(state[1]))
            ys.append(float(state[2]))
            times.append(time)
            radial.append(float(np.hypot(state[1], state[2])))
            angle = float(np.arctan2(state[2], state[1]))
            polar.append(polar[-1] + (angle - polar[-1] + np.pi) % (2 * np.pi) - np.pi)
            radius_sum += radial[-1]
            mean_radial.append(radius_sum / len(radial))
            equivalent_x.append(mean_radial[-1] * (polar[-1] - initial_angle))
            speeds.append(float(np.hypot(state[3], state[4])))
            tangential_speeds.append(
                float((state[1] * state[4] - state[2] * state[3]) / radial[-1])
                if radial[-1] > 0.0 else float("nan")
            )
        return render(update_graphs=not fast or _frame % FAST_GRAPH_UPDATE_EVERY == 0,
                      rescale=not fast or _frame % FAST_RESCALE_EVERY == 0)

    render()
    if fast:
        fig.tight_layout()
    animation = FuncAnimation(
        fig, update, init_func=render, frames=frames, interval=1000 / FPS,
        blit=False, cache_frame_data=False,
    )
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=None,
                        help="2D time step (default: 0.001 * 8 / compiled magnet count)")
    parser.add_argument("--steps-per-frame", type=int, default=STEPS_PER_FRAME)
    parser.add_argument("--frames", type=int, help="stop after this many display frames")
    parser.add_argument("--hide-magnets", action="store_true", help="omit magnet markers from the animation")
    parser.add_argument("--fast", action="store_true", help="batch solver steps and reduce plotting overhead")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    import matplotlib.pyplot as plt

    figure, animation = animate(
        dt=args.dt, steps_per_frame=args.steps_per_frame,
        frames=range(args.frames) if args.frames is not None else None,
        show_magnets=not args.hide_magnets,
        fast=args.fast,
    )
    plt.show()


if __name__ == "__main__":
    main()
