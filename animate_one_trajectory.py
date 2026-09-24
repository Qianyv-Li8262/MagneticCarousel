"""Animate one CPU trajectory of the two-dimensional hysteresis model.

From src2d, build with ``cmake --preset local-cpu`` and
``cmake --build --preset local-cpu-release``, then run with the daily Python.
The defaults use the 1D reference coefficients converted to disk-angular time
for eight alternating magnets; finite curvature still changes the field.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from project_paths import extension_dir


DT = 1.0e-3
STEPS_PER_FRAME = 1000
FPS = 30

# Radius and angular-speed multipliers at the same selected point as the 1D animation.
SR = 0.428571
SOMEGA = 0.1

# Eight alternating magnets give four field periods per disk revolution.
# Converting from 1D phase time to disk-angular time multiplies the three
# acceleration coefficients by 4**2. These references assume magnet_number=8.
ALPHA = 0.33333333
DELTA = 1.7212357e-4
GAMMA_G = 2.64965696
GAMMA = 7830.34224
GAMMA_M = 176182.704


def scaled_model_parameters() -> tuple[float, float, float, float, float]:
    """Scale ball radius and disk angular speed as in the 1D animation."""
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


def animate(*, dt=DT, steps_per_frame=STEPS_PER_FRAME, frames=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    if not np.isfinite(dt) or dt <= 0 or steps_per_frame <= 0:
        raise ValueError("dt and steps_per_frame must be positive")

    module = load_hysteresis()
    params = module.SolverParams(*scaled_model_parameters())
    state = make_initial_state(module)
    sliding = None
    sliding_dir = None
    time = 0.0
    completed_steps = 0
    magnet_count = module.magnet_number
    magnet_radius = magnet_count / 2
    base_angles = 2 * np.pi * np.arange(magnet_count) / magnet_count

    fig = plt.figure(figsize=(14, 9), layout="constrained")
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
    magnets = scene.scatter([], [], s=140, c="0.48", edgecolors="0.28", linewidths=0.8, zorder=2)
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

    def render():
        angle = base_angles + float(state[0])
        magnets.set_offsets(np.column_stack((magnet_radius * np.cos(angle), magnet_radius * np.sin(angle))))
        ball.set_data([xs[-1]], [ys[-1]])
        path_line.set_data(xs, ys)
        radial_line.set_data(times, radial)
        polar_line.set_data(times, polar)
        displacement_line.set_data(times, equivalent_x)
        speed_line.set_data(times, speeds)
        tangential_line.set_data(times, tangential_speeds)
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
        return (magnets, ball, path_line, radial_line, polar_line, displacement_line,
                speed_line, tangential_line, status)

    def update(_frame):
        nonlocal state, sliding, sliding_dir, time, completed_steps, radius_sum
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
        return render()

    render()
    animation = FuncAnimation(
        fig, update, init_func=render, frames=frames, interval=1000 / FPS,
        blit=False, cache_frame_data=False,
    )
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps-per-frame", type=int, default=STEPS_PER_FRAME)
    parser.add_argument("--frames", type=int, help="stop after this many display frames")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    import matplotlib.pyplot as plt

    figure, animation = animate(
        dt=args.dt, steps_per_frame=args.steps_per_frame,
        frames=range(args.frames) if args.frames is not None else None,
    )
    plt.show()


if __name__ == "__main__":
    main()
