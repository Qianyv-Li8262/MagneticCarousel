# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Animate one CPU trajectory of the two-dimensional hysteresis model.

From src2d, build with ``cmake --preset local-cpu`` and
``cmake --build --preset local-cpu-release``, then run with the daily Python.
The defaults use the 1D reference coefficients converted to disk-angular time
for the compiled magnet count; finite curvature still changes the field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from project_paths import extension_dir


DT_AT_EIGHT_MAGNETS = 1.0e-3
STEPS_PER_FRAME = 100
FPS = 30
FAST_BATCH_SIZE = 5
FAST_GRAPH_UPDATE_EVERY = 5
FAST_RESCALE_EVERY = 20
FAST_MAX_PLOT_POINTS = 2000

# Radius and angular-speed multipliers at the same selected point as the 1D animation.
SR = 1
SOMEGA = 2.0

# These are the 8-magnet references. The field period rate is N/2 per
# revolution, so the three acceleration coefficients scale as (N/8)**2.
ALPHA = 0.33333333
DELTA = 1.7212357e-4
GAMMA_G = 2.64965696
GAMMA = 7830.34224
GAMMA_M = 176182.704


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_build_config(*, kahan: bool) -> dict:
    directory = extension_dir("fp32-kahan" if kahan else "fp64")
    path = directory / "build_config.json"
    if not path.is_file():
        raise RuntimeError(f"Build manifest missing: {path}. Rebuild the CPU extension with CMake.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("precision") != ("fp32-kahan" if kahan else "fp64"):
        raise RuntimeError(f"Invalid or mismatched build manifest: {path}")
    binary_name = config.get("binary")
    if not isinstance(binary_name, str) or Path(binary_name).name != binary_name:
        raise RuntimeError(f"Invalid extension name in {path}")
    binary = directory / binary_name
    if not binary.is_file() or file_sha256(binary) != config.get("binary_sha256"):
        raise RuntimeError(f"The CPU extension does not match {path}; rebuild it.")
    root = Path(__file__).resolve().parent
    for relative, recorded_hash in config.get("source_sha256", {}).items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != recorded_hash:
            raise RuntimeError(f"Source changed since the build: {source}. Rebuild the CPU extension.")
    return config


def print_run_config(config: dict, *, dt: float | None, steps_per_frame: int,
                     frames: int | None, pre: float, fast: bool, show_magnets: bool,
                     show_work=False, sr=SR, somega=SOMEGA, v0=0.0, s0=0.0, phi0=0.0) -> None:
    def row(label: str, value: object) -> None:
        print(f"  {label:<28} {value}")

    def section(title: str) -> None:
        print(f"\n  [{title}]")

    def display_literal(value: str) -> str:
        return value.removeprefix("REAL_LITERAL(").removesuffix(")") if value.startswith("REAL_LITERAL(") else value

    count = config["magnet_number"]
    effective_dt = timestep_for_magnets(count) if dt is None else dt
    parameters = scaled_model_parameters(count, sr=sr, somega=somega)
    print("\n=== 2D CPU animation | effective configuration ===")
    section("Build")
    cache_info = (str(config["lut_cache_mode"]) if config["field_mode"] == "lut" else "inactive (exact field)")
    row("Precision / field", f"{config['precision']} / {config['field_mode']}")
    row("Magnets / radius", f"{count} / {count / 2:g}")
    row("LUT cache", cache_info)
    section("C++ field parameters")
    for label, value in config["field_parameters"].items():
        row(label, display_literal(value))
    section("C++ model parameters")
    for label, value in config["model_parameters"].items():
        row(label, display_literal(value))
    section("Trajectory")
    row("s_r / s_omega", f"{sr:g} / {somega:g}")
    row("Initial v0 / S0 / phi0", f"{v0:.12g} / {s0:.12g} / {phi0:.12g}")
    row("Magnetic mechanical power", show_work)
    for label, value in zip(("alpha", "delta", "gamma_g", "gamma", "gamma_m"), parameters):
        row(label, f"{value:.9g}")
    row("dt", f"{effective_dt:.12g}")
    row("Steps per frame", steps_per_frame)
    row("Pre-evolution", f"{pre:g}")
    row("Frames", frames if frames is not None else "unlimited")
    row("Fast / magnets", f"{fast} / {show_magnets}")
    section("Files")
    row("Manifest", extension_dir(config["precision"]) / "build_config.json")
    row("Extension", extension_dir(config["precision"]) / config["binary"])
    print(flush=True)


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


def load_hysteresis(*, kahan=False):
    kind = "fp32-kahan" if kahan else "fp64"
    directory = extension_dir(kind)
    modules = sorted(directory.glob("hysteresis*.pyd"))
    compatible = [path for path in modules if path.name.startswith(f"hysteresis.cp{sys.version_info.major}{sys.version_info.minor}-")]
    if len(compatible) != 1:
        raise RuntimeError(
            f"Expected one extension for Python {sys.version_info.major}.{sys.version_info.minor} "
            f"in {directory}. Build the {'Kahan FP32' if kahan else 'FP64'} CMake target "
            f"hysteresis_{'fp32_kahan' if kahan else 'fp64'} using the daily environment."
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
    expected_dtype = np.dtype(np.float32 if kahan else np.float64)
    if np.dtype(module.real_dtype) != expected_dtype:
        raise RuntimeError(f"Expected {kind} extension to use {expected_dtype}, got {module.real_dtype}")
    return module


def make_initial_state(module, *, alpha=None, v0=0.0, s0=0.0, phi0=0.0):
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    radius = module.magnet_number / 2
    radial_direction = np.array([np.cos(phi0), np.sin(phi0)])
    state[1:3] = radius * radial_direction
    state[3:5] = radius * v0 * np.array([-radial_direction[1], radial_direction[0]])
    if alpha is not None:
        state[9:11] = radius * (s0 - v0) / alpha * radial_direction
    elif v0 != 0 or s0 != 0:
        raise ValueError("alpha is required for a nonzero initial velocity or slip")
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


def animate(*, dt=None, steps_per_frame=STEPS_PER_FRAME, frames=None, show_magnets=True, fast=False, pre=0.0,
            kahan=False, expected_config=None, show_work=False,
            sr=SR, somega=SOMEGA, v0=0.0, s0=0.0, phi0=0.0,
            path_max_points=FAST_MAX_PLOT_POINTS):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    module = load_hysteresis(kahan=kahan)
    if expected_config is not None:
        if module.magnet_number != expected_config["magnet_number"] or module.field_mode != (
            "exact" if expected_config["field_mode"] == "exact" else "fourier-spline"
        ):
            raise RuntimeError("Loaded CPU extension disagrees with its build manifest")
    magnet_count = module.magnet_number
    if dt is None:
        dt = timestep_for_magnets(magnet_count)
    if not np.isfinite(dt) or dt <= 0 or steps_per_frame <= 0:
        raise ValueError("dt and steps_per_frame must be positive")
    if not np.isfinite(pre) or pre < 0:
        raise ValueError("pre must be finite and non-negative")
    if path_max_points < 0:
        raise ValueError("path_max_points must be non-negative (0 keeps every point)")
    if not all(np.isfinite(value) for value in (v0, s0, phi0)):
        raise ValueError("initial values must be finite")
    params = module.SolverParams(*scaled_model_parameters(magnet_count, sr=sr, somega=somega))
    state = make_initial_state(module, alpha=params.alpha, v0=v0, s0=s0, phi0=phi0)
    sliding = None
    sliding_dir = None
    pre_step_count = pre / dt
    if not np.isfinite(pre_step_count) or pre_step_count > np.iinfo(np.int32).max + 0.5:
        raise ValueError("pre-evolution exceeds the C++ solver's maximum steps per call")
    pre_steps = int(round(pre_step_count))
    if pre_steps:
        print(f"Pre-evolving in C++ for {pre_steps} steps (t={pre_steps * dt:g})...")
        state, sliding, sliding_dir = module.evolve_trajectory(
            state, dt, pre_steps, params, sliding, sliding_dir, 0
        )
    time = pre_steps * dt
    completed_steps = pre_steps
    magnet_radius = magnet_count / 2
    base_angles = 2 * np.pi * np.arange(magnet_count) / magnet_count if show_magnets else None

    fig = plt.figure(figsize=(14, 11 if show_work else 9), **({} if fast else {"layout": "constrained"}))
    grid = fig.add_gridspec(5 if show_work else 4, 2, width_ratios=(1.3, 1.0))
    scene = fig.add_subplot(grid[:, 0])
    radial_axis = fig.add_subplot(grid[0, 1])
    angular_axis = fig.add_subplot(grid[1, 1], sharex=radial_axis)
    displacement_axis = fig.add_subplot(grid[2, 1], sharex=radial_axis)
    velocity_axis = fig.add_subplot(grid[3, 1], sharex=radial_axis)
    work_axis = fig.add_subplot(grid[4, 1], sharex=radial_axis) if show_work else None

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

    times = [time]
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
    power_values = []
    power_lines = []
    if show_work:
        power_values = [[float(value)] for value in module.magnetic_power(state, params)]
        for label, color, values in zip(("F_mag dot v", "torque_mag dot omega", "total magnetic power"),
                                        ("tab:blue", "tab:purple", "black"), power_values):
            (line,) = work_axis.plot(times, values, color=color, label=label, linewidth=1.0)
            power_lines.append(line)
        work_axis.set(xlabel="time (2D units)", ylabel="magnetic power / mass\n(2D units)")
        work_axis.axhline(0, color="0.5", linewidth=0.7)
        work_axis.legend(loc="upper right", fontsize=8)
        work_axis.grid(alpha=0.25)
    for axis in (radial_axis, angular_axis, displacement_axis, velocity_axis):
        axis.grid(alpha=0.25)
    fig.suptitle(f"Single 2D trajectory · sr={sr:g}, sω={somega:g}")

    def render(*, update_graphs=True, rescale=True):
        if magnets is not None:
            angle = base_angles + float(state[0])
            magnets.set_offsets(np.column_stack((magnet_radius * np.cos(angle), magnet_radius * np.sin(angle))))
        ball.set_data([xs[-1]], [ys[-1]])
        if fast and path_max_points and len(xs) > path_max_points:
            # Preserve the initial excursion at its original sampling density.
            early_count = min(1000, path_max_points // 2)
            indices = np.r_[np.arange(early_count),
                            np.linspace(early_count, len(xs) - 1,
                                        path_max_points - early_count, dtype=np.int64)]
            path_line.set_data(np.asarray(xs)[indices], np.asarray(ys)[indices])
        else:
            path_line.set_data(xs, ys)
        if update_graphs:
            graph_series = [(radial_line, radial), (polar_line, polar),
                            (displacement_line, equivalent_x), (speed_line, speeds),
                            (tangential_line, tangential_speeds)]
            graph_series.extend(zip(power_lines, power_values))
            for line, values in graph_series:
                display_times, display_values = (minmax_downsample(times, values, FAST_MAX_PLOT_POINTS)
                                                 if fast else (times, values))
                line.set_data(display_times, display_values)
        if rescale:
            radial_axis.set_xlim(0, max(1.0, time * 1.05))
            limits = [(radial_axis, radial), (angular_axis, polar),
                      (displacement_axis, equivalent_x), (velocity_axis, speeds + tangential_speeds)]
            if show_work:
                limits.append((work_axis, [value for series in power_values for value in series]))
            for axis, values in limits:
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
                   speed_line, tangential_line, *power_lines, status)
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
            if show_work:
                for values, value in zip(power_values, module.magnetic_power(state, params)):
                    values.append(float(value))
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
    parser.add_argument("--pre", type=float, default=0.0,
                        help="pre-evolve once in C++ to this 2D time, then start the animation (default: 0)")
    parser.add_argument("--kahan", action="store_true",
                        help="force-load the Kahan-compensated FP32 extension")
    parser.add_argument("--frames", type=int, help="stop after this many display frames")
    parser.add_argument("--hide-magnets", action="store_true", help="omit magnet markers from the animation")
    parser.add_argument("--fast", action="store_true", help="batch solver steps and reduce plotting overhead")
    parser.add_argument("--show-work", action="store_true", help="plot magnetic mechanical power F dot v + torque dot omega (per mass, 2D units)")
    parser.add_argument("--path-max-points", type=int, default=FAST_MAX_PLOT_POINTS,
                        help="maximum fast-mode trajectory points; 0 keeps all points; initial points are preserved")
    parser.add_argument("--sr", type=float, default=SR)
    parser.add_argument("--somega", type=float, default=SOMEGA)
    parser.add_argument("--v0", type=float, default=0.0, help="initial signed tangential speed in 1D units")
    parser.add_argument("--s0", type=float, default=0.0, help="initial signed tangential slip in 1D units; omega0=(s0-v0)/alpha")
    parser.add_argument("--phi0", type=float, default=0.0, help="initial orbital angle in radians; disk angle starts at zero")
    parser.add_argument("--dry-run", action="store_true", help="print and validate the build and runtime configuration without starting the animation")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args()
    if args.path_max_points < 0:
        parser.error("--path-max-points must be non-negative")
    if not all(np.isfinite(value) for value in (args.sr, args.somega, args.v0, args.s0, args.phi0)):
        parser.error("parameters and initial values must be finite")
    if args.sr <= 0 or args.somega <= 0:
        parser.error("--sr and --somega must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if not np.isfinite(args.pre) or args.pre < 0:
        parser.error("--pre must be finite and non-negative")
    if args.steps_per_frame <= 0:
        parser.error("--steps-per-frame must be positive")
    if args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0):
        parser.error("--dt must be positive and finite")
    config = read_build_config(kahan=args.kahan)
    print_run_config(config, dt=args.dt, steps_per_frame=args.steps_per_frame,
                     frames=args.frames, pre=args.pre, fast=args.fast,
                     show_magnets=not args.hide_magnets, show_work=args.show_work,
                     sr=args.sr, somega=args.somega, v0=args.v0, s0=args.s0, phi0=args.phi0)
    if args.dry_run:
        return
    if not args.yes:
        try:
            answer = input("Run this CPU animation? Type yes to continue: ").strip()
        except EOFError as exc:
            raise SystemExit("No confirmation received; animation cancelled. Use --yes for non-interactive runs.") from exc
        if answer != "yes":
            print("Animation cancelled.")
            return
    import matplotlib.pyplot as plt

    figure, animation = animate(
        dt=args.dt, steps_per_frame=args.steps_per_frame,
        frames=range(args.frames) if args.frames is not None else None,
        show_magnets=not args.hide_magnets,
        fast=args.fast,
        pre=args.pre,
        kahan=args.kahan,
        expected_config=config,
        show_work=args.show_work, sr=args.sr, somega=args.somega,
        v0=args.v0, s0=args.s0, phi0=args.phi0,
        path_max_points=args.path_max_points,
    )
    plt.show()


if __name__ == "__main__":
    main()
