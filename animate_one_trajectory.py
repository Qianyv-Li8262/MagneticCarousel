# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Animate one CPU trajectory of the two-dimensional hysteresis model.

From src2d, build with ``cmake --preset local-cpu`` and
``cmake --build --preset local-cpu-release``, then run with the daily Python.
The launcher defaults use a 12 mm steel ball and the video-measured 200 rpm.
Hold the ball for two magnet-disk turns, then release at v0=-0.075, phase 0.58
with the compiled tilt and rolling-resistance switches. Optional air blowing uses
--blow-start TIME --blow-duration TIME [--blow-force MN] after release.
Blowing times use the same 2D time units as the graph; force remains in mN.
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
from build_metadata import print_build_metadata


DT_AT_EIGHT_MAGNETS = 5.0e-4
STEPS_PER_FRAME = 3000
FPS = 30
FAST_BATCH_SIZE = 5
FAST_GRAPH_UPDATE_EVERY = 5
FAST_RESCALE_EVERY = 20
FAST_MAX_PLOT_POINTS = 2000

# Multipliers relative to the 12 mm ball and 100 rpm apparatus reference.
SR = 1.66666666666
SOMEGA = 2.0
ANIMATION_SOMEGA = 2.0
ANIMATION_V0 = -0.00
MAGNETIZE_TURNS = 0
DISK_PHASE = 0.58
DEFAULT_BLOW_FORCE_MN = 0.7



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
    if config.get("field_mode") == "lut" and "lut" in config:
        lut = Path(config["lut"])
        if not lut.is_file() or file_sha256(lut) != config.get("lut_sha256"):
            raise RuntimeError("CPU LUT differs from its build manifest; rebuild the CMake target")
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
                     show_work=False, sr=SR, somega=ANIMATION_SOMEGA, v0=ANIMATION_V0, s0=0.0, phi0=0.0, oz=0.0,
                     magnetize_turns=MAGNETIZE_TURNS, disk_phase=DISK_PHASE, fps=FPS,
                     blow_start=None, blow_duration=None, blow_force=DEFAULT_BLOW_FORCE_MN,
                     spin_resets=()) -> None:
    def row(label: str, value: object) -> None:
        print(f"  {label:<28} {value}")

    def section(title: str) -> None:
        print(f"\n  [{title}]")

    def display_literal(value: str) -> str:
        return value.removeprefix("REAL_LITERAL(").removesuffix(")") if value.startswith("REAL_LITERAL(") else value

    count = config["magnet_number"]
    effective_dt = timestep_for_magnets(count) if dt is None else dt
    runtime_module = load_hysteresis(kahan=config["precision"] == "fp32-kahan")
    parameters = runtime_module.SolverParams(sr=sr, somega=somega)
    print("\n=== 2D CPU animation | effective configuration ===")
    print_build_metadata(config)
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
    row("Initial omega_z (2D)", f"{oz:.12g}")
    row("Timed omega_z resets (2D)", list(spin_resets) or "disabled")
    row("Held magnetization turns", magnetize_turns)
    row("Disk phase at release", disk_phase)
    row("Magnetic mechanical power", show_work)
    row("Blow start / duration (2D)", "disabled" if blow_start is None else f"{blow_start:g} / {blow_duration:g}")
    row("Blow force (mN)", blow_force)
    for label, value in zip(("alpha", "delta", "gamma_g", "gamma", "gamma_m", "eddy_current_scale"),
                            (getattr(parameters, key) for key in ("alpha", "delta", "gamma_g", "gamma", "gamma_m", "eddy_current_scale"))):
        row(label, f"{value:.9g}")
    row("dt", f"{effective_dt:.12g}")
    row("Steps per frame", steps_per_frame)
    row("Target display FPS", fps)
    row("Pre-evolution", f"{pre:g}")
    row("Frames", frames if frames is not None else "unlimited")
    row("Fast / magnets", f"{fast} / {show_magnets}")
    section("Files")
    row("Manifest", extension_dir(config["precision"]) / "build_config.json")
    row("Extension", extension_dir(config["precision"]) / config["binary"])
    print(flush=True)


def timestep_for_magnets(magnet_count: int) -> float:
    """Use the same phase-time step as dt=0.0005 with eight magnets."""
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
        kernels / "headers" / "hysteresis_physics.hpp",
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
    expected_size = 15 + (3 * module.eddy_mode_count if module.eddy_currents_enabled else 0)
    if module.state_size != expected_size:
        raise RuntimeError("The extension does not expose the two-dimensional state")
    expected_dtype = np.dtype(np.float32 if kahan else np.float64)
    if np.dtype(module.real_dtype) != expected_dtype:
        raise RuntimeError(f"Expected {kind} extension to use {expected_dtype}, got {module.real_dtype}")
    return module


def make_initial_state(module, *, alpha=None, v0=0.0, s0=0.0, phi0=0.0, oz=0.0):
    if not np.isfinite(oz):
        raise ValueError("initial omega_z must be finite")
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    radius = module.magnet_number / 2
    radial_direction = np.array([np.cos(phi0), np.sin(phi0)])
    state[1:3] = radius * radial_direction
    state[3:5] = radius * v0 * np.array([-radial_direction[1], radial_direction[0]])
    if alpha is not None:
        state[9:11] = radius * (s0 - v0) / alpha * radial_direction
    elif v0 != 0 or s0 != 0:
        raise ValueError("alpha is required for a nonzero initial velocity or slip")
    state[11] = oz
    state[5] = 1  # identity attitude quaternion, (w, x, y, z)
    return state


def prepare_initial_state(module, params, dt, *, v0=0.0, s0=0.0, phi0=0.0, oz=0.0,
                          magnetize_turns=MAGNETIZE_TURNS, disk_phase=DISK_PHASE):
    """Prepare magnetic memory at the requested position, then set release velocities."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(magnetize_turns) or magnetize_turns < 0 or not np.isfinite(disk_phase):
        raise ValueError("magnetization turns must be non-negative and disk phase finite")
    step_count = 2 * np.pi * magnetize_turns / dt
    if not np.isfinite(step_count) or step_count > np.iinfo(np.int32).max:
        raise ValueError("held magnetization exceeds the C++ step limit")
    steps = int(round(step_count))
    state = make_initial_state(module, alpha=params.alpha, v0=v0, s0=s0, phi0=phi0, oz=oz)
    speed, spin = state[3:5].copy(), state[9:12].copy()
    state[0] = disk_phase - steps * dt
    if steps:
        state = module.magnetize_held_ball(state, dt, steps, params)
    state[0] = disk_phase
    state[3:5], state[9:12] = speed, spin
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


def validate_blow(blow_start, blow_duration, blow_force):
    if (blow_start is None) != (blow_duration is None):
        raise ValueError("--blow-start and --blow-duration must be supplied together")
    if not np.isfinite(blow_force) or blow_force < 0:
        raise ValueError("--blow-force must be finite and non-negative (mN)")
    if blow_start is not None:
        if not np.isfinite(blow_start) or blow_start < 0:
            raise ValueError("--blow-start must be finite and non-negative (2D time units)")
        if not np.isfinite(blow_duration) or blow_duration <= 0:
            raise ValueError("--blow-duration must be positive and finite (2D time units)")
        if not np.isfinite(blow_start + blow_duration):
            raise ValueError("blowing end time must be finite")


def parse_spin_reset(value):
    try:
        time, oz = map(float, value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected TIME,OMEGA_Z, for example 2000,-5") from None
    if not np.isfinite(time) or time < 0 or not np.isfinite(oz):
        raise argparse.ArgumentTypeError("reset time must be finite and non-negative; omega_z must be finite")
    return time, oz


class BlowSchedule:
    """Split solver chunks at 2D-time force switches, including --pre."""

    def __init__(self, module, params, dt, *, sr, somega, start=None, duration=None,
                 force_mn=DEFAULT_BLOW_FORCE_MN, spin_resets=()):
        validate_blow(start, duration, force_mn)
        self.module, self.params, self.dt = module, params, dt
        units = module.physical_units(sr=sr, somega=somega)
        self.omega = units["disk_omega_rad_s"]
        self.start = start
        self.end = None if start is None else start + duration
        self.acceleration = module.push_acceleration(force_mn, sr, somega)
        self.elapsed = 0.0
        self.steps = 0
        self.spin_resets = sorted((parse_spin_reset(f"{t},{oz}") for t, oz in spin_resets),
                                  key=lambda event: event[0])
        self.next_reset = 0
        self.reset_samples = []
        if not hasattr(params, "reverse_push_acceleration"):
            raise RuntimeError("Rebuild the CPU extension to expose reverse_push_acceleration")
        params.reverse_push_acceleration = 0.0

    def active(self, time):
        return self.start is not None and self.start <= time < self.end

    def apply_resets(self, state):
        while self.next_reset < len(self.spin_resets):
            reset_time, oz = self.spin_resets[self.next_reset]
            if reset_time > self.elapsed:
                break
            state = state.copy()
            self.reset_samples.append((reset_time, float(state[11]), oz))
            state[11] = oz
            self.next_reset += 1
        return state

    def evolve(self, state, steps, sliding=None, sliding_dir=None):
        state = self.apply_resets(state)
        target = self.elapsed + steps * self.dt
        boundaries = sorted({value for value in (self.start, self.end,
                            *(t for t, _ in self.spin_resets[self.next_reset:]))
                            if value is not None and self.elapsed < value < target})
        for stop in [*boundaries, target]:
            self.params.reverse_push_acceleration = self.acceleration if self.active(self.elapsed) else 0.0
            span = stop - self.elapsed
            nearest = round(span / self.dt)
            tolerance = 8 * np.finfo(float).eps * max(abs(stop), abs(self.elapsed), self.dt)
            if abs(span - nearest * self.dt) <= tolerance:
                count, remainder = nearest, 0.0
            else:
                count = int(np.floor(span / self.dt))
                remainder = span - count * self.dt
            for step_dt, step_count in ((self.dt, count), (remainder, int(remainder > 0))):
                if step_count:
                    state, sliding, sliding_dir = self.module.evolve_trajectory(
                        state, step_dt, step_count, self.params, sliding, sliding_dir, self.steps)
                    self.steps += step_count
            self.elapsed = stop
            state = self.apply_resets(state)
        self.params.reverse_push_acceleration = self.acceleration if self.active(target) else 0.0
        return state, sliding, sliding_dir


def animate(*, dt=None, steps_per_frame=STEPS_PER_FRAME, frames=None, show_magnets=True, fast=False, pre=0.0,
            kahan=False, expected_config=None, show_work=False,
            sr=SR, somega=ANIMATION_SOMEGA, v0=ANIMATION_V0, s0=0.0, phi0=0.0, oz=0.0,
            magnetize_turns=MAGNETIZE_TURNS, disk_phase=DISK_PHASE,
            path_max_points=FAST_MAX_PLOT_POINTS, fps=FPS,
            blow_start=None, blow_duration=None, blow_force=DEFAULT_BLOW_FORCE_MN,
            sample_callback=None, spin_resets=()):
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
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(pre) or pre < 0:
        raise ValueError("pre must be finite and non-negative")
    if path_max_points < 0:
        raise ValueError("path_max_points must be non-negative (0 keeps every point)")
    if not all(np.isfinite(value) for value in (v0, s0, phi0, oz)):
        raise ValueError("initial values must be finite")
    params = module.SolverParams(sr=sr, somega=somega)
    blowing = BlowSchedule(module, params, dt, sr=sr, somega=somega,
                          start=blow_start, duration=blow_duration, force_mn=blow_force,
                          spin_resets=spin_resets)
    state = prepare_initial_state(module, params, dt, v0=v0, s0=s0, phi0=phi0, oz=oz,
                                  magnetize_turns=magnetize_turns, disk_phase=disk_phase)
    sliding = None
    sliding_dir = None
    state = blowing.apply_resets(state)
    pre_step_count = pre / dt
    if not np.isfinite(pre_step_count) or pre_step_count > np.iinfo(np.int32).max + 0.5:
        raise ValueError("pre-evolution exceeds the C++ solver's maximum steps per call")
    pre_steps = int(round(pre_step_count))
    if pre_steps:
        print(f"Pre-evolving in C++ for {pre_steps} steps (t={pre_steps * dt:g})...")
        state, sliding, sliding_dir = blowing.evolve(state, pre_steps, sliding, sliding_dir)
    time = pre_steps * dt
    if sample_callback is not None:
        sample_callback(time, state.copy(), sliding)
    magnet_radius = magnet_count / 2
    units = module.physical_units(sr=sr, somega=somega)
    length_unit_mm = units["length_unit_m"] * 1000
    geometry = (expected_config or read_build_config(kahan=kahan)).get("magnet_geometry")
    magnet_body_radius = float(geometry["radius"]) if geometry is not None else 0.1
    magnet_description = (
        f"diameter {2 * magnet_body_radius * length_unit_mm:g} mm, "
        f"thickness {float(geometry['thickness']) * length_unit_mm:g} mm"
        if geometry is not None else "point dipoles")
    base_angles = 2 * np.pi * np.arange(magnet_count) / magnet_count if show_magnets else None

    fig = plt.figure(figsize=(14, 13 if show_work else 11), **({} if fast else {"layout": "constrained"}))
    grid = fig.add_gridspec(6 if show_work else 5, 2, width_ratios=(1.3, 1.0))
    scene = fig.add_subplot(grid[:, 0])
    radial_axis = fig.add_subplot(grid[0, 1])
    angular_axis = fig.add_subplot(grid[1, 1], sharex=radial_axis)
    displacement_axis = fig.add_subplot(grid[2, 1], sharex=radial_axis)
    velocity_axis = fig.add_subplot(grid[3, 1], sharex=radial_axis)
    spin_axis = fig.add_subplot(grid[4, 1], sharex=radial_axis)
    work_axis = fig.add_subplot(grid[5, 1], sharex=radial_axis) if show_work else None

    extent = magnet_radius + 1.0
    scene.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y")
    scene.set_aspect("equal", adjustable="box")
    scene.grid(alpha=0.12)
    # This is the measured magnet-centre ring, not an assumed physical plate edge.
    disk = Circle((0, 0), magnet_radius, facecolor="0.94", edgecolor="0.48", linewidth=1.5, linestyle="--", zorder=0)
    scene.add_patch(disk)
    magnets = []
    if show_magnets:
        for angle in base_angles + float(state[0]):
            magnet = Circle((magnet_radius * np.cos(angle), magnet_radius * np.sin(angle)),
                            magnet_body_radius, facecolor="0.48", edgecolor="0.28", linewidth=0.8, zorder=2)
            scene.add_patch(magnet)
            magnets.append(magnet)
    (path_line,) = scene.plot([], [], color="tab:blue", linewidth=1.6, alpha=0.85, zorder=3)
    ball = Circle((float(state[1]), float(state[2])), params.alpha,
                  facecolor="tab:orange", edgecolor="white", zorder=4)
    scene.add_patch(ball)
    status = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top", family="monospace",
                        fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    scene.set_title(f"Top view | {magnet_count} magnets: diameter 28 mm, thickness 15 mm\nDashed circle: magnet centres, radius 70 mm")

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
    spins_z = [float(state[11])]
    spin_times = [time]
    xs = [float(state[1])]
    ys = [float(state[2])]
    (radial_line,) = radial_axis.plot(times, radial, color="tab:blue")
    (polar_line,) = angular_axis.plot(times, polar, color="tab:purple")
    (displacement_line,) = displacement_axis.plot(times, equivalent_x, color="tab:green")
    (speed_line,) = velocity_axis.plot(times, speeds, color="tab:orange", label="COM speed |v|")
    (tangential_line,) = velocity_axis.plot(times, tangential_speeds, color="tab:red", label="signed tangential v")
    (spin_line,) = spin_axis.plot(times, spins_z, color="tab:brown")
    radial_axis.set_ylabel("r(t)")
    angular_axis.set_ylabel(r"$\phi(t)$ [rad]")
    displacement_axis.set_ylabel(r"$x_{eq} = \mathrm{mean}(r)\,\Delta\phi$")
    velocity_axis.set(xlabel="time (2D units)", ylabel="velocity (2D units)")
    velocity_axis.legend(loc="upper right", fontsize=8)
    velocity_axis.axhline(0, color="0.5", linewidth=0.7)
    spin_axis.set(xlabel="time (2D units)", ylabel=r"$\omega_z$ (2D units)")
    spin_axis.axhline(0, color="0.5", linewidth=0.7)
    for reset_time, _ in blowing.spin_resets:
        spin_axis.axvline(reset_time, color="0.5", linewidth=0.7, linestyle=":")
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
    for axis in (radial_axis, angular_axis, displacement_axis, velocity_axis, spin_axis):
        axis.grid(alpha=0.25)
    fig.suptitle(rf"Single 2D trajectory | sr={sr:g}, $s_\omega$={somega:g}")

    def render(*, update_graphs=True, rescale=True):
        if magnets:
            angle = base_angles + float(state[0])
            for magnet, x, y in zip(magnets, magnet_radius * np.cos(angle), magnet_radius * np.sin(angle)):
                magnet.center = (x, y)
        ball.center = (xs[-1], ys[-1])
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
            # Preserve both sides of each instantaneous reset, even in fast mode.
            spin_line.set_data(spin_times, spins_z)
        if rescale:
            radial_axis.set_xlim(0, max(1.0, time * 1.05))
            limits = [(radial_axis, radial), (angular_axis, polar),
                      (displacement_axis, equivalent_x), (velocity_axis, speeds + tangential_speeds),
                      (spin_axis, spins_z)]
            if show_work:
                limits.append((work_axis, [value for series in power_values for value in series]))
            for axis, values in limits:
                lower, upper = min(values), max(values)
                padding = max(0.05, 0.1 * (upper - lower))
                axis.set_ylim(lower - padding, upper + padding)
        status.set_text(
            f"t = {time:.3f}  (1D t = {magnet_radius * time:.3f})\n"
            f"r = {radial[-1]:.6f}  mean(r) = {mean_radial[-1]:.6f}\n"
            f"phi = {polar[-1]:.6f}  x_eq = {equivalent_x[-1]:.6f}\n"
            f"v = ({float(state[3]):.6f}, {float(state[4]):.6f})\n"
            f"|v| = {speeds[-1]:.6f}  v_tan = {tangential_speeds[-1]:.6f}\n"
            f"v_tan / {magnet_radius:g} (1D units) = {tangential_speeds[-1] / magnet_radius:.6f}\n"
            f"omega_z = {float(state[11]):.6f} (2D units)\n"
            f"omega_z = {float(state[11]) * blowing.omega:.6f} rad/s\n"
            f"{'sliding' if sliding else 'rolling'}"
            f"\nrelease time = {time / blowing.omega:.3f} s"
            f"\nblow = {blow_force if blowing.active(blowing.elapsed) else 0:g} mN"
        )
        artists = (ball, path_line, radial_line, polar_line, displacement_line,
                   speed_line, tangential_line, spin_line, *power_lines, status)
        return (*magnets, *artists)

    def update(_frame):
        nonlocal state, sliding, sliding_dir, time, radius_sum
        for _ in range(FAST_BATCH_SIZE if fast else 1):
            reset_count = len(blowing.reset_samples)
            state, sliding, sliding_dir = blowing.evolve(state, steps_per_frame, sliding, sliding_dir)
            time = blowing.elapsed
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
            for reset_time, before, after in blowing.reset_samples[reset_count:]:
                spin_times.extend((reset_time, reset_time))
                spins_z.extend((before, after))
            spin_times.append(time)
            spins_z.append(float(state[11]))
            tangential_speeds.append(
                float((state[1] * state[4] - state[2] * state[3]) / radial[-1])
                if radial[-1] > 0.0 else float("nan")
            )
            if show_work:
                for values, value in zip(power_values, module.magnetic_power(state, params)):
                    values.append(float(value))
            if sample_callback is not None:
                sample_callback(time, state.copy(), sliding)
        return render(update_graphs=not fast or _frame % FAST_GRAPH_UPDATE_EVERY == 0,
                      rescale=not fast or _frame % FAST_RESCALE_EVERY == 0)

    render()
    if fast:
        fig.tight_layout()
    animation = FuncAnimation(
        fig, update, init_func=render, frames=frames, interval=1000 / fps,
        blit=False, cache_frame_data=False,
    )
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=None,
                        help="2D time step (default: 0.0005 * 8 / compiled magnet count)")
    parser.add_argument("--steps-per-frame", type=int, default=STEPS_PER_FRAME)
    parser.add_argument("--fps", type=float, default=FPS, help="target display refresh rate (actual rate depends on rendering speed)")
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
    parser.add_argument("--somega", type=float, default=ANIMATION_SOMEGA)
    parser.add_argument("--v0", type=float, default=ANIMATION_V0, help="initial signed tangential speed in 1D units")
    parser.add_argument("--s0", type=float, default=0.0, help="initial signed tangential slip in 1D units; omega0=(s0-v0)/alpha")
    parser.add_argument("--phi0", type=float, default=0.0, help="initial orbital angle in radians")
    parser.add_argument("-oz", type=float, default=0.0,
                        help="initial omega_z in 2D units, matching the spin graph (default: 0)")
    parser.add_argument("--t,oz", dest="spin_resets", type=parse_spin_reset, action="append", default=[],
                        metavar="TIME,OMEGA_Z",
                        help="instantly reset omega_z at this release-relative 2D time, e.g. 2000,-5; repeatable")
    parser.add_argument("--disk-phase", type=float, default=DISK_PHASE, help="magnet-disk angle at release, in radians")
    parser.add_argument("--magnetize-turns", type=float, default=MAGNETIZE_TURNS,
                        help="hold the ball fixed for this many magnet-disk turns before release (default: 2)")
    parser.add_argument("--blow-start", type=float, default=None,
                        help="start blowing at this 2D time after release (graph units); requires --blow-duration")
    parser.add_argument("--blow-duration", type=float, default=None,
                        help="blowing duration in 2D time units (graph units); requires --blow-start")
    parser.add_argument("--blow-force", type=float, default=DEFAULT_BLOW_FORCE_MN,
                        help="following reverse tangential force through the ball centre, in mN (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", help="print and validate the build and runtime configuration without starting the animation")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # Keep older commands working.
    args = parser.parse_args()
    try:
        validate_blow(args.blow_start, args.blow_duration, args.blow_force)
    except ValueError as error:
        parser.error(str(error))
    if args.path_max_points < 0:
        parser.error("--path-max-points must be non-negative")
    if not all(np.isfinite(value) for value in (args.sr, args.somega, args.v0, args.s0, args.phi0, args.oz)):
        parser.error("parameters and initial values must be finite")
    if args.sr <= 0 or args.somega <= 0:
        parser.error("--sr and --somega must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if not np.isfinite(args.pre) or args.pre < 0:
        parser.error("--pre must be finite and non-negative")
    if args.steps_per_frame <= 0:
        parser.error("--steps-per-frame must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0):
        parser.error("--dt must be positive and finite")
    if not np.isfinite(args.disk_phase) or not np.isfinite(args.magnetize_turns) or args.magnetize_turns < 0:
        parser.error("--disk-phase must be finite and --magnetize-turns non-negative and finite")
    config = read_build_config(kahan=args.kahan)
    print_run_config(config, dt=args.dt, steps_per_frame=args.steps_per_frame,
                     frames=args.frames, pre=args.pre, fast=args.fast,
                     show_magnets=not args.hide_magnets, show_work=args.show_work,
                     sr=args.sr, somega=args.somega, v0=args.v0, s0=args.s0, phi0=args.phi0, oz=args.oz,
                     magnetize_turns=args.magnetize_turns, disk_phase=args.disk_phase, fps=args.fps,
                     blow_start=args.blow_start, blow_duration=args.blow_duration, blow_force=args.blow_force,
                     spin_resets=args.spin_resets)
    if args.dry_run:
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
        v0=args.v0, s0=args.s0, phi0=args.phi0, oz=args.oz,
        magnetize_turns=args.magnetize_turns, disk_phase=args.disk_phase,
        path_max_points=args.path_max_points, fps=args.fps,
        blow_start=args.blow_start, blow_duration=args.blow_duration, blow_force=args.blow_force,
        spin_resets=args.spin_resets,
    )
    plt.show()


if __name__ == "__main__":
    main()
