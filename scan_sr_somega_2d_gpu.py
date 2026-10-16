# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Scan the 2D hysteresis model over an ``(sr, somega)`` grid on CUDA.

Edit the constants below, then run this file directly from the ``src2d``
directory with a CuPy environment compatible with the target GPU. CMake builds
the native CUDA cubin; this script loads it and checks its build manifest.

The first 19 terminal-window metrics use the same column order and sampling rules
as the 1D phase kernel; mean_radius is appended. For the 2D model, v is signed tangential speed divided
by N/2, omega is radial-axis spin divided by N/2, and the primary order
parameter is mean_phase_rate = mean(1 - orbital angular speed - omega).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

from cuda_build import load_cuda_artifact
from project_paths import BUILD_DIR, RUNS_DIR, SCRATCH_DIR, image_output_path


# ============================= Run settings =============================
SR_LOWER = 0.1
SR_UPPER = 6.0
SOMEGA_LOWER = 0.1
SOMEGA_UPPER = 5.0
K = 256
DT_AT_EIGHT = 1.0e-3
TOTAL_TIME_AT_EIGHT = 1500.0
OBSERVATION_WINDOW_AT_EIGHT = 500.0
SAMPLE_TIME_AT_EIGHT = 0.05
LOCK_THRESHOLD = 0.05
SPEED_ONE_THRESHOLD = 0.05
RADIUS_TOLERANCE_AT_EIGHT = 0.8
# The following baseline times are for eight magnets. One-dimensional phase
# time is then 4 times the 2D time:
# TOTAL_TIME=2500 corresponds to a 1D horizon of 10000.
# DT=0.001 here corresponds to a 1D phase-time step of 0.004; use
# DT=0.00025 if a 1D step of 0.001 must also be matched.

# Local GPU runs use Kahan FP32 unless --fp64 is explicitly supplied.
USE_FP64 = False
# Field mode: 0 = exact field, 1 = Fourier-spline LUT.
FIELD_EVAL_MODE = 0
LUT_CACHE_MODE = 0  # 0 = off, 1 = nodes, 2 = spline coefficients.
BLOCK_X = 16
BLOCK_Y = 8

SHOW_PLOT = False
SAVE_PATH: str | Path | None = None  # Set below after reading the header.
SAVE_DATA_PATH: str | Path | None = None
SAVE_DPI = 180
# ========================================================================


ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kernels"
HEADER_DIR = KERNEL_DIR / "headers"
CONFIG_HEADER = HEADER_DIR / "hysteresis_config_field.hpp"


def read_magnet_number() -> int:
    source = CONFIG_HEADER.read_text(encoding="utf-8")
    matches = re.findall(r"^\s*CONFIG_CONST\s+int\s+magnet_number\s*=\s*(\d+)\s*;", source, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one literal magnet_number in {CONFIG_HEADER}; found {len(matches)}")
    count = int(matches[0])
    if count < 2 or count % 2:
        raise ValueError(f"magnet_number must be an even integer >= 2; got {count}")
    return count


MAGNET_NUMBER = read_magnet_number()
MAGNET_RADIUS = MAGNET_NUMBER / 2.0
TIME_SCALE = 8.0 / MAGNET_NUMBER
DT = DT_AT_EIGHT * TIME_SCALE
TOTAL_TIME = TOTAL_TIME_AT_EIGHT * TIME_SCALE
OBSERVATION_WINDOW = OBSERVATION_WINDOW_AT_EIGHT * TIME_SCALE
SAMPLE_TIME = SAMPLE_TIME_AT_EIGHT * TIME_SCALE
RADIUS_TOLERANCE = RADIUS_TOLERANCE_AT_EIGHT / TIME_SCALE
SAVE_PATH = SCRATCH_DIR / "phase-behavior" / f"sr_somega_2d-{MAGNET_NUMBER}-CYL.png"
SAVE_DATA_PATH = RUNS_DIR / "phase-behavior" / f"sr_somega_2d-{MAGNET_NUMBER}-CYL.npz"
LUT_PATH = BUILD_DIR / "field-lut" / ("field_lut_fp64.bin" if USE_FP64 else "field_lut_fp32.bin")
STATE_NAMES = (
    "down_disk_angle", "pos_x", "pos_y", "speed_x", "speed_y",
    "attitude_w", "attitude_x", "attitude_y", "attitude_z",
    "omega_x", "omega_y", "omega_z", "q_x", "q_y", "q_z",
)
METRIC_NAMES = (
    "mean_v", "std_v", "mean_omega", "mean_phase_rate", "std_phase_rate",
    "mean_abs_slip", "sliding_fraction", "mean_abs_q", "final_v",
    "final_omega", "final_phase_rate", "final_abs_q", "delta_mean_v",
    "delta_mean_phase_rate", "min_v", "max_v", "sample_count",
    "global_minimum_step", "velocity_recovered", "mean_radius",
)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}
_CUDA_MODULE = None


def _get_cupy():
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError("CuPy is required; install a build compatible with the CUDA driver on the target GPU.") from exc
    return cp


def _load_lut(cp, dtype: np.dtype, manifest: dict):
    lut_path = Path(manifest["lut"])
    if not lut_path.is_file():
        raise FileNotFoundError(
            f"CMake-generated LUT not found: {lut_path}. Configure/build src2d with IYPT16_FIELD_MODE=lut first."
        )
    # Keep the host copy as the reference for exact compatibility checks;
    # the kernel stages the uploaded table into per-CTA shared memory.
    parameters = manifest["field_parameters"]
    expected_values = int(parameters["kFieldFourierModeCount"]) * int(parameters["kSplineNodes"]) * 4
    lut_host = np.fromfile(lut_path, dtype=dtype)
    if lut_host.size != expected_values:
        raise ValueError(
            f"LUT has {lut_host.size} scalars, expected {expected_values}; regenerate it after changing "
            "magnet_number, kFieldFourierModeCount, or kSplineNodes."
        )
    return cp.asarray(lut_host)


def _get_cuda_module(cp):
    global _CUDA_MODULE, FIELD_EVAL_MODE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    cubin, manifest = load_cuda_artifact(
        "sr_somega", "fp64" if USE_FP64 else "fp32_kahan",
        "lut" if FIELD_EVAL_MODE else "exact", LUT_CACHE_MODE,
    )
    if manifest.get("magnet_model") == "cylinder":
        FIELD_EVAL_MODE = 1
    _CUDA_MODULE = cp.RawModule(path=str(cubin)), manifest
    return _CUDA_MODULE


def run_scan():
    if FIELD_EVAL_MODE not in (0, 1):
        raise ValueError("FIELD_EVAL_MODE must be 0 (exact) or 1 (LUT)")
    if LUT_CACHE_MODE not in (0, 1, 2):
        raise ValueError("LUT_CACHE_MODE must be 0, 1, or 2")
    if not (0.0 < SR_LOWER <= SR_UPPER and 0.0 < SOMEGA_LOWER <= SOMEGA_UPPER):
        raise ValueError("scan bounds must be finite, positive, and increasing")
    if not 1 <= K <= 46340:
        raise ValueError("K must be between 1 and 46340")
    time_values = np.asarray((DT, TOTAL_TIME, OBSERVATION_WINDOW, SAMPLE_TIME), dtype=np.float64)
    if not np.all(np.isfinite(time_values)) or np.any(time_values <= 0):
        raise ValueError("DT, TOTAL_TIME, OBSERVATION_WINDOW and SAMPLE_TIME must be positive and finite")
    if OBSERVATION_WINDOW > TOTAL_TIME:
        raise ValueError("OBSERVATION_WINDOW must not exceed TOTAL_TIME")
    if not np.isfinite(LOCK_THRESHOLD) or LOCK_THRESHOLD <= 0:
        raise ValueError("LOCK_THRESHOLD must be positive and finite")
    steps = round(TOTAL_TIME / DT)
    window_steps = min(steps, round(OBSERVATION_WINDOW / DT))
    sample_stride = max(1, round(SAMPLE_TIME / DT))
    if steps > np.iinfo(np.int32).max:
        raise ValueError("step count exceeds int32")
    if window_steps < 2 * sample_stride:
        raise ValueError("observation window must contain at least two samples in each half")
    if BLOCK_X <= 0 or BLOCK_Y <= 0 or BLOCK_X * BLOCK_Y > 128:
        raise ValueError("CUDA block dimensions must contain between 1 and 128 threads")

    dtype = np.dtype(np.float64 if USE_FP64 else np.float32)
    sr = np.linspace(SR_LOWER, SR_UPPER, K, dtype=dtype)
    somega = np.linspace(SOMEGA_LOWER, SOMEGA_UPPER, K, dtype=dtype)
    cp = _get_cupy()
    module, manifest = _get_cuda_module(cp)
    lut_gpu = _load_lut(cp, dtype, manifest) if FIELD_EVAL_MODE == 1 else np.uintp(0)
    kernel = module.get_function("scan_sr_somega_cuda")

    states_gpu = cp.empty((K * K, len(STATE_NAMES)), dtype=dtype)
    status_gpu = cp.full(K * K, -1, dtype=cp.int32)
    metrics_gpu = cp.full((K * K, len(METRIC_NAMES)), cp.nan, dtype=dtype)
    block = (BLOCK_X, BLOCK_Y, 1)
    grid = ((K + BLOCK_X - 1) // BLOCK_X, (K + BLOCK_Y - 1) // BLOCK_Y, 1)
    args = (
        dtype.type(SR_LOWER), dtype.type(SR_UPPER),
        dtype.type(SOMEGA_LOWER), dtype.type(SOMEGA_UPPER),
        np.int32(K), dtype.type(DT), np.int32(steps),
        np.int32(window_steps), np.int32(sample_stride),
        lut_gpu,
        states_gpu, status_gpu, metrics_gpu,
    )

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    kernel(grid, block, args)
    stop.record()
    stop.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000.0

    states = cp.asnumpy(states_gpu).reshape(K, K, len(STATE_NAMES))
    status = cp.asnumpy(status_gpu).reshape(K, K)
    metrics = cp.asnumpy(metrics_gpu).reshape(K, K, len(METRIC_NAMES))
    success = status == 0
    if np.any(success & ~np.all(np.isfinite(states), axis=-1)):
        raise RuntimeError("successful CUDA trajectories contain non-finite state values")
    if np.any(success & ~np.all(np.isfinite(metrics), axis=-1)):
        raise RuntimeError("successful CUDA trajectories contain non-finite terminal-window metrics")
    print(f"2D CUDA scan completed in {elapsed:.3f} s: {np.count_nonzero(success)}/{K*K} trajectories succeeded.")
    failed_status, failed_count = np.unique(status[~success], return_counts=True)
    if failed_status.size:
        print("Failed trajectory status counts: " + ", ".join(
            f"{int(code)}={int(count)}" for code, count in zip(failed_status, failed_count)
        ))
    return sr, somega, status, metrics, states


def _cell_edges(points: np.ndarray) -> np.ndarray:
    if points.size == 1:
        return np.asarray((points[0] - 0.5, points[0] + 0.5))
    midpoints = 0.5 * (points[:-1] + points[1:])
    return np.concatenate(((2 * points[0] - midpoints[0],), midpoints, (2 * points[-1] - midpoints[-1],)))


def phase_locked(status: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    mean_phase_rate = metrics[..., METRIC_INDEX["mean_phase_rate"]]
    return (status == 0) & (np.abs(mean_phase_rate) < LOCK_THRESHOLD)


def radius_in_tolerance(status: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    mean_radius = metrics[..., METRIC_INDEX["mean_radius"]]
    return (status == 0) & (np.abs(mean_radius - MAGNET_RADIUS) <= RADIUS_TOLERANCE)


def speed_matches_disk(status: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    """The v≈1 band used to identify following in the 1D phase diagram."""
    mean_v = metrics[..., METRIC_INDEX["mean_v"]]
    return (status == 0) & (np.abs(mean_v - 1.0) <= SPEED_ONE_THRESHOLD)


def forward_orbit_follows_disk(status: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    """Identify a near-magnet orbit whose mean tangential speed follows the disk."""
    return speed_matches_disk(status, metrics) & radius_in_tolerance(status, metrics)


def save_plot(sr, somega, status, metrics):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    valid = status == 0
    speed_one = speed_matches_disk(status, metrics)
    mean_phase_rate = np.ma.masked_where(
        ~valid, metrics[..., METRIC_INDEX["mean_phase_rate"]]
    )
    mean_velocity_1d = np.ma.masked_where(
        ~valid, metrics[..., METRIC_INDEX["mean_v"]]
    )
    mean_v = metrics[..., METRIC_INDEX["mean_v"]]
    mask_class = speed_one.astype(np.int8)
    mask_class[mean_v < 0.0] = 2
    classification = np.ma.masked_where(~valid, mask_class)
    mean_radius = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["mean_radius"]])
    radius_mask = np.ma.masked_where(~valid, radius_in_tolerance(status, metrics).astype(np.int8))
    sr_edges = _cell_edges(sr)
    somega_edges = _cell_edges(somega)
    figure, grid_axes = plt.subplots(2, 3, figsize=(19.0, 10.2), constrained_layout=True)
    axes = grid_axes.ravel()

    clipped = axes[0].pcolormesh(
        sr_edges, somega_edges, np.clip(mean_phase_rate, 0.0, 1.0),
        shading="flat", cmap="magma", vmin=0.0, vmax=1.0,
    )
    axes[0].set(title=r"Order parameter $\langle\dot\psi\rangle$ (clipped)")
    figure.colorbar(clipped, ax=axes[0], label=r"$\langle\dot\psi\rangle$ (clipped)")

    raw_phase_cmap = plt.get_cmap("magma").copy()
    raw_phase_cmap.set_bad("0.65")
    raw = axes[1].pcolormesh(
        sr_edges, somega_edges, mean_phase_rate, shading="flat", cmap=raw_phase_cmap,
    )
    axes[1].set(title=r"Order parameter $\langle\dot\psi\rangle$ (unclipped)")
    figure.colorbar(raw, ax=axes[1], label=r"$\langle\dot\psi\rangle$")

    velocity_cmap = plt.get_cmap("viridis").copy()
    velocity_cmap.set_bad("0.65")
    velocity = axes[2].pcolormesh(
        sr_edges, somega_edges, mean_velocity_1d, shading="flat", cmap=velocity_cmap,
    )
    axes[2].set(title=rf"Terminal mean velocity in 1D units, $\langle v_{{\rm tan}}/{MAGNET_RADIUS:g}\rangle$")
    figure.colorbar(velocity, ax=axes[2], label=rf"$\langle v_{{\rm tan}}/{MAGNET_RADIUS:g}\rangle$")

    mask_cmap = ListedColormap(("green", "red", "blue"))
    mask_cmap.set_bad("0.65")
    classification_image = axes[3].pcolormesh(
        sr_edges, somega_edges, classification, shading="flat",
        cmap=mask_cmap, norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5), mask_cmap.N),
    )
    mask_bar = figure.colorbar(classification_image, ax=axes[3], ticks=(0, 1, 2))
    mask_bar.ax.set_yticklabels(("other speed", "v near 1", "mean v < 0"))
    axes[3].set(title=rf"Terminal mean velocity mask: $|\langle v\rangle-1|\leq {SPEED_ONE_THRESHOLD:g}$")

    radius_cmap = plt.get_cmap("viridis").copy()
    radius_cmap.set_bad("0.65")
    radius_image = axes[4].pcolormesh(
        sr_edges, somega_edges, mean_radius, shading="flat", cmap=radius_cmap,
    )
    axes[4].set(title=r"Terminal mean radius $\langle r\rangle$")
    figure.colorbar(radius_image, ax=axes[4], label=r"$\langle r\rangle$")

    radius_mask_cmap = ListedColormap(("tab:red", "tab:green"))
    radius_mask_cmap.set_bad("0.65")
    radius_mask_image = axes[5].pcolormesh(
        sr_edges, somega_edges, radius_mask, shading="flat",
        cmap=radius_mask_cmap, norm=BoundaryNorm((-0.5, 0.5, 1.5), radius_mask_cmap.N),
    )
    radius_bar = figure.colorbar(radius_mask_image, ax=axes[5], ticks=(0, 1))
    radius_bar.ax.set_yticklabels(("outside", "within"))
    axes[5].set(title=rf"Radius mask: $|\langle r\rangle-{MAGNET_RADIUS:g}|\leq {RADIUS_TOLERANCE:g}$")

    for axis in axes:
        axis.set(xlabel=r"$s_r$", ylabel=r"$s_\omega$")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"2D hysteresis parameter scan · K={K} · T={steps_to_time():g} · "
        f"{'FP64' if USE_FP64 else 'Kahan FP32'}"
    )
    return figure


def steps_to_time() -> float:
    return round(TOTAL_TIME / DT) * DT


def print_run_config(*, profile: bool) -> None:
    global FIELD_EVAL_MODE
    from build_metadata import print_build_metadata
    cubin, manifest = load_cuda_artifact(
        "sr_somega", "fp64" if USE_FP64 else "fp32_kahan",
        "lut" if FIELD_EVAL_MODE else "exact", LUT_CACHE_MODE,
    )
    if int(manifest["field_parameters"]["magnet_number"]) != MAGNET_NUMBER:
        raise RuntimeError("GPU magnet count differs from this scan; rebuild the CMake target")
    FIELD_EVAL_MODE = 1 if manifest["field_mode"] == "lut" else 0
    def row(label: str, value: object) -> None:
        print(f"  {label:<20} {value}")

    def section(title: str) -> None:
        print(f"\n  [{title}]")

    steps = round(TOTAL_TIME / DT)
    window_steps = min(steps, round(OBSERVATION_WINDOW / DT))
    sample_stride = max(1, round(SAMPLE_TIME / DT))
    print("\n=== 2D CUDA scan | effective configuration ===")
    print_build_metadata(manifest)
    section("Model and grid")
    row("Magnets / radius", f"{MAGNET_NUMBER} / {MAGNET_RADIUS:g}")
    row("Grid", f"{K} x {K}")
    row("s_r range", f"{SR_LOWER:g} .. {SR_UPPER:g}")
    row("s_omega range", f"{SOMEGA_LOWER:g} .. {SOMEGA_UPPER:g}")
    row("Precision / field", f"{'FP64' if USE_FP64 else 'Kahan FP32'} / {'exact' if FIELD_EVAL_MODE == 0 else 'LUT'}")
    row("LUT cache", LUT_CACHE_MODE if FIELD_EVAL_MODE == 1 else "inactive (exact field)")
    row("CUDA block", f"{BLOCK_X} x {BLOCK_Y}")
    section("Time (2D units)")
    row("dt", f"{DT:.12g}")
    row("Total", f"{TOTAL_TIME:.12g}  ({steps:,} steps)")
    row("Observation window", f"{OBSERVATION_WINDOW:.12g}  ({window_steps:,} steps)")
    row("Sample interval", f"{SAMPLE_TIME:.12g}  ({sample_stride:,} steps)")
    section("Classification")
    row("Phase lock", f"{LOCK_THRESHOLD:g}")
    row("Speed near 1", f"{SPEED_ONE_THRESHOLD:g}")
    row("Radius tolerance", f"{RADIUS_TOLERANCE:g}")
    section("Files")
    row("CUDA build", cubin)
    row("Magnet source", CONFIG_HEADER)
    if FIELD_EVAL_MODE == 1:
        row("LUT", LUT_PATH)
        row("LUT shape", "read from cubin build manifest")
    if not profile:
        row("Data", SAVE_DATA_PATH)
        row("Figure", SAVE_PATH)
    print(flush=True)


def main(*, use_fp64: bool = False, profile: bool = False, field_mode: str = "exact",
         lut_cache_mode: int = 0, save_figure: Path | None = None,
         save_data: Path | None = None, yes: bool = False, dry_run: bool = False):
    global USE_FP64, LUT_PATH, TOTAL_TIME, OBSERVATION_WINDOW, LUT_CACHE_MODE
    global FIELD_EVAL_MODE, SAVE_PATH, SAVE_DATA_PATH, _CUDA_MODULE
    USE_FP64 = bool(use_fp64)
    FIELD_EVAL_MODE = 1 if field_mode == "lut" else 0
    LUT_CACHE_MODE = lut_cache_mode
    _CUDA_MODULE = None
    if save_figure is not None:
        SAVE_PATH = save_figure
    if save_data is not None:
        SAVE_DATA_PATH = save_data
    LUT_PATH = BUILD_DIR / "field-lut" / ("field_lut_fp64.bin" if USE_FP64 else "field_lut_fp32.bin")
    if profile:
        TOTAL_TIME = 50.0 * TIME_SCALE
        OBSERVATION_WINDOW = 0.5 * TIME_SCALE
    print_run_config(profile=profile)
    if dry_run:
        return
    if not yes:
        try:
            answer = input("Run this GPU scan? Type yes to continue: ").strip()
        except EOFError as exc:
            raise SystemExit("No confirmation received; GPU scan cancelled. Use --yes for non-interactive runs.") from exc
        if answer != "yes":
            print("GPU scan cancelled.")
            return
    if profile:
        run_scan()
        return
    sr, somega, status, metrics, states = run_scan()
    locked = phase_locked(status, metrics)
    speed_one = speed_matches_disk(status, metrics)
    print(f"v≈1 cells: {np.count_nonzero(speed_one)}/{np.count_nonzero(status == 0)} successful cells.")
    forward_orbit = forward_orbit_follows_disk(status, metrics)
    print(f"v≈1 cells near the magnet radius: {np.count_nonzero(forward_orbit)} successful cells.")
    if SAVE_DATA_PATH is not None:
        data_path = Path(SAVE_DATA_PATH)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_path,
            sr=sr,
            somega=somega,
            status=status,
            metrics=metrics,
            metric_names=np.asarray(METRIC_NAMES),
            locked=locked,
            speed_one=speed_one,
            forward_orbit=forward_orbit,
            lock_threshold=np.asarray(LOCK_THRESHOLD),
            speed_one_threshold=np.asarray(SPEED_ONE_THRESHOLD),
            mean_radius=metrics[..., METRIC_INDEX["mean_radius"]],
            radius_in_tolerance=radius_in_tolerance(status, metrics),
            magnet_radius=np.asarray(MAGNET_RADIUS),
            radius_tolerance=np.asarray(RADIUS_TOLERANCE),
            final_state=states,
            state_names=np.asarray(STATE_NAMES),
            dt=np.asarray(DT),
            total_time=np.asarray(steps_to_time()),
            observation_window=np.asarray(OBSERVATION_WINDOW),
            sample_time=np.asarray(SAMPLE_TIME),
            use_fp64=np.asarray(USE_FP64),
            use_kahan=np.asarray(not USE_FP64),
            lut_cache_mode=np.asarray(LUT_CACHE_MODE),
        )
        print(f"Saved data to {data_path.resolve()}")

    figure = save_plot(sr, somega, status, metrics)
    if SAVE_PATH is not None:
        path = image_output_path(SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=SAVE_DPI)
        print(f"Saved figure to {path.resolve()}")
    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp64",
        action="store_true",
        help="use FP64 arithmetic and the CMake-generated FP64 LUT (intended for the cloud V100S)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a 50000-step scan without saving data or plots",
    )
    parser.add_argument(
        "--lut-cache-mode", type=int, choices=(0, 1, 2), default=0,
        help="LUT interpolation cache: 0=off (default), 1=nodes, 2=spline coefficients",
    )
    parser.add_argument("--field-mode", choices=("exact", "lut"), default="exact")
    parser.add_argument("--save-figure", type=Path)
    parser.add_argument("--save-data", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the effective configuration without starting the GPU")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation after printing the configuration")
    args = parser.parse_args()
    main(use_fp64=args.fp64, profile=args.profile, field_mode=args.field_mode,
         lut_cache_mode=args.lut_cache_mode, save_figure=args.save_figure, save_data=args.save_data,
         yes=args.yes, dry_run=args.dry_run)
