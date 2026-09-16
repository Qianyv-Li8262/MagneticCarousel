"""Run the long-time ``(sr, somega)`` phase sweep on CUDA.

The default run focuses on the locking boundary and saves both the raw
observables and a phase diagram::

    conda run -n daily python scan_sr_somega_gpu.py
    conda run -n daily python scan_sr_somega_gpu.py --ordinary
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from build_codegen import COEFFS_INIT
from project_paths import RUNS_DIR, SCRATCH_DIR, image_output_path


# ============================= Run settings =============================
SR_LOWER = 3.0
SR_UPPER = 8.0
SOMEGA_LOWER = 0.1
SOMEGA_UPPER = 2.0
K = 128
DT = 1.0e-3
TOTAL_TIME = 50000.0
OBSERVATION_WINDOW = 4000.0
SAMPLE_TIME = 0.2

BLOCK_X = 16
BLOCK_Y = 8

SHOW_PLOT = False
SAVE_PATH: str | Path | None = SCRATCH_DIR / "phase-behavior" / "sr_somega_phase_0.3_real_scanright2.png"
SAVE_DATA_PATH: str | Path | None = RUNS_DIR / "phase-behavior" / "sr_somega_phase.npz"
SAVE_DPI = 220
FINAL_VELOCITY_TARGET = 1.0
FINAL_VELOCITY_TOL = 0.05

# Same reference point as animate_hysteresis_cpu.py.
REFERENCE_ALPHA = 0.33333333
REFERENCE_DELTA = 1.7212357e-4
REFERENCE_GAMMA_G = 0.16560356
REFERENCE_GAMMA = 489.39639
REFERENCE_GAMMA_M = 11011.419

# Material constants used only for the analytic torque-limit overlay.
MU_R = 2.2926667e3
ALPHA_JA = 3.2e-3
C = 0.38
K_HYSTER = 1.0909091e-3
H_EXT = 1.4661546e-1
ETA = 0.4
XI_JA = 1.0 - 1.0 / (1.0 + ALPHA_JA * (MU_R - 1.0))
Q_LIMIT = XI_JA / ALPHA_JA * K_HYSTER
E_DEMAG = (1.0 - C) / ((MU_R + 2.0) * (1.0 - XI_JA))
BOUNDARY_SCALE = np.sqrt(
    REFERENCE_GAMMA_M * E_DEMAG * Q_LIMIT * H_EXT * ETA
    / (REFERENCE_ALPHA * REFERENCE_DELTA)
)
# ========================================================================


ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kernels"
HEADER_DIR = KERNEL_DIR / "headers"
CUDA_SOURCE = KERNEL_DIR / "1dmodel-hysteresis.cu"
PHASE_CUDA_SOURCE = KERNEL_DIR / "phase_behavior.cu"
CUDA_HEADER = HEADER_DIR / "1dmodel-hysteresis.hpp"
CUDA_GENERATED_HEADER = KERNEL_DIR / "generated" / "field_coefficients_nvrtc.generated.cuh"
CUDA_SUPPORT_HEADERS = (
    CUDA_HEADER,
    HEADER_DIR / "hysteresis_types.hpp",
    HEADER_DIR / "hysteresis_config.hpp",
    HEADER_DIR / "numeric_extensions" / "kahan.hpp",
    CUDA_GENERATED_HEADER,
)
_CUDA_MODULES = {}

METRIC_NAMES = (
    "mean_v",
    "std_v",
    "mean_omega",
    "mean_phase_rate",
    "std_phase_rate",
    "mean_abs_slip",
    "sliding_fraction",
    "mean_abs_q",
    "final_v",
    "final_omega",
    "final_phase_rate",
    "final_abs_q",
    "delta_mean_v",
    "delta_mean_phase_rate",
    "min_v",
    "max_v",
    "sample_count",
    "global_minimum_step",
    "velocity_recovered",
)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


@dataclass(frozen=True)
class ReferenceParams:
    alpha: float
    delta: float
    gamma_g: float
    gamma: float
    gamma_m: float


DEFAULT_REFERENCE = ReferenceParams(
    alpha=REFERENCE_ALPHA,
    delta=REFERENCE_DELTA,
    gamma_g=REFERENCE_GAMMA_G,
    gamma=REFERENCE_GAMMA,
    gamma_m=REFERENCE_GAMMA_M,
)


def _get_cupy():
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuPy is not installed. Run this script in a Python environment "
            "containing a CuPy build compatible with the installed CUDA driver."
        ) from exc
    return cp


def _get_cuda_module(cp, use_kahan: bool = True):
    """Compile both legacy-minimum and long-time phase kernels in one module."""
    COEFFS_INIT("nvrtc")
    dependencies = (*CUDA_SUPPORT_HEADERS, CUDA_SOURCE, PHASE_CUDA_SOURCE)
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"CUDA source not found: {path}")
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    source_signature = digest.hexdigest()[:16]
    module_key = (use_kahan, source_signature)
    if module_key not in _CUDA_MODULES:
        legacy = CUDA_SOURCE.read_text(encoding="utf-8")
        phase = PHASE_CUDA_SOURCE.read_text(encoding="utf-8")
        include_line = '#include "1dmodel-hysteresis.hpp"'
        if include_line not in legacy or include_line not in phase:
            raise RuntimeError("Expected shared-header include was not found")
        source = legacy
        source += "\n" + phase.replace(include_line, "", 1)
        # CuPy's NVRTC cache key does not track the contents of headers reached
        # through -I.  Include their digest in the options so changing any
        # model macro or generated coefficient always recompiles the kernel.
        options = (
            "--std=c++17",
            f"--include-path={HEADER_DIR}",
            f"-DIYPT16_SOURCE_SIGNATURE_{source_signature}=1",
        )
        if use_kahan:
            options += ("-DUSE_KAHAN",)
        _CUDA_MODULES[module_key] = cp.RawModule(code=source, options=options, backend="nvrtc")
    return _CUDA_MODULES[module_key]


def _cell_edges(points: np.ndarray) -> np.ndarray:
    if points.size == 1:
        half_width = max(0.05 * abs(float(points[0])), 0.05)
        return np.asarray((points[0] - half_width, points[0] + half_width))
    midpoints = 0.5 * (points[:-1] + points[1:])
    return np.concatenate(
        ((points[0] - (midpoints[0] - points[0]),), midpoints, (points[-1] + (points[-1] - midpoints[-1]),))
    )


def _symmetric_limit(values: np.ndarray, valid: np.ndarray) -> float:
    finite_values = values[valid & np.isfinite(values)]
    if finite_values.size == 0:
        return 1.0
    limit = float(np.max(np.abs(finite_values)))
    return limit if limit > 0.0 else 1.0


def locking_limit(sr):
    sr = np.asarray(sr)
    return BOUNDARY_SCALE * np.abs(1.0 - REFERENCE_ALPHA * sr) ** 1.5 / sr


def _time_arguments(dt: float, total_time: float, window: float, sample_time: float):
    values = np.asarray((dt, total_time, window, sample_time), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("time settings must be positive and finite")
    steps = round(total_time / dt)
    window_steps = min(steps, round(window / dt))
    sample_stride = max(1, round(sample_time / dt))
    if steps > np.iinfo(np.int32).max:
        raise ValueError("step count exceeds int32")
    if window_steps < 2 * sample_stride:
        raise ValueError("observation window must contain at least two samples")
    return steps, window_steps, sample_stride


def run_sweep(
    sr_lower: float,
    sr_upper: float,
    somega_lower: float,
    somega_upper: float,
    k: int,
    dt: float,
    total_time: float,
    reference: ReferenceParams = DEFAULT_REFERENCE,
    block_shape: tuple[int, int] = (BLOCK_X, BLOCK_Y),
    observation_window: float = OBSERVATION_WINDOW,
    sample_time: float = SAMPLE_TIME,
    use_kahan: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``sr``, ``somega``, status and long-time metrics arrays."""
    if not (0.0 < sr_lower < sr_upper and 0.0 < somega_lower < somega_upper):
        raise ValueError("parameter bounds must be positive and increasing")
    if not 2 <= k <= 46340:
        raise ValueError("k must be between 2 and 46340")
    steps, window_steps, sample_stride = _time_arguments(dt, total_time, observation_window, sample_time)
    sr = np.linspace(sr_lower, sr_upper, k, dtype=np.float32)
    somega = np.linspace(somega_lower, somega_upper, k, dtype=np.float32)

    cp = _get_cupy()
    print(
        f"Loading CUDA module ({'Kahan FP32' if use_kahan else 'ordinary FP32'})...",
        flush=True,
    )
    kernel = _get_cuda_module(cp, use_kahan=use_kahan).get_function("AnalyzeParameterGrid")
    status_gpu = cp.full((k, k), 4, dtype=cp.int32)
    metrics_gpu = cp.full((k, k, len(METRIC_NAMES)), cp.nan, dtype=cp.float32)
    block_x, block_y = block_shape
    block = (block_x, block_y, 1)
    grid = ((k + block_x - 1) // block_x, (k + block_y - 1) // block_y, 1)

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    kernel(
        grid,
        block,
        (
            cp.asarray(sr), cp.asarray(somega), np.int32(k), np.float32(dt), np.int32(steps),
            np.int32(window_steps), np.int32(sample_stride), np.float32(reference.alpha),
            np.float32(reference.delta), np.float32(reference.gamma_g), np.float32(reference.gamma),
            np.float32(reference.gamma_m), status_gpu, metrics_gpu,
        ),
    )
    stop.record()
    stop.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000.0
    status = cp.asnumpy(status_gpu)
    metrics = cp.asnumpy(metrics_gpu)
    success = status == 0
    if np.any(success & ~np.all(np.isfinite(metrics), axis=-1)):
        raise RuntimeError("CUDA returned non-finite metrics for a successful trajectory")
    print(f"CUDA phase sweep finished in {elapsed:.3f} s: {np.count_nonzero(success)}/{k*k} succeeded.")
    return sr, somega, status, metrics


def _collapse(ratio: np.ndarray, phase_rate: np.ndarray):
    edges = np.geomspace(0.1, 20.0, 36)
    centers = np.sqrt(edges[:-1] * edges[1:])
    quantiles = np.full((centers.size, 3), np.nan)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = phase_rate[(ratio >= lower) & (ratio < upper) & np.isfinite(phase_rate)]
        if selected.size:
            quantiles[index] = np.quantile(selected, (0.1, 0.5, 0.9))
    return centers, quantiles


def plot_results(
    sr: np.ndarray,
    somega: np.ndarray,
    status: np.ndarray,
    metrics: np.ndarray,
    *,
    use_kahan: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    valid = status == 0
    mean_phase_rate = np.ma.masked_where(
        ~valid, metrics[..., METRIC_INDEX["mean_phase_rate"]]
    )
    terminal_mean_velocity = np.ma.masked_where(
        ~valid, metrics[..., METRIC_INDEX["mean_v"]]
    )
    mask_class = (
        np.abs(metrics[..., METRIC_INDEX["mean_v"]] - FINAL_VELOCITY_TARGET)
        <= FINAL_VELOCITY_TOL
    ).astype(np.int8)
    mask_class[metrics[..., METRIC_INDEX["mean_v"]] < 0.0] = 2
    velocity_mask = np.ma.masked_where(~valid, mask_class)

    figure, grid_axes = plt.subplots(2, 2, figsize=(13.5, 10.2), constrained_layout=True)
    axes = grid_axes.ravel()

    phase_image = axes[0].pcolormesh(
        _cell_edges(sr), _cell_edges(somega), np.clip(mean_phase_rate, 0.0, 1.0),
        shading="flat", cmap="magma", vmin=0.0, vmax=1.0,
    )
    left = np.linspace(float(sr.min()), 1.0 / REFERENCE_ALPHA - 1.0e-4, 1000)
    right = np.linspace(1.0 / REFERENCE_ALPHA + 1.0e-4, float(sr.max()), 1000)
    axes[0].plot(left, locking_limit(left), "w--", lw=2.0, label="analytic torque limit")
    axes[0].plot(right, locking_limit(right), "w--", lw=2.0)
    axes[0].set(title=r"Order parameter $\langle\dot\phi\rangle$")
    axes[0].set_ylim(float(somega.min()), float(somega.max()))
    axes[0].legend(loc="upper left")
    figure.colorbar(phase_image, ax=axes[0], label=r"$\langle\dot\phi\rangle$ (clipped)")

    raw_phase_cmap = plt.get_cmap("magma").copy()
    raw_phase_cmap.set_bad("0.65")
    raw_phase_image = axes[1].pcolormesh(
        _cell_edges(sr), _cell_edges(somega), mean_phase_rate,
        shading="flat", cmap=raw_phase_cmap,
    )
    axes[1].set(title=r"Order parameter $\langle\dot\phi\rangle$ (unclipped)")
    figure.colorbar(raw_phase_image, ax=axes[1], label=r"$\langle\dot\phi\rangle$")

    velocity_cmap = plt.get_cmap("viridis").copy()
    velocity_cmap.set_bad("0.65")
    velocity_image = axes[2].pcolormesh(
        _cell_edges(sr), _cell_edges(somega), terminal_mean_velocity,
        cmap=velocity_cmap,
        shading="flat",
    )
    figure.colorbar(velocity_image, ax=axes[2], label=r"terminal-window $\langle v\rangle$")
    axes[2].set(title="Terminal mean velocity")

    # 0: terminal mean v not close to 1 (green); 1: close to 1 (red);
    # 2: mean velocity below zero (blue, taking priority over both).
    mask_cmap = ListedColormap(("green", "red", "blue"))
    mask_cmap.set_bad("0.65")
    mask_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), mask_cmap.N)
    mask_image = axes[3].pcolormesh(
        _cell_edges(sr), _cell_edges(somega), velocity_mask,
        cmap=mask_cmap,
        norm=mask_norm,
        shading="flat",
    )
    mask_bar = figure.colorbar(mask_image, ax=axes[3], ticks=(0, 1, 2))
    mask_bar.ax.set_yticklabels(("mean v not close to 1", "mean v close to 1", "mean v < 0"))
    axes[3].set(
        title=rf"Terminal mean velocity mask ($|\langle v\rangle-1|\leq {FINAL_VELOCITY_TOL:g}$)",
    )

    for axis in axes:
        axis.set(
            xlabel=r"$s_r=r/r_{\mathrm{ref}}$",
            ylabel=r"$s_\omega=\omega/\omega_{\mathrm{ref}}$",
        )
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Self-consistent hysteresis model: terminal velocities "
        f"(k={sr.size}, T={TOTAL_TIME:g}, {'Kahan FP32' if use_kahan else 'ordinary FP32'})",
        fontsize=15,
    )
    return figure


def _print_phase_summary(sr, somega, status, metrics):
    valid = status == 0
    phase_rate = metrics[..., METRIC_INDEX["mean_phase_rate"]]
    observed = valid & (phase_rate > 0.05)
    predicted = valid & (somega[:, None] > locking_limit(sr[None, :]))
    accuracy = np.mean(observed[valid] == predicted[valid]) if np.any(valid) else np.nan
    slip = metrics[..., METRIC_INDEX["mean_abs_slip"]]
    print(f"Locked cells: {np.count_nonzero(valid & (np.abs(phase_rate) < 0.05))}/{np.count_nonzero(valid)}")
    print(f"Analytic-boundary classification accuracy: {accuracy:.3%}")
    print(f"Mean rolling residual: median={np.nanmedian(slip[valid]):.3e}, max={np.nanmax(slip[valid]):.3e}")


def main(*, use_kahan: bool = True) -> None:
    sr, somega, status, metrics = run_sweep(
        SR_LOWER, SR_UPPER, SOMEGA_LOWER, SOMEGA_UPPER, K, DT, TOTAL_TIME,
        reference=DEFAULT_REFERENCE, block_shape=(BLOCK_X, BLOCK_Y),
        observation_window=OBSERVATION_WINDOW, sample_time=SAMPLE_TIME,
        use_kahan=use_kahan,
    )
    _print_phase_summary(sr, somega, status, metrics)
    figure = plot_results(sr, somega, status, metrics, use_kahan=use_kahan)

    if SAVE_DATA_PATH is not None:
        data_path = Path(SAVE_DATA_PATH)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_path, sr=sr, somega=somega, status=status, metrics=metrics,
            metric_names=np.asarray(METRIC_NAMES),
            kahan=np.asarray(use_kahan),
        )
        print(f"Saved data to {data_path.resolve()}")
    if SAVE_PATH is not None:
        save_path = image_output_path(SAVE_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=SAVE_DPI)
        print(f"Saved figure to {save_path.resolve()}")
    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ordinary",
        action="store_false",
        dest="use_kahan",
        help="use ordinary FP32 instead of the default compensated FP32 state",
    )
    main(use_kahan=parser.parse_args().use_kahan)
