# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Scan the initial ``(v0, S0)`` plane using long-time phase locking.

Run from this directory in an environment containing CuPy with::

    conda run -n daily python scan_v0_s0_gpu.py
    conda run -n daily python scan_v0_s0_gpu.py --kahan

Here ``S0 = v0 + alpha * omega0`` is the initial contact-point slip
velocity.  The CUDA kernel recovers ``omega0`` separately at each grid point
and measures the exact kinematic phase rate

    phase_rate = 1 - v - omega

over the final observation window.  A trajectory is classified as 1:1
phase-locked when ``abs(mean_phase_rate) < LOCK_THRESHOLD``.  This replaces
the old finite-time velocity-minimum/recovery proxy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project_paths import image_output_path

from scan_sr_somega_gpu import (
    DEFAULT_REFERENCE,
    METRIC_INDEX,
    METRIC_NAMES,
    ReferenceParams,
    _cell_edges,
    _get_cuda_module,
    _get_cupy,
    _symmetric_limit,
    _time_arguments,
)


# ============================= Run settings =============================
SR = 2.5
SOMEGA = 2.25447

# Initial-condition plane.  S0 = v0 + alpha * omega0.
V0_LOWER = 4.0
V0_UPPER = 4.5
S0_LOWER = -7.0
S0_UPPER = -6.5

K = 256
DT = 1.0e-3
TOTAL_TIME = 50000.0
OBSERVATION_WINDOW = 4000.0
SAMPLE_TIME = 0.2

# Validated long-time order-parameter threshold used by the (sr, somega)
# phase scan.  Locked means |<phase_rate>| < this value.
LOCK_THRESHOLD = 0.05

BLOCK_X = 16
BLOCK_Y = 16

SHOW_PLOT = True
SAVE_PATH: str | Path | None = None
SAVE_DATA_PATH: str | Path | None = None
SAVE_DPI = 180
# ========================================================================


def _validate_inputs(
    sr: float,
    somega: float,
    v0_lower: float,
    v0_upper: float,
    s0_lower: float,
    s0_upper: float,
    k: int,
    dt: float,
    total_time: float,
    observation_window: float,
    sample_time: float,
    lock_threshold: float,
    reference: ReferenceParams,
    block_shape: tuple[int, int],
    use_fp64: bool,
) -> tuple[int, int, int]:
    float_type = np.float64 if use_fp64 else np.float32
    scalars = np.asarray(
        (
            sr,
            somega,
            v0_lower,
            v0_upper,
            s0_lower,
            s0_upper,
            dt,
            total_time,
            observation_window,
            sample_time,
            lock_threshold,
            reference.alpha,
            reference.delta,
            reference.gamma_g,
            reference.gamma,
            reference.gamma_m,
        ),
        dtype=float_type,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError(f"All scalar inputs must be finite when converted to {float_type.__name__}.")
    if sr <= 0.0:
        raise ValueError("sr must be positive.")
    if somega <= 0.0:
        raise ValueError("somega must be positive.")
    if v0_lower > v0_upper:
        raise ValueError("v0_lower must not exceed v0_upper.")
    if s0_lower > s0_upper:
        raise ValueError("s0_lower must not exceed s0_upper.")
    if not 1 <= k <= 46340:
        raise ValueError("k must be between 1 and 46340.")
    if observation_window > total_time:
        raise ValueError("observation_window must not exceed total_time.")
    if lock_threshold <= 0.0:
        raise ValueError("lock_threshold must be positive.")
    if reference.alpha <= 0.0:
        raise ValueError("reference.alpha must be positive.")
    if reference.delta < 0.0:
        raise ValueError("reference.delta must be nonnegative.")

    block_x, block_y = block_shape
    if block_x <= 0 or block_y <= 0 or block_x * block_y > 1024:
        raise ValueError("block_shape must be positive and contain at most 1024 threads.")
    return _time_arguments(dt, total_time, observation_window, sample_time)


def run_sweep(
    sr: float,
    somega: float,
    v0_lower: float,
    v0_upper: float,
    s0_lower: float,
    s0_upper: float,
    k: int,
    dt: float,
    total_time: float,
    reference: ReferenceParams = DEFAULT_REFERENCE,
    block_shape: tuple[int, int] = (BLOCK_X, BLOCK_Y),
    observation_window: float = OBSERVATION_WINDOW,
    sample_time: float = SAMPLE_TIME,
    lock_threshold: float = LOCK_THRESHOLD,
    use_kahan: bool = False,
    use_fp64: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``v0``, ``S0``, status, and long-time metrics.

    ``metrics`` has shape ``(S0, v0, len(METRIC_NAMES))``.  Phase
    classification is deliberately derived from the measured order parameter
    rather than baked into the kernel, so thresholds can be checked without
    rerunning the trajectories.
    """
    steps, window_steps, sample_stride = _validate_inputs(
        sr,
        somega,
        v0_lower,
        v0_upper,
        s0_lower,
        s0_upper,
        k,
        dt,
        total_time,
        observation_window,
        sample_time,
        lock_threshold,
        reference,
        block_shape,
        use_fp64,
    )

    float_type = np.float64 if use_fp64 else np.float32
    v0 = np.linspace(v0_lower, v0_upper, k, dtype=float_type)
    s0 = np.linspace(s0_lower, s0_upper, k, dtype=float_type)

    cp = _get_cupy()
    print(
        f"Loading CUDA module ({'Kahan ' if use_kahan else 'ordinary '}{'FP64' if use_fp64 else 'FP32'}); "
        "the first run may spend several minutes compiling it...",
        flush=True,
    )
    kernel = _get_cuda_module(
        cp, use_kahan=use_kahan, use_fp64=use_fp64
    ).get_function("AnalyzeInitialGrid")
    status_gpu = cp.full((k, k), 4, dtype=cp.int32)
    metrics_gpu = cp.full((k, k, len(METRIC_NAMES)), cp.nan, dtype=float_type)

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
            float_type(sr),
            float_type(somega),
            cp.asarray(v0),
            cp.asarray(s0),
            np.int32(k),
            float_type(dt),
            np.int32(steps),
            np.int32(window_steps),
            np.int32(sample_stride),
            float_type(reference.alpha),
            float_type(reference.delta),
            float_type(reference.gamma_g),
            float_type(reference.gamma),
            float_type(reference.gamma_m),
            status_gpu,
            metrics_gpu,
        ),
    )
    stop.record()
    stop.synchronize()
    elapsed_seconds = cp.cuda.get_elapsed_time(start, stop) / 1000.0

    status = cp.asnumpy(status_gpu)
    metrics = cp.asnumpy(metrics_gpu)
    succeeded = status == 0
    invalid = succeeded & ~np.all(np.isfinite(metrics), axis=-1)
    if np.any(invalid):
        raise RuntimeError(
            f"CUDA returned non-finite phase metrics for {np.count_nonzero(invalid)} successful points."
        )

    phase_rate = metrics[..., METRIC_INDEX["mean_phase_rate"]]
    locked = succeeded & (np.abs(phase_rate) < lock_threshold)
    success_count = int(np.count_nonzero(succeeded))
    print(
        f"CUDA initial-state phase sweep finished in {elapsed_seconds:.3f} s: "
        f"{success_count}/{k * k} points succeeded; "
        f"{np.count_nonzero(locked)}/{success_count} are phase-locked "
        f"(|<phase_rate>| < {lock_threshold:g}).",
        flush=True,
    )
    return v0, s0, status, metrics


def phase_locked_mask(
    status: np.ndarray,
    metrics: np.ndarray,
    lock_threshold: float = LOCK_THRESHOLD,
) -> np.ndarray:
    """Return the validated 1:1 phase-lock classification mask."""
    if status.ndim != 2 or metrics.shape != status.shape + (len(METRIC_NAMES),):
        raise ValueError("Expected status (k, k) and metrics (k, k, len(METRIC_NAMES)).")
    return (status == 0) & (
        np.abs(metrics[..., METRIC_INDEX["mean_phase_rate"]]) < lock_threshold
    )


def plot_results(
    v0: np.ndarray,
    s0: np.ndarray,
    status: np.ndarray,
    metrics: np.ndarray,
    sr: float,
    somega: float,
    total_time: float = TOTAL_TIME,
    observation_window: float = OBSERVATION_WINDOW,
    lock_threshold: float = LOCK_THRESHOLD,
    use_kahan: bool = False,
    use_fp64: bool = False,
):
    """Plot the final-window mean and standard deviation of the phase rate."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if v0.ndim != 1 or s0.ndim != 1 or status.shape != (s0.size, v0.size):
        raise ValueError("Coordinate and status shapes are inconsistent.")
    if metrics.shape != status.shape + (len(METRIC_NAMES),):
        raise ValueError("Metrics shape is inconsistent with status.")

    valid = status == 0
    phase_rate = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["mean_phase_rate"]])
    phase_rate_std = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["std_phase_rate"]])

    v0_edges = _cell_edges(v0)
    s0_edges = _cell_edges(s0)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)

    finite_mean = phase_rate.compressed()
    order_limit = (
        float(np.nanpercentile(np.abs(finite_mean), 99.5)) if finite_mean.size else 1.0
    )
    if not np.isfinite(order_limit) or order_limit <= 0.0:
        order_limit = 1.0
    order_image = axes[0].pcolormesh(
        v0_edges,
        s0_edges,
        phase_rate,
        cmap="coolwarm",
        norm=Normalize(vmin=-order_limit, vmax=order_limit),
        shading="flat",
    )
    figure.colorbar(order_image, ax=axes[0], label=r"$\langle\dot\phi\rangle_{\rm end}$")
    axes[0].set_title(r"Final-window mean $\langle\dot\phi\rangle$")

    finite_std = phase_rate_std.compressed()
    std_limit = float(np.nanpercentile(finite_std, 99.5)) if finite_std.size else 1.0
    if not np.isfinite(std_limit) or std_limit <= 0.0:
        std_limit = 1.0
    std_image = axes[1].pcolormesh(
        v0_edges,
        s0_edges,
        phase_rate_std,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=std_limit),
        shading="flat",
    )
    figure.colorbar(std_image, ax=axes[1], label=r"$\sigma_{\dot\phi,\rm end}$")
    axes[1].set_title(r"Final-window standard deviation $\sigma_{\dot\phi}$")

    for axis in axes:
        axis.set_xlabel(r"initial $v_0$")
        axis.set_ylabel(r"initial slip $S_0=v_0+\alpha\omega_0$")
        axis.grid(alpha=0.2)

    figure.suptitle(
        rf"Initial-state phase-lock sweep: $s_r={sr:g}$, $s_\omega={somega:g}$, "
        rf"$k={v0.size}$, $T={total_time:g}$, final window $={observation_window:g}$; "
        rf"$\dot\phi=1-v-\dot\theta$; "
        f"{'Kahan ' if use_kahan else 'ordinary '}{'FP64' if use_fp64 else 'FP32'}"
    )
    return figure


def main(*, use_kahan: bool = False, use_fp64: bool = False) -> None:
    v0, s0, status, metrics = run_sweep(
        SR,
        SOMEGA,
        V0_LOWER,
        V0_UPPER,
        S0_LOWER,
        S0_UPPER,
        K,
        DT,
        TOTAL_TIME,
        reference=DEFAULT_REFERENCE,
        block_shape=(BLOCK_X, BLOCK_Y),
        observation_window=OBSERVATION_WINDOW,
        sample_time=SAMPLE_TIME,
        lock_threshold=LOCK_THRESHOLD,
        use_kahan=use_kahan,
        use_fp64=use_fp64,
    )
    figure = plot_results(
        v0,
        s0,
        status,
        metrics,
        SR,
        SOMEGA,
        total_time=TOTAL_TIME,
        observation_window=OBSERVATION_WINDOW,
        lock_threshold=LOCK_THRESHOLD,
        use_kahan=use_kahan,
        use_fp64=use_fp64,
    )

    if SAVE_DATA_PATH is not None:
        data_path = Path(SAVE_DATA_PATH)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_path,
            v0=v0,
            s0=s0,
            status=status,
            metrics=metrics,
            metric_names=np.asarray(METRIC_NAMES),
            phase_locked=phase_locked_mask(status, metrics, LOCK_THRESHOLD),
            lock_threshold=np.asarray(LOCK_THRESHOLD),
            kahan=np.asarray(use_kahan),
            fp64=np.asarray(use_fp64),
        )
        print(f"Saved data to {data_path.resolve()}", flush=True)

    if SAVE_PATH is not None:
        save_path = image_output_path(SAVE_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=SAVE_DPI)
        print(f"Saved figure to {save_path.resolve()}", flush=True)

    if SHOW_PLOT:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kahan",
        action="store_true",
        help="use compensated FP32 accumulators for persistent CUDA state",
    )
    parser.add_argument(
        "--fp64",
        action="store_true",
        help="use FP64 instead of FP32",
    )
    args = parser.parse_args()
    main(use_kahan=args.kahan and not args.fp64, use_fp64=args.fp64)
