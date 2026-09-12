"""Scan the initial ``(v0, S0)`` plane using long-time phase locking.

Run from this directory in an environment containing CuPy with::

    conda run -n daily python scan_v0_s0_gpu.py

Here ``S0 = v0 + alpha * omega0`` is the initial contact-point slip
velocity.  The CUDA kernel recovers ``omega0`` separately at each grid point
and measures the exact kinematic phase rate

    phase_rate = 1 - v - omega

over the final observation window.  A trajectory is classified as 1:1
phase-locked when ``abs(mean_phase_rate) < LOCK_THRESHOLD``.  This replaces
the old finite-time velocity-minimum/recovery proxy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
SOMEGA = 2.254

# Initial-condition plane.  S0 = v0 + alpha * omega0.
V0_LOWER = -10.0
V0_UPPER = 10.0
S0_LOWER = -10.0
S0_UPPER = 10.0

K = 64
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
) -> tuple[int, int, int]:
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
        dtype=np.float32,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("All scalar inputs must be finite when converted to float32.")
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
    )

    v0 = np.linspace(v0_lower, v0_upper, k, dtype=np.float32)
    s0 = np.linspace(s0_lower, s0_upper, k, dtype=np.float32)

    cp = _get_cupy()
    print("Loading CUDA module; the first run may spend several minutes compiling it...", flush=True)
    kernel = _get_cuda_module(cp).get_function("AnalyzeInitialGrid")
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
            np.float32(sr),
            np.float32(somega),
            cp.asarray(v0),
            cp.asarray(s0),
            np.int32(k),
            np.float32(dt),
            np.int32(steps),
            np.int32(window_steps),
            np.int32(sample_stride),
            np.float32(reference.alpha),
            np.float32(reference.delta),
            np.float32(reference.gamma_g),
            np.float32(reference.gamma),
            np.float32(reference.gamma_m),
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
):
    """Plot the phase classification and its long-time convergence checks."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    if v0.ndim != 1 or s0.ndim != 1 or status.shape != (s0.size, v0.size):
        raise ValueError("Coordinate and status shapes are inconsistent.")
    if metrics.shape != status.shape + (len(METRIC_NAMES),):
        raise ValueError("Metrics shape is inconsistent with status.")

    valid = status == 0
    locked = phase_locked_mask(status, metrics, lock_threshold)
    classification = np.ma.masked_where(~valid, np.where(locked, 0, 1))
    phase_rate = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["mean_phase_rate"]])
    phase_rate_std = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["std_phase_rate"]])
    delta_phase_rate = np.ma.masked_where(
        ~valid, metrics[..., METRIC_INDEX["delta_mean_phase_rate"]]
    )
    mean_velocity = np.ma.masked_where(~valid, metrics[..., METRIC_INDEX["mean_v"]])

    v0_edges = _cell_edges(v0)
    s0_edges = _cell_edges(s0)
    figure, grid_axes = plt.subplots(2, 3, figsize=(16.2, 9.2), constrained_layout=True)
    axes = grid_axes.ravel()
    figure.delaxes(axes[5])

    phase_cmap = ListedColormap(("#2468b4", "#d1495b"))
    phase_cmap.set_bad("0.65")
    phase_norm = BoundaryNorm((-0.5, 0.5, 1.5), phase_cmap.N)
    classification_image = axes[0].pcolormesh(
        v0_edges, s0_edges, classification, cmap=phase_cmap, norm=phase_norm, shading="flat"
    )
    classification_bar = figure.colorbar(classification_image, ax=axes[0], ticks=(0, 1))
    classification_bar.ax.set_yticklabels(("phase-locked", "running"))
    axes[0].set_title(rf"Phase class: $|\langle\dot\phi\rangle|<{lock_threshold:g}$")

    order_limit = max(1.0, _symmetric_limit(phase_rate.filled(np.nan), valid))
    order_image = axes[1].pcolormesh(
        v0_edges,
        s0_edges,
        phase_rate,
        cmap="coolwarm",
        norm=Normalize(vmin=-order_limit, vmax=order_limit),
        shading="flat",
    )
    figure.colorbar(order_image, ax=axes[1], label=r"$\langle\dot\phi\rangle$")
    axes[1].set_title(r"Lock-in order parameter $\langle1-v-\omega\rangle$")

    std_image = axes[2].pcolormesh(
        v0_edges, s0_edges, phase_rate_std, cmap="magma", shading="flat"
    )
    figure.colorbar(std_image, ax=axes[2], label=r"std$(\dot\phi)$")
    axes[2].set_title("Phase-rate fluctuations in final window")

    velocity_limit = _symmetric_limit(mean_velocity.filled(np.nan), valid)
    velocity_image = axes[3].pcolormesh(
        v0_edges,
        s0_edges,
        mean_velocity,
        cmap="coolwarm",
        norm=Normalize(vmin=-velocity_limit, vmax=velocity_limit),
        shading="flat",
    )
    figure.colorbar(velocity_image, ax=axes[3], label=r"$\langle v\rangle$")
    axes[3].set_title("Mean velocity in final window")

    drift_limit = _symmetric_limit(delta_phase_rate.filled(np.nan), valid)
    drift_image = axes[4].pcolormesh(
        v0_edges,
        s0_edges,
        delta_phase_rate,
        cmap="coolwarm",
        norm=Normalize(vmin=-drift_limit, vmax=drift_limit),
        shading="flat",
    )
    figure.colorbar(
        drift_image,
        ax=axes[4],
        label=r"$\langle\dot\phi\rangle_2-\langle\dot\phi\rangle_1$",
    )
    axes[4].set_title("Within-window convergence drift")

    for axis in axes[:5]:
        axis.set_xlabel(r"initial $v_0$")
        axis.set_ylabel(r"initial slip $S_0=v_0+\alpha\omega_0$")
        axis.grid(alpha=0.2)

    figure.suptitle(
        rf"Initial-state phase-lock sweep: $s_r={sr:g}$, $s_\omega={somega:g}$, "
        rf"$k={v0.size}$, $T={total_time:g}$, final window $={observation_window:g}$"
    )
    return figure


def main() -> None:
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
        )
        print(f"Saved data to {data_path.resolve()}", flush=True)

    if SAVE_PATH is not None:
        save_path = Path(SAVE_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=SAVE_DPI)
        print(f"Saved figure to {save_path.resolve()}", flush=True)

    if SHOW_PLOT:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
