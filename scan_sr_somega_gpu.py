"""Run the hysteresis ``sr``/``somega`` sweep on CUDA and plot its velocity minimum.

Run from the repository root in an environment containing CuPy with::

    conda run -n daily python src/run_hysteresis_phase.py

CuPy compiles ``1dmodel-hysteresis.cu`` on first use and manages all device
allocations. The five reference parameters below are kernel arguments, not
CUDA compile-time constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================= Run settings =============================
SR_LOWER = 0.01
SR_UPPER = 5.0
SOMEGA_LOWER = 0.01
SOMEGA_UPPER = 5.0
K = 128
DT = 1.0e-3
TOTAL_TIME = 4000.0

BLOCK_X = 16
BLOCK_Y = 16

SHOW_PLOT = True
SAVE_PATH: str | Path | None = None
SAVE_DPI = 180

# Same reference point as animate_hysteresis_cpu.py.
REFERENCE_ALPHA = 0.33333333
REFERENCE_DELTA = 1.7212357e-4
REFERENCE_GAMMA_G = 0.16560356
REFERENCE_GAMMA = 489.39639
REFERENCE_GAMMA_M = 11011.419
# ========================================================================


CUDA_SOURCE = Path(__file__).with_name("1dmodel-hysteresis.cu")
CUDA_HEADER = Path(__file__).with_name("1dmodel-hysteresis.hpp")
_CUDA_MODULE = None


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


def _get_cuda_module(cp):
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        if not CUDA_HEADER.is_file():
            raise FileNotFoundError(f"CUDA header not found: {CUDA_HEADER}")
        header = CUDA_HEADER.read_text(encoding="utf-8")
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        include_line = '#include "1dmodel-hysteresis.hpp"'
        if include_line not in source:
            raise RuntimeError(f"Expected {include_line!r} in {CUDA_SOURCE}")
        # Embedding the shared header makes CuPy's cache depend on its contents
        # and avoids platform-specific include-path quoting problems.
        source = source.replace(include_line, header, 1)
        _CUDA_MODULE = cp.RawModule(
            code=source,
            options=("--std=c++17",),
            backend="nvrtc",
        )
    return _CUDA_MODULE


def _validate_inputs(
    sr_lower: float,
    sr_upper: float,
    somega_lower: float,
    somega_upper: float,
    k: int,
    dt: float,
    total_time: float,
    reference: ReferenceParams,
    block_shape: tuple[int, int],
) -> None:
    scalars = np.asarray(
        (
            sr_lower,
            sr_upper,
            somega_lower,
            somega_upper,
            dt,
            total_time,
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
    if sr_lower <= 0.0 or sr_upper <= 0.0:
        raise ValueError("sr bounds must be positive.")
    if somega_lower <= 0.0 or somega_upper <= 0.0:
        raise ValueError("somega bounds must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > 46340:
        raise ValueError("k is too large for the kernel's int32 flattened index.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if total_time < 0.0:
        raise ValueError("total_time must be nonnegative.")
    if reference.alpha <= 0.0:
        raise ValueError("reference.alpha must be positive.")
    if reference.delta < 0.0:
        raise ValueError("reference.delta must be nonnegative.")

    block_x, block_y = block_shape
    if block_x <= 0 or block_y <= 0 or block_x * block_y > 1024:
        raise ValueError("block_shape must be positive and contain at most 1024 threads.")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return five CPU arrays, each with shape ``(somega, sr)``.

    The arrays are status, whether velocity recovered after its global signed
    minimum, x at that minimum, its completed-step index, and the minimum
    velocity. Failed points retain zero/NaN/-1 sentinels outside status.
    """
    _validate_inputs(
        sr_lower,
        sr_upper,
        somega_lower,
        somega_upper,
        k,
        dt,
        total_time,
        reference,
        block_shape,
    )
    cp = _get_cupy()
    kernel = _get_cuda_module(cp).get_function("EvolveHysteresisGrid")

    # Rows are somega and columns are sr, matching the kernel's flat index.
    status_gpu = cp.ones((k, k), dtype=cp.int32)
    recovered_gpu = cp.zeros((k, k), dtype=cp.int32)
    minimum_x_gpu = cp.full((k, k), cp.nan, dtype=cp.float32)
    minimum_step_gpu = cp.full((k, k), -1, dtype=cp.int32)
    minimum_velocity_gpu = cp.full((k, k), cp.nan, dtype=cp.float32)

    block_x, block_y = block_shape
    block = (block_x, block_y, 1)
    grid = (
        (k + block_x - 1) // block_x,
        (k + block_y - 1) // block_y,
        1,
    )

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    kernel(
        grid,
        block,
        (
            np.float32(sr_lower),
            np.float32(sr_upper),
            np.float32(somega_lower),
            np.float32(somega_upper),
            np.int32(k),
            np.float32(dt),
            np.float32(total_time),
            np.float32(reference.alpha),
            np.float32(reference.delta),
            np.float32(reference.gamma_g),
            np.float32(reference.gamma),
            np.float32(reference.gamma_m),
            status_gpu,
            recovered_gpu,
            minimum_x_gpu,
            minimum_step_gpu,
            minimum_velocity_gpu,
        ),
    )
    stop.record()
    stop.synchronize()
    elapsed_seconds = cp.cuda.get_elapsed_time(start, stop) / 1000.0

    status = cp.asnumpy(status_gpu)
    recovered = cp.asnumpy(recovered_gpu)
    minimum_x = cp.asnumpy(minimum_x_gpu)
    minimum_step = cp.asnumpy(minimum_step_gpu)
    minimum_velocity = cp.asnumpy(minimum_velocity_gpu)

    if not np.all(np.isin(status, (0, 1))):
        raise RuntimeError("CUDA returned a status value other than zero or one.")
    succeeded = status == 0
    invalid_recovered = succeeded & ~np.isin(recovered, (0, 1))
    invalid_minimum = succeeded & (
        ~np.isfinite(minimum_x) | ~np.isfinite(minimum_velocity) | (minimum_step < 0)
    )
    if np.any(invalid_recovered | invalid_minimum):
        count = int(np.count_nonzero(invalid_recovered | invalid_minimum))
        raise RuntimeError(f"CUDA returned invalid minimum data for {count} successful points.")

    success_count = int(np.count_nonzero(succeeded))
    print(
        f"CUDA sweep finished in {elapsed_seconds:.3f} s: "
        f"{success_count}/{k * k} points succeeded.",
        flush=True,
    )
    return status, recovered, minimum_x, minimum_step, minimum_velocity


def _cell_edges(points: np.ndarray) -> np.ndarray:
    if points.size == 1:
        half_width = max(0.05 * abs(float(points[0])), 0.05)
        return np.asarray((points[0] - half_width, points[0] + half_width))

    midpoints = 0.5 * (points[:-1] + points[1:])
    return np.concatenate(
        (
            (points[0] - (midpoints[0] - points[0]),),
            midpoints,
            (points[-1] + (points[-1] - midpoints[-1]),),
        )
    )


def _symmetric_limit(values: np.ndarray, valid: np.ndarray) -> float:
    finite_values = values[valid & np.isfinite(values)]
    if finite_values.size == 0:
        return 1.0
    limit = float(np.max(np.abs(finite_values)))
    return limit if limit > 0.0 else 1.0


def plot_results(
    status: np.ndarray,
    recovered: np.ndarray,
    minimum_x: np.ndarray,
    minimum_step: np.ndarray,
    minimum_velocity: np.ndarray,
    sr_lower: float,
    sr_upper: float,
    somega_lower: float,
    somega_upper: float,
):
    """Create five panels for status and the signed velocity minimum."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    arrays = (status, recovered, minimum_x, minimum_step, minimum_velocity)
    if any(array.shape != status.shape for array in arrays[1:]):
        raise ValueError("All five output arrays must have the same shape.")
    if status.ndim != 2 or status.shape[0] != status.shape[1]:
        raise ValueError("Output arrays must have shape (k, k).")

    k = status.shape[0]
    sr = np.linspace(sr_lower, sr_upper, k, dtype=np.float32)
    somega = np.linspace(somega_lower, somega_upper, k, dtype=np.float32)
    sr_edges = _cell_edges(sr)
    somega_edges = _cell_edges(somega)
    valid = status == 0

    recovered_masked = np.ma.masked_where(~valid, recovered)
    minimum_x_masked = np.ma.masked_where(~valid, minimum_x)
    minimum_step_masked = np.ma.masked_where(~valid, minimum_step)
    minimum_velocity_masked = np.ma.masked_where(~valid, minimum_velocity)
    x_limit = _symmetric_limit(minimum_x, valid)
    velocity_limit = _symmetric_limit(minimum_velocity, valid)

    figure, grid_axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    axes = grid_axes.ravel()
    figure.delaxes(axes[5])

    binary_cmap = ListedColormap(("white", "black"))
    binary_norm = BoundaryNorm((-0.5, 0.5, 1.5), binary_cmap.N)
    status_image = axes[0].pcolormesh(
        sr_edges,
        somega_edges,
        status,
        cmap=binary_cmap,
        norm=binary_norm,
        shading="flat",
    )
    status_bar = figure.colorbar(status_image, ax=axes[0], ticks=(0, 1))
    status_bar.ax.set_yticklabels(("success", "failed"))
    axes[0].set_title("Evolution status")

    recovered_cmap = binary_cmap.copy()
    recovered_cmap.set_bad("0.65")
    recovered_image = axes[1].pcolormesh(
        sr_edges,
        somega_edges,
        recovered_masked,
        cmap=recovered_cmap,
        norm=binary_norm,
        shading="flat",
    )
    recovered_bar = figure.colorbar(recovered_image, ax=axes[1], ticks=(0, 1))
    recovered_bar.ax.set_yticklabels(("no", "yes"))
    axes[1].set_title("Velocity recovered after minimum")

    state_cmap = plt.get_cmap("coolwarm").copy()
    state_cmap.set_bad("0.65")
    x_image = axes[2].pcolormesh(
        sr_edges,
        somega_edges,
        minimum_x_masked,
        cmap=state_cmap,
        norm=Normalize(vmin=-x_limit, vmax=x_limit),
        shading="flat",
    )
    figure.colorbar(x_image, ax=axes[2], label=r"$x$ at minimum $\dot{x}$")
    axes[2].set_title("Position at minimum velocity")

    step_cmap = plt.get_cmap("viridis").copy()
    step_cmap.set_bad("0.65")
    step_image = axes[3].pcolormesh(
        sr_edges,
        somega_edges,
        minimum_step_masked,
        cmap=step_cmap,
        shading="flat",
    )
    figure.colorbar(step_image, ax=axes[3], label="completed step")
    axes[3].set_title("Step of minimum velocity")

    velocity_image = axes[4].pcolormesh(
        sr_edges,
        somega_edges,
        minimum_velocity_masked,
        cmap=state_cmap,
        norm=Normalize(vmin=-velocity_limit, vmax=velocity_limit),
        shading="flat",
    )
    figure.colorbar(velocity_image, ax=axes[4], label=r"minimum $\dot{x}$")
    axes[4].set_title("Minimum center-of-mass velocity")

    for axis in axes[:5]:
        axis.set_xlabel(r"$s_r=r/r_{\mathrm{ref}}$")
        axis.set_ylabel(r"$s_\omega=\omega/\omega_{\mathrm{ref}}$")

    figure.suptitle(f"Hysteresis velocity-minimum sweep: k={k}, {k * k} trajectories")
    return figure


def main() -> None:
    arrays = run_sweep(
        SR_LOWER,
        SR_UPPER,
        SOMEGA_LOWER,
        SOMEGA_UPPER,
        K,
        DT,
        TOTAL_TIME,
        reference=DEFAULT_REFERENCE,
        block_shape=(BLOCK_X, BLOCK_Y),
    )
    figure = plot_results(
        *arrays,
        SR_LOWER,
        SR_UPPER,
        SOMEGA_LOWER,
        SOMEGA_UPPER,
    )

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
