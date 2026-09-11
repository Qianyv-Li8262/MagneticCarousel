"""Scan the initial ``(v0, S0)`` plane for fixed ``sr`` and ``somega``.

Run from the repository root in an environment containing CuPy with::

    conda run -n daily python src/run_hysteresis_initial_phase.py

Here ``S0 = v0 + alpha * omega0`` is the initial contact-point slip
velocity.  The CUDA kernel recovers ``omega0`` separately at each grid point.
The five outputs and plots deliberately use the same finite-time
velocity-minimum criterion as ``run_hysteresis_phase.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from run_hysteresis_phase import (
    DEFAULT_REFERENCE,
    ReferenceParams,
    _cell_edges,
    _get_cuda_module,
    _get_cupy,
    _symmetric_limit,
)


# ============================= Run settings =============================
# Fixed physical-parameter ratios.  This is the point where the two tested
# initial conditions were observed to approach different long-time states.
SR = 0.8
SOMEGA = 0.6

# Initial-condition plane.  S0 = v0 + alpha * omega0.
V0_LOWER = -0.4
V0_UPPER = 0.4
S0_LOWER = -0.4
S0_UPPER = 0.4

K = 256
DT = 1.0e-3
TOTAL_TIME = 40000.0

BLOCK_X = 16
BLOCK_Y = 16

SHOW_PLOT = True
SAVE_PATH: str | Path | None = None
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
    reference: ReferenceParams,
    block_shape: tuple[int, int],
) -> None:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the original five diagnostics with shape ``(S0, v0)``."""
    _validate_inputs(
        sr,
        somega,
        v0_lower,
        v0_upper,
        s0_lower,
        s0_upper,
        k,
        dt,
        total_time,
        reference,
        block_shape,
    )

    cp = _get_cupy()
    print("Loading CUDA module; the first run may spend several minutes compiling it...", flush=True)
    kernel = _get_cuda_module(cp).get_function("EvolveHysteresisInitialGrid")

    # Rows are S0 and columns are v0, matching the kernel's flat index.
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
            np.float32(sr),
            np.float32(somega),
            np.float32(v0_lower),
            np.float32(v0_upper),
            np.float32(s0_lower),
            np.float32(s0_upper),
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
        f"CUDA initial-state sweep finished in {elapsed_seconds:.3f} s: "
        f"{success_count}/{k * k} points succeeded.",
        flush=True,
    )
    return status, recovered, minimum_x, minimum_step, minimum_velocity


def plot_results(
    status: np.ndarray,
    recovered: np.ndarray,
    minimum_x: np.ndarray,
    minimum_step: np.ndarray,
    minimum_velocity: np.ndarray,
    v0_lower: float,
    v0_upper: float,
    s0_lower: float,
    s0_upper: float,
    sr: float,
    somega: float,
):
    """Create the original five panels on the initial-condition plane."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    arrays = (status, recovered, minimum_x, minimum_step, minimum_velocity)
    if any(array.shape != status.shape for array in arrays[1:]):
        raise ValueError("All five output arrays must have the same shape.")
    if status.ndim != 2 or status.shape[0] != status.shape[1]:
        raise ValueError("Output arrays must have shape (k, k).")

    k = status.shape[0]
    v0 = np.linspace(v0_lower, v0_upper, k, dtype=np.float32)
    s0 = np.linspace(s0_lower, s0_upper, k, dtype=np.float32)
    v0_edges = _cell_edges(v0)
    s0_edges = _cell_edges(s0)
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
        v0_edges, s0_edges, status, cmap=binary_cmap, norm=binary_norm, shading="flat"
    )
    status_bar = figure.colorbar(status_image, ax=axes[0], ticks=(0, 1))
    status_bar.ax.set_yticklabels(("success", "failed"))
    axes[0].set_title("Evolution status")

    recovered_cmap = binary_cmap.copy()
    recovered_cmap.set_bad("0.65")
    recovered_image = axes[1].pcolormesh(
        v0_edges,
        s0_edges,
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
        v0_edges,
        s0_edges,
        minimum_x_masked,
        cmap=state_cmap,
        norm=Normalize(vmin=-x_limit, vmax=x_limit),
        shading="flat",
    )
    figure.colorbar(x_image, ax=axes[2], label=r"$x$ at minimum $v$")
    axes[2].set_title("Position at minimum velocity")

    step_cmap = plt.get_cmap("viridis").copy()
    step_cmap.set_bad("0.65")
    step_image = axes[3].pcolormesh(
        v0_edges, s0_edges, minimum_step_masked, cmap=step_cmap, shading="flat"
    )
    figure.colorbar(step_image, ax=axes[3], label="completed step")
    axes[3].set_title("Step of minimum velocity")

    velocity_image = axes[4].pcolormesh(
        v0_edges,
        s0_edges,
        minimum_velocity_masked,
        cmap=state_cmap,
        norm=Normalize(vmin=-velocity_limit, vmax=velocity_limit),
        shading="flat",
    )
    figure.colorbar(velocity_image, ax=axes[4], label=r"minimum $v$")
    axes[4].set_title("Minimum center-of-mass velocity")

    for axis in axes[:5]:
        axis.set_xlabel(r"initial $v_0$")
        axis.set_ylabel(r"initial slip $S_0=v_0+\alpha\omega_0$")

    figure.suptitle(
        rf"Initial-state velocity-minimum sweep: $s_r={sr:g}$, "
        rf"$s_\omega={somega:g}$, $k={k}$ ({k * k} trajectories)"
    )
    return figure


def main() -> None:
    arrays = run_sweep(
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
    )
    figure = plot_results(
        *arrays,
        V0_LOWER,
        V0_UPPER,
        S0_LOWER,
        S0_UPPER,
        SR,
        SOMEGA,
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
