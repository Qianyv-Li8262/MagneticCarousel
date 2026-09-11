from __future__ import annotations

from pathlib import Path

import numpy as np


# ============================= Run settings =============================
# Edit these values, then run this file directly.
ALPHA_LOWER = 0.8
ALPHA_UPPER = 1.0
OMEGA_LOWER = 0.5
OMEGA_UPPER = 1.5
K = 256
TOTAL_TIME = 600.0
DT = 1.0e-3

BLOCK_X = 16
BLOCK_Y = 16

SHOW_PLOTS = True
SAVE_DIR: str | Path | None = None
SAVE_DPI = 180

# GPU log settings. These reproduce solve_toymodel_animated.py exactly.
GPU_TRACE_ENABLED = True
TRACE_ALPHA = 0.918
TRACE_DRIVE_OMEGA = 1.0
TRACE_INITIAL_STATE = (0.0, 0.5, 0.0, 0.0)
TRACE_EVERY = 1000
TRACE_CAPACITY = 4096
# ========================================================================


CUDA_SOURCE = Path(__file__).with_name("solve_toymodel_phase_core.cu")
_CUDA_MODULE = None


def _get_cuda_module(cp):
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        _CUDA_MODULE = cp.RawModule(
            code=CUDA_SOURCE.read_text(encoding="utf-8"),
            options=("--std=c++17",),
            name_expressions=("Evolve", "TraceEvolve"),
        )
    return _CUDA_MODULE


def _validate_inputs(
    alpha_lower: float,
    alpha_upper: float,
    omega_lower: float,
    omega_upper: float,
    k: int,
    total_time: float,
    dt: float,
) -> None:
    values = np.asarray(
        [alpha_lower, alpha_upper, omega_lower, omega_upper,
         total_time, dt],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("All scalar inputs must be finite FP32 values.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > 46340:
        raise ValueError("k is too large for the kernel's int32 flattened index.")
    if k > 1 and alpha_lower == alpha_upper:
        raise ValueError("alpha_lower and alpha_upper must differ when k > 1.")
    if k > 1 and omega_lower == omega_upper:
        raise ValueError("omega_lower and omega_upper must differ when k > 1.")
    if total_time < 0.0:
        raise ValueError("total_time must be nonnegative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if alpha_lower == 0.0 or alpha_upper == 0.0:
        raise ValueError("alpha cannot be zero because the equations contain 1/alpha.")
    if min(alpha_lower, alpha_upper) < 0.0 < max(alpha_lower, alpha_upper):
        raise ValueError("The alpha interval cannot cross zero.")
    if omega_lower == 0.0 or omega_upper == 0.0:
        raise ValueError("omega cannot be zero because epsilon and gamma scale as 1/omega^2.")
    if min(omega_lower, omega_upper) < 0.0 < max(omega_lower, omega_upper):
        raise ValueError("The omega interval cannot cross zero.")


def run_sweep(
    alpha_lower: float,
    alpha_upper: float,
    omega_lower: float,
    omega_upper: float,
    k: int,
    total_time: float,
    dt: float,
    block_shape: tuple[int, int] = (BLOCK_X, BLOCK_Y),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch Evolve and return three CPU arrays with shape (omega, alpha)."""
    _validate_inputs(
        alpha_lower, alpha_upper,
        omega_lower, omega_upper,
        k, total_time, dt,
    )

    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuPy is not installed. Install a CuPy build compatible with your "
            "CUDA toolkit before running this scheduler."
        ) from exc

    block_x, block_y = block_shape
    if block_x <= 0 or block_y <= 0 or block_x * block_y > 1024:
        raise ValueError("block_shape must contain positive values with at most 1024 threads.")

    module = _get_cuda_module(cp)
    evolve_kernel = module.get_function("Evolve")

    # Rows are driving omega; columns are alpha.
    stick300_gpu = cp.empty((k, k), dtype=cp.int32)
    x_sign_gpu = cp.zeros((k, k), dtype=cp.int32)
    final_v_gpu = cp.full((k, k), cp.nan, dtype=cp.float32)

    block = (block_x, block_y, 1)
    grid = (
        (k + block_x - 1) // block_x,
        (k + block_y - 1) // block_y,
        1,
    )
    evolve_kernel(
        grid,
        block,
        (
            np.float32(alpha_lower),
            np.float32(alpha_upper),
            np.int32(k),
            np.float32(omega_lower),
            np.float32(omega_upper),
            np.float32(total_time),
            np.float32(dt),
            stick300_gpu,
            x_sign_gpu,
            final_v_gpu,
        ),
    )
    print("CUDA kernel 已启动。", flush=True)

    cp.cuda.runtime.deviceSynchronize()
    stick300 = cp.asnumpy(stick300_gpu)
    x_sign = cp.asnumpy(x_sign_gpu)
    final_v = cp.asnumpy(final_v_gpu)
    bad_velocity = (stick300 != -1) & ~np.isfinite(final_v)
    if np.any(bad_velocity):
        bad_count = int(np.count_nonzero(bad_velocity))
        raise RuntimeError(
            f"CUDA returned {bad_count} non-finite velocities at points "
            "which were not marked as negative-normal-force exits."
        )
    return stick300, x_sign, final_v


def run_gpu_trace(
    alpha: float,
    drive_omega: float,
    initial_state: tuple[float, float, float, float],
    total_time: float,
    dt: float,
    log_every: int,
    capacity: int,
) -> None:
    """Run one GPU thread and print comparable periodic/event trace records."""
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuPy is not installed. Install a CuPy build compatible with your "
            "CUDA toolkit before running this scheduler."
        ) from exc

    if drive_omega == 0.0:
        raise ValueError("TRACE_DRIVE_OMEGA cannot be zero.")
    if log_every <= 0 or capacity <= 0:
        raise ValueError("TRACE_EVERY and TRACE_CAPACITY must be positive.")

    module = _get_cuda_module(cp)
    trace_kernel = module.get_function("TraceEvolve")
    log_steps_gpu = cp.empty(capacity, dtype=cp.int32)
    log_events_gpu = cp.empty(capacity, dtype=cp.int32)
    log_modes_gpu = cp.empty(capacity, dtype=cp.int32)
    log_values_gpu = cp.empty((capacity, 10), dtype=cp.float32)
    log_count_gpu = cp.zeros(1, dtype=cp.int32)

    trace_kernel(
        (1, 1, 1),
        (1, 1, 1),
        (
            np.float32(alpha),
            np.float32(drive_omega),
            np.float32(total_time),
            np.float32(dt),
            *(np.float32(value) for value in initial_state),
            np.int32(log_every),
            np.int32(capacity),
            log_steps_gpu,
            log_events_gpu,
            log_modes_gpu,
            log_values_gpu,
            log_count_gpu,
        ),
    )
    print("GPU trace kernel 已启动。", flush=True)
    cp.cuda.runtime.deviceSynchronize()

    count = int(cp.asnumpy(log_count_gpu)[0])
    steps = cp.asnumpy(log_steps_gpu[:count])
    events = cp.asnumpy(log_events_gpu[:count])
    modes = cp.asnumpy(log_modes_gpu[:count])
    values = cp.asnumpy(log_values_gpu[:count])
    event_names = {
        0: "PERIODIC",
        1: "STICK_TO_SLIP",
        2: "SLIP_TO_STICK",
        3: "SLIP_REVERSE",
        4: "NEGATIVE_NORMAL",
    }

    epsilon = np.float32(50.0) / np.float32(drive_omega * drive_omega)
    gamma = np.float32(2.5) / np.float32(drive_omega * drive_omega)
    # print(
    #     "GPU_TRACE_CONFIG"
    #     f"\talpha={np.float32(alpha):.9e}"
    #     f"\tdrive_omega={np.float32(drive_omega):.9e}"
    #     f"\tgamma={gamma:.9e}"
    #     f"\tepsilon={epsilon:.9e}"
    #     f"\tdelta={np.float32(6.4e-5):.9e}",
    #     flush=True,
    # )
    for step, event, mode, row in zip(steps, events, modes, values, strict=True):
        time, x, v, theta, state_omega, phase, normal, slip_speed, friction, slip_sign = row
        # print(
        #     f"GPU_TRACE\tstep={int(step)}\tevent={event_names[int(event)]}"
        #     f"\tt={time:.9e}\tmode={'SLIP' if mode else 'STICK'}"
        #     f"\tx={x:.9e}\tv={v:.9e}\ttheta={theta:.9e}"
        #     f"\tstate_omega={state_omega:.9e}\tphase={phase:.9e}"
        #     f"\tN={normal:.9e}\tS={slip_speed:.9e}"
        #     f"\tF_req={friction:.9e}\tslip_sign={slip_sign:+.0f}",
        #     flush=True,
        # )


def _cell_edges(points: np.ndarray) -> np.ndarray:
    if points.size == 1:
        half_width = max(0.5, 0.05 * abs(float(points[0])))
        return np.asarray(
            [points[0] - half_width, points[0] + half_width],
            dtype=np.float32,
        )

    midpoints = 0.5 * (points[:-1] + points[1:])
    return np.concatenate((
        [points[0] - (midpoints[0] - points[0])],
        midpoints,
        [points[-1] + (points[-1] - midpoints[-1])],
    ))


def plot_results(
    stick300: np.ndarray,
    x_sign: np.ndarray,
    final_v: np.ndarray,
    alpha_lower: float,
    alpha_upper: float,
    omega_lower: float,
    omega_upper: float,
):
    """Create three figures; alpha is horizontal and omega is vertical."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if stick300.shape != x_sign.shape or stick300.shape != final_v.shape:
        raise ValueError("All three output arrays must have the same shape.")
    if stick300.ndim != 2 or stick300.shape[0] != stick300.shape[1]:
        raise ValueError("Output arrays must have shape (k, k).")

    k = stick300.shape[0]
    alpha = np.linspace(alpha_lower, alpha_upper, k, dtype=np.float32)
    omega = np.linspace(omega_lower, omega_upper, k, dtype=np.float32)
    alpha_edges = _cell_edges(alpha)
    omega_edges = _cell_edges(omega)
    category_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)

    stick_cmap = ListedColormap(["red", "black", "green"])
    fig_stick, ax_stick = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    image_stick = ax_stick.pcolormesh(
        alpha_edges,
        omega_edges,
        stick300,
        cmap=stick_cmap,
        norm=category_norm,
        shading="flat",
    )
    colorbar_stick = fig_stick.colorbar(image_stick, ax=ax_stick, ticks=[-1, 0, 1])
    colorbar_stick.ax.set_yticklabels(["N < 0", "not all stick", "all stick"])
    ax_stick.set_title("Stick state during the final 100 steps")
    ax_stick.set_xlabel("alpha")
    ax_stick.set_ylabel("omega")

    invalid = stick300 == -1
    x_sign_masked = np.ma.masked_where(invalid, x_sign)
    x_cmap = ListedColormap(["green", "red", "blue"])
    x_cmap.set_bad("0.75")
    fig_x, ax_x = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    image_x = ax_x.pcolormesh(
        alpha_edges,
        omega_edges,
        x_sign_masked,
        cmap=x_cmap,
        norm=category_norm,
        shading="flat",
    )
    colorbar_x = fig_x.colorbar(image_x, ax=ax_x, ticks=[-1, 0, 1])
    colorbar_x.ax.set_yticklabels(["x < 0", "x = 0", "x > 0"])
    ax_x.set_title("Sign of final x")
    ax_x.set_xlabel("alpha")
    ax_x.set_ylabel("omega")

    from matplotlib.colors import ListedColormap

    final_v_masked = np.ma.masked_where(invalid, final_v)

    velocity_cmap = plt.get_cmap("turbo").copy()
    velocity_cmap.set_bad("0.75")

    fig_v, ax_v = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    image_v = ax_v.pcolormesh(
    alpha_edges,
    omega_edges,
    final_v_masked,
    cmap=velocity_cmap,
    shading="flat",
    )

    near_zero = (~invalid) & (np.abs(final_v) <= 7.5)
    red_mask = np.ma.masked_where(~near_zero, np.ones_like(final_v))

    ax_v.pcolormesh(
    alpha_edges,
    omega_edges,
    red_mask,
    cmap=ListedColormap(["black"]),
    shading="flat",
    )

    colorbar_v = fig_v.colorbar(image_v, ax=ax_v)
    colorbar_v.set_label("final v")

    ax_v.set_title("Final center-of-mass velocity")
    ax_v.set_xlabel("alpha")
    ax_v.set_ylabel("omega")

    return fig_stick, fig_x, fig_v


def main() -> None:
    if GPU_TRACE_ENABLED:
        run_gpu_trace(
            TRACE_ALPHA,
            TRACE_DRIVE_OMEGA,
            TRACE_INITIAL_STATE,
            TOTAL_TIME,
            DT,
            TRACE_EVERY,
            TRACE_CAPACITY,
        )

    arrays = run_sweep(
        ALPHA_LOWER,
        ALPHA_UPPER,
        OMEGA_LOWER,
        OMEGA_UPPER,
        K,
        TOTAL_TIME,
        DT,
        block_shape=(BLOCK_X, BLOCK_Y),
    )
    figures = plot_results(
        *arrays,
        ALPHA_LOWER,
        ALPHA_UPPER,
        OMEGA_LOWER,
        OMEGA_UPPER,
    )

    if SAVE_DIR is not None:
        save_dir = Path(SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        for figure, filename in zip(
            figures,
            ("stick300.png", "x_sign.png", "final_v.png"),
            strict=True,
        ):
            figure.savefig(save_dir / filename, dpi=SAVE_DPI)

    if SHOW_PLOTS:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
