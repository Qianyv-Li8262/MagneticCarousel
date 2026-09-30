# SPDX-License-Identifier: AGPL-3.0-only
"""Initial (v0, S0) basin scan of the 2D model; --cpu or --gpu.

At x=N/2, y=0: vy=(N/2)*v0 and omega_x=(N/2)*(S0-v0)/alpha.
S0 is the initial tangential contact slip in the 1D-normalized units.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

import scan_sr_somega_2d_gpu as reference
from cuda_build import load_cuda_artifact
from project_paths import BUILD_DIR, RUNS_DIR, SCRATCH_DIR, extension_dir, image_output_path

def settings(args):
    if not (1 <= args.k <= 46340):
        raise ValueError("k must be between 1 and 46340")
    values = np.asarray((args.v0_lower, args.v0_upper, args.s0_lower, args.s0_upper,
                         args.sr, args.somega, args.dt, args.total_time,
                         args.observation_window, args.sample_time), dtype=float)
    if not np.all(np.isfinite(values)) or args.v0_lower > args.v0_upper or args.s0_lower > args.s0_upper:
        raise ValueError("bounds and time settings must be finite and ordered")
    if min(args.sr, args.somega, args.dt, args.total_time,
           args.observation_window, args.sample_time) <= 0:
        raise ValueError("scales and time settings must be positive")
    if args.observation_window > args.total_time:
        raise ValueError("observation window exceeds total time")
    steps = round(args.total_time / args.dt)
    window = round(args.observation_window / args.dt)
    stride = round(args.sample_time / args.dt)
    if steps > np.iinfo(np.int32).max or window > steps or window < 2 * stride or stride < 1:
        raise ValueError("invalid integer step/window/sample counts")
    if args.threads < 0 or args.field_mode not in ("exact", "lut") or args.lut_cache_mode not in (0, 1, 2):
        raise ValueError("invalid worker or field settings")
    return steps, window, stride


def run_gpu(args, v0, s0, steps, window, stride):
    cp = reference._get_cupy()
    dtype = np.dtype(v0.dtype)
    cubin, manifest = load_cuda_artifact(
        "initial", "fp64" if args.fp64 else "fp32_kahan", args.field_mode, args.lut_cache_mode,
    )
    lut = reference._load_lut(cp, dtype, manifest) if args.field_mode == "lut" else np.uintp(0)
    module = cp.RawModule(path=str(cubin))
    kernel = module.get_function("scan_initial_cuda")
    k = args.k
    states_gpu = cp.zeros((k * k, len(reference.STATE_NAMES)), dtype=dtype)
    status_gpu = cp.full(k * k, -2, dtype=cp.int32)
    metrics_gpu = cp.full((k * k, len(reference.METRIC_NAMES)), cp.nan, dtype=dtype)
    params = (dtype.type(args.sr), dtype.type(args.somega),
              dtype.type(reference.REFERENCE_ALPHA), dtype.type(reference.REFERENCE_DELTA),
              dtype.type(reference.REFERENCE_GAMMA_G), dtype.type(reference.REFERENCE_GAMMA),
              dtype.type(reference.REFERENCE_GAMMA_M))
    block = (reference.BLOCK_X, reference.BLOCK_Y, 1)
    grid = ((k + block[0] - 1) // block[0], (k + block[1] - 1) // block[1], 1)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    kernel(grid, block, (cp.asarray(v0), cp.asarray(s0), np.int32(k), dtype.type(args.dt),
                         np.int32(steps), np.int32(window), np.int32(stride), *params, lut,
                         states_gpu, status_gpu, metrics_gpu))
    stop.record()
    stop.synchronize()
    print(f"GPU kernel: {cp.cuda.get_elapsed_time(start, stop) / 1000:.3f} s", flush=True)
    return (cp.asnumpy(status_gpu).reshape(k, k),
            cp.asnumpy(metrics_gpu).reshape(k, k, len(reference.METRIC_NAMES)),
            cp.asnumpy(states_gpu).reshape(k, k, len(reference.STATE_NAMES)))


def run_cpu(args, v0, s0, steps, window, stride):
    kind = "fp64" if args.fp64 else "fp32-kahan"
    directory = extension_dir(kind)
    manifest_path = directory / "build_config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CPU module is not built: {manifest_path}; build hysteresis_{kind.replace('-', '_')} with CMake")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("precision") != kind or manifest.get("field_mode") != args.field_mode:
        raise RuntimeError("CPU module precision or field mode differs from this scan; reconfigure/rebuild src2d")
    if args.field_mode == "lut" and manifest.get("lut_cache_mode") != args.lut_cache_mode:
        raise RuntimeError("CPU LUT cache mode differs from this scan; reconfigure/rebuild src2d")
    binary = directory / manifest["binary"]
    if not binary.is_file():
        raise FileNotFoundError(f"CPU module is missing: {binary}")
    spec = importlib.util.spec_from_file_location("hysteresis", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CPU module: {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "scan_initial_grid"):
        raise RuntimeError(f"CPU module lacks scan_initial_grid; rebuild {binary}")
    if np.dtype(module.real_dtype) != v0.dtype or int(module.magnet_number) != reference.MAGNET_NUMBER:
        raise RuntimeError("CPU module precision or magnet count differs from this scan")
    parameters = module.SolverParams(reference.REFERENCE_ALPHA, reference.REFERENCE_DELTA,
                                     reference.REFERENCE_GAMMA_G, reference.REFERENCE_GAMMA,
                                     reference.REFERENCE_GAMMA_M)
    start = time.perf_counter()
    result = module.scan_initial_grid(v0, s0, args.dt, steps, window, stride,
                                      args.sr, args.somega, parameters, args.threads)
    print(f"CPU worker pool: {time.perf_counter() - start:.3f} s", flush=True)
    return result


def plot(v0, s0, status, metrics, args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    valid = status == 0
    metric = lambda name: np.ma.masked_where(~valid, metrics[..., reference.METRIC_INDEX[name]])
    phase, velocity, radius = metric("mean_phase_rate"), metric("mean_v"), metric("mean_radius")
    speed = (np.abs(metrics[..., reference.METRIC_INDEX["mean_v"]] - 1.0) <= reference.SPEED_ONE_THRESHOLD).astype(np.int8)
    speed[metrics[..., reference.METRIC_INDEX["mean_v"]] < 0] = 2
    speed = np.ma.masked_where(~valid, speed)
    radius_mask = np.ma.masked_where(~valid,
        (np.abs(metrics[..., reference.METRIC_INDEX["mean_radius"]] - reference.MAGNET_RADIUS)
         <= reference.RADIUS_TOLERANCE).astype(np.int8))
    edges_x, edges_y = reference._cell_edges(v0), reference._cell_edges(s0)
    fig, grid = plt.subplots(2, 3, figsize=(19, 10.2), constrained_layout=True)
    axes = grid.ravel()
    def panel(i, data, title, label, cmap, *, vmin=None, vmax=None, norm=None, ticks=None):
        colors = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
        colors.set_bad("0.65")
        im = axes[i].pcolormesh(edges_x, edges_y, data, shading="flat", cmap=colors,
                                vmin=vmin, vmax=vmax, norm=norm)
        axes[i].set_title(title)
        bar = fig.colorbar(im, ax=axes[i], label=label, ticks=ticks)
        return bar
    panel(0, np.clip(phase, 0, 1), r"Order parameter $\langle\dot\psi\rangle$ (clipped)",
          r"$\langle\dot\psi\rangle$ (clipped)", "magma", vmin=0, vmax=1)
    panel(1, phase, r"Order parameter $\langle\dot\psi\rangle$ (unclipped)",
          r"$\langle\dot\psi\rangle$", "magma")
    panel(2, velocity, "Terminal mean velocity in 1D units",
          r"$\langle v_{\rm tan}/m\rangle$", "viridis")
    cmap = ListedColormap(("green", "red", "blue"))
    bar = panel(3, speed, "Terminal mean velocity mask", "", cmap,
                norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5), 3), ticks=(0, 1, 2))
    bar.ax.set_yticklabels(("other speed", "v near 1", "mean v < 0"))
    panel(4, radius, r"Terminal mean radius $\langle r\rangle$", r"$\langle r\rangle$", "viridis")
    cmap = ListedColormap(("tab:red", "tab:green"))
    bar = panel(5, radius_mask, "Radius mask", "", cmap,
                norm=BoundaryNorm((-0.5, 0.5, 1.5), 2), ticks=(0, 1))
    bar.ax.set_yticklabels(("outside", "within"))
    for axis in axes:
        axis.set(xlabel=r"initial $v_0=v_{{\rm tan},0}/m$",
                 ylabel=r"initial slip $S_0=v_0+\alpha\omega_0$")
        axis.grid(alpha=0.2)
    fig.suptitle(f"2D initial-state scan · sr={args.sr:g} · sω={args.somega:g} · "
                 f"K={args.k} · T={args.total_time:g} · {args.cpu and 'CPU' or 'GPU'} · "
                 f"{'FP64' if args.fp64 else 'Kahan FP32'}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", action="store_true")
    device.add_argument("--gpu", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="CPU workers; 0 uses hardware concurrency")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v0-lower", type=float, default=-1.0)
    parser.add_argument("--v0-upper", type=float, default=1.0)
    parser.add_argument("--s0-lower", type=float, default=-14.0)
    parser.add_argument("--s0-upper", type=float, default=14.0)
    parser.add_argument("--sr", type=float, default=2.5)
    parser.add_argument("--somega", type=float, default=2.25447)
    parser.add_argument("--dt", type=float, default=reference.DT)
    parser.add_argument("--total-time", type=float, default=reference.TOTAL_TIME)
    parser.add_argument("--observation-window", type=float, default=reference.OBSERVATION_WINDOW)
    parser.add_argument("--sample-time", type=float, default=reference.SAMPLE_TIME)
    parser.add_argument("--field-mode", choices=("exact", "lut"), default="exact")
    parser.add_argument("--lut-cache-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--save-data", type=Path)
    parser.add_argument("--save-figure", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    steps, window, stride = settings(args)
    dtype = np.float64 if args.fp64 else np.float32
    v0 = np.linspace(args.v0_lower, args.v0_upper, args.k, dtype=dtype)
    s0 = np.linspace(args.s0_lower, args.s0_upper, args.k, dtype=dtype)
    print(f"{args.cpu and 'CPU' or 'GPU'} initial scan: {args.k}x{args.k} points, {steps} steps, "
          f"{window} window steps, {stride} sample stride", flush=True)
    status, metrics, states = (run_cpu if args.cpu else run_gpu)(args, v0, s0, steps, window, stride)
    valid = status == 0
    if np.any(valid & ~np.all(np.isfinite(metrics), axis=-1)) or np.any(valid & ~np.all(np.isfinite(states), axis=-1)):
        raise RuntimeError("successful trajectories contain non-finite results")
    print(f"Successful trajectories: {np.count_nonzero(valid)}/{args.k*args.k}", flush=True)
    run_tag = f"initial_2d_{'cpu' if args.cpu else 'gpu'}_{'fp64' if args.fp64 else 'kahan'}"
    data_path = args.save_data or RUNS_DIR / "phase-behavior" / f"{run_tag}.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, v0=v0, s0=s0, status=status, metrics=metrics, states=states,
                        metric_names=np.asarray(reference.METRIC_NAMES), state_names=np.asarray(reference.STATE_NAMES),
                        sr=args.sr, somega=args.somega, dt=args.dt, steps=steps,
                        window_steps=window, sample_stride=stride,
                        fp64=args.fp64, field_mode=args.field_mode)
    figure = plot(v0, s0, status, metrics, args)
    figure_path = image_output_path(args.save_figure or SCRATCH_DIR / "phase-behavior" / f"{run_tag}.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=reference.SAVE_DPI)
    print(f"Saved {data_path} and {figure_path}", flush=True)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
