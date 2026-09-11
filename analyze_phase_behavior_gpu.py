"""CUDA scans of long-time observables for the self-consistent hysteresis model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
import numpy as np

from scan_sr_somega_gpu import DEFAULT_REFERENCE, _cell_edges, _get_cupy


ROOT = Path(__file__).resolve().parent
CUDA_SOURCE = ROOT / "phase_behavior.cu"
CUDA_HEADER = ROOT / "1dmodel-hysteresis.hpp"
BASE_H_EXT = 1.4661546e-1
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
)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


@dataclass(frozen=True)
class ScanResult:
    x: np.ndarray
    y: np.ndarray
    status: np.ndarray
    metrics: np.ndarray
    x_name: str
    y_name: str
    metadata: dict[str, float | int | str]

    def metric(self, name: str) -> np.ndarray:
        return self.metrics[..., METRIC_INDEX[name]]


def _module(cp, *, project_sticking: bool, field_scale: float = 1.0):
    header = CUDA_HEADER.read_text(encoding="utf-8")
    if not np.isfinite(field_scale) or field_scale <= 0.0:
        raise ValueError("field_scale must be positive and finite")
    original_field = "#define H_EXT REAL_LITERAL(1.4661546e-1)"
    scaled_field = f"#define H_EXT REAL_LITERAL({BASE_H_EXT * field_scale:.9e})"
    if original_field not in header:
        raise RuntimeError("Expected H_EXT definition was not found in the shared header")
    header = header.replace(original_field, scaled_field, 1)
    if project_sticking:
        old = """            // keep sticking
            ball = trial_final;
            return 0;"""
        new = """            // keep sticking and remove accumulated floating-point constraint drift
            ball = trial_final;
            ball.theta_dot = -ball.xdot / params.alpha;
            return 0;"""
        if old not in header:
            raise RuntimeError("Expected sticking branch was not found in the shared header")
        header = header.replace(old, new, 1)
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    include = '#include "1dmodel-hysteresis.hpp"'
    if include not in source:
        raise RuntimeError(f"Expected {include!r} in {CUDA_SOURCE}")
    return cp.RawModule(
        code=source.replace(include, header, 1),
        options=("--std=c++17",),
        backend="nvrtc",
    )


def _values(lower: float, upper: float, count: int, scale: str) -> np.ndarray:
    if count < 2:
        raise ValueError("count must be at least two")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and strictly increasing")
    if scale == "log":
        if lower <= 0.0:
            raise ValueError("logarithmic lower bound must be positive")
        return np.geomspace(lower, upper, count, dtype=np.float32)
    return np.linspace(lower, upper, count, dtype=np.float32)


def _time_arguments(dt: float, total_time: float, window_time: float, sample_time: float):
    values = np.asarray((dt, total_time, window_time, sample_time), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all time settings must be positive and finite")
    steps = round(total_time / dt)
    window_steps = min(steps, round(window_time / dt))
    sample_stride = max(1, round(sample_time / dt))
    if steps > np.iinfo(np.int32).max:
        raise ValueError("step count exceeds int32")
    if window_steps < 2 * sample_stride:
        raise ValueError("observation window must contain at least two samples")
    return steps, window_steps, sample_stride


def _launch(kernel, cp, k: int, arguments: tuple):
    status_gpu = cp.full((k, k), 4, dtype=cp.int32)
    metrics_gpu = cp.full((k, k, len(METRIC_NAMES)), cp.nan, dtype=cp.float32)
    block = (16, 16, 1)
    grid = ((k + 15) // 16, (k + 15) // 16, 1)
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    kernel(grid, block, (*arguments, status_gpu, metrics_gpu))
    stop.record()
    stop.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000.0
    return cp.asnumpy(status_gpu), cp.asnumpy(metrics_gpu), elapsed


def run_parameter_scan(
    sr: np.ndarray,
    somega: np.ndarray,
    *,
    dt: float,
    total_time: float,
    window_time: float,
    sample_time: float,
    project_sticking: bool = False,
    field_scale: float = 1.0,
) -> ScanResult:
    if sr.shape != somega.shape or sr.ndim != 1:
        raise ValueError("sr and somega must be one-dimensional arrays of equal length")
    k = sr.size
    steps, window_steps, sample_stride = _time_arguments(dt, total_time, window_time, sample_time)
    cp = _get_cupy()
    kernel = _module(cp, project_sticking=project_sticking, field_scale=field_scale).get_function("AnalyzeParameterGrid")
    status, metrics, elapsed = _launch(
        kernel,
        cp,
        k,
        (
            cp.asarray(sr),
            cp.asarray(somega),
            np.int32(k),
            np.float32(dt),
            np.int32(steps),
            np.int32(window_steps),
            np.int32(sample_stride),
            np.float32(DEFAULT_REFERENCE.alpha),
            np.float32(DEFAULT_REFERENCE.delta),
            np.float32(DEFAULT_REFERENCE.gamma_g),
            np.float32(DEFAULT_REFERENCE.gamma),
            np.float32(DEFAULT_REFERENCE.gamma_m),
        ),
    )
    metadata = dict(
        kind="parameter",
        dt=dt,
        total_time=total_time,
        window_time=window_time,
        sample_time=sample_time,
        project_sticking=int(project_sticking),
        field_scale=field_scale,
        elapsed_seconds=elapsed,
    )
    return ScanResult(sr, somega, status, metrics, "sr", "somega", metadata)


def run_initial_scan(
    v0: np.ndarray,
    s0: np.ndarray,
    *,
    sr: float,
    somega: float,
    dt: float,
    total_time: float,
    window_time: float,
    sample_time: float,
    project_sticking: bool = False,
    field_scale: float = 1.0,
) -> ScanResult:
    if v0.shape != s0.shape or v0.ndim != 1:
        raise ValueError("v0 and s0 must be one-dimensional arrays of equal length")
    k = v0.size
    steps, window_steps, sample_stride = _time_arguments(dt, total_time, window_time, sample_time)
    cp = _get_cupy()
    kernel = _module(cp, project_sticking=project_sticking, field_scale=field_scale).get_function("AnalyzeInitialGrid")
    status, metrics, elapsed = _launch(
        kernel,
        cp,
        k,
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
            np.float32(DEFAULT_REFERENCE.alpha),
            np.float32(DEFAULT_REFERENCE.delta),
            np.float32(DEFAULT_REFERENCE.gamma_g),
            np.float32(DEFAULT_REFERENCE.gamma),
            np.float32(DEFAULT_REFERENCE.gamma_m),
        ),
    )
    metadata = dict(
        kind="initial",
        sr=sr,
        somega=somega,
        dt=dt,
        total_time=total_time,
        window_time=window_time,
        sample_time=sample_time,
        project_sticking=int(project_sticking),
        field_scale=field_scale,
        elapsed_seconds=elapsed,
    )
    return ScanResult(v0, s0, status, metrics, "v0", "s0", metadata)


def classify(result: ScanResult) -> np.ndarray:
    """Conservative descriptive classes; continuous observables remain primary."""
    valid = result.status == 0
    mean_v = result.metric("mean_v")
    mean_omega = result.metric("mean_omega")
    mean_phase = result.metric("mean_phase_rate")
    delta_v = result.metric("delta_mean_v")
    delta_phase = result.metric("delta_mean_phase_rate")
    stationary = valid & (np.abs(delta_v) < 0.01) & (np.abs(delta_phase) < 0.01)
    locked = stationary & (np.abs(mean_phase) < 0.02)
    quiescent = stationary & ~locked & (np.abs(mean_v) < 0.02) & (np.abs(mean_omega) < 0.02)
    running = stationary & ~locked & ~quiescent
    labels = np.zeros(result.status.shape, dtype=np.int8)
    labels[locked] = 1
    labels[quiescent] = 2
    labels[running] = 3
    labels[valid & ~stationary] = 4
    return labels


def _robust_symmetric(values: np.ndarray, valid: np.ndarray) -> float:
    finite = np.abs(values[valid & np.isfinite(values)])
    if not finite.size:
        return 1.0
    limit = float(np.percentile(finite, 99.5))
    return max(limit, 1.0e-8)


def _positive_norm(values: np.ndarray, valid: np.ndarray):
    finite = values[valid & np.isfinite(values) & (values > 0)]
    if not finite.size:
        return None
    lo = max(float(np.percentile(finite, 1.0)), 1.0e-8)
    hi = max(float(np.percentile(finite, 99.5)), lo * 1.001)
    return LogNorm(lo, hi)


def plot(result: ScanResult, path: Path, *, xscale: str = "linear", yscale: str = "linear") -> None:
    valid = result.status == 0
    x_edges = _cell_edges(result.x)
    y_edges = _cell_edges(result.y)
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 13.5), constrained_layout=True)
    panels = (
        ("mean_v", "Mean translation velocity", "signed"),
        ("mean_phase_rate", "Mean phase drift", "signed"),
        ("std_v", "Velocity fluctuation", "positive"),
        ("std_phase_rate", "Phase-rate fluctuation", "positive"),
        ("mean_abs_slip", "Mean absolute slip", "positive"),
        ("sliding_fraction", "Sliding fraction", "fraction"),
        ("mean_abs_q", "Mean |q|", "positive"),
        ("delta_mean_phase_rate", "Late - early phase drift", "signed"),
    )
    for axis, (name, title, style) in zip(axes.ravel()[:8], panels):
        values = np.ma.masked_where(~valid, result.metric(name))
        kwargs = {}
        cmap = "viridis"
        if style == "signed":
            limit = _robust_symmetric(result.metric(name), valid)
            cmap = "coolwarm"
            kwargs["norm"] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        elif style == "positive":
            kwargs["norm"] = _positive_norm(result.metric(name), valid)
        else:
            kwargs.update(vmin=0.0, vmax=1.0)
        image = axis.pcolormesh(x_edges, y_edges, values, shading="flat", cmap=cmap, **kwargs)
        figure.colorbar(image, ax=axis)
        axis.set_title(title)

    classes = classify(result)
    class_cmap = ListedColormap(("0.55", "#2166ac", "#fddbc7", "#b2182b", "#762a83"))
    class_norm = BoundaryNorm(np.arange(-0.5, 5.5), class_cmap.N)
    image = axes[2, 2].pcolormesh(x_edges, y_edges, classes, cmap=class_cmap, norm=class_norm, shading="flat")
    bar = figure.colorbar(image, ax=axes[2, 2], ticks=range(5))
    bar.ax.set_yticklabels(("failed", "locked-like", "quiescent", "running", "unsettled"))
    axes[2, 2].set_title("Descriptive classification")

    for axis in axes.ravel():
        axis.set_xlabel(result.x_name)
        axis.set_ylabel(result.y_name)
        axis.set_xscale(xscale)
        axis.set_yscale(yscale)
    figure.suptitle(
        f"Long-time phase observables ({result.metadata['kind']}, "
        f"T={result.metadata['total_time']:g}, window={result.metadata['window_time']:g})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save(result: ScanResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=result.x,
        y=result.y,
        status=result.status,
        metrics=result.metrics,
        metric_names=np.asarray(METRIC_NAMES),
        classes=classify(result),
        x_name=result.x_name,
        y_name=result.y_name,
        metadata=np.asarray(repr(result.metadata)),
    )


def summarize(result: ScanResult) -> str:
    classes = classify(result)
    status_values, status_counts = np.unique(result.status, return_counts=True)
    class_values, class_counts = np.unique(classes, return_counts=True)
    return (
        f"elapsed={result.metadata['elapsed_seconds']:.3f}s; "
        f"status={dict(zip(status_values.tolist(), status_counts.tolist()))}; "
        f"classes={dict(zip(class_values.tolist(), class_counts.tolist()))}; "
        f"mean_phase_rate=[{np.nanmin(result.metric('mean_phase_rate')):.6g}, "
        f"{np.nanmax(result.metric('mean_phase_rate')):.6g}]; "
        f"mean_v=[{np.nanmin(result.metric('mean_v')):.6g}, {np.nanmax(result.metric('mean_v')):.6g}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    parameter = subparsers.add_parser("parameter")
    parameter.add_argument("--sr-lower", type=float, default=0.02)
    parameter.add_argument("--sr-upper", type=float, default=5.0)
    parameter.add_argument("--somega-lower", type=float, default=0.02)
    parameter.add_argument("--somega-upper", type=float, default=5.0)
    parameter.add_argument("--scale", choices=("linear", "log"), default="log")

    initial = subparsers.add_parser("initial")
    initial.add_argument("--sr", type=float, default=1.0)
    initial.add_argument("--somega", type=float, default=1.0)
    initial.add_argument("--v0-lower", type=float, default=-0.5)
    initial.add_argument("--v0-upper", type=float, default=0.5)
    initial.add_argument("--s0-lower", type=float, default=-0.5)
    initial.add_argument("--s0-upper", type=float, default=0.5)

    for subparser in (parameter, initial):
        subparser.add_argument("--k", type=int, default=128)
        subparser.add_argument("--dt", type=float, default=1.0e-3)
        subparser.add_argument("--time", type=float, default=4000.0)
        subparser.add_argument("--window", type=float, default=1000.0)
        subparser.add_argument("--sample-time", type=float, default=0.1)
        subparser.add_argument(
            "--project-sticking",
            action="store_true",
            help="in the analysis-only CUDA copy, enforce v+alpha*omega=0 after every sticking step",
        )
        subparser.add_argument(
            "--field-scale",
            type=float,
            default=1.0,
            help="multiply the compile-time H_EXT value in the analysis-only CUDA copy",
        )
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.k <= 46340:
        parser.error("k must be between 2 and 46340")

    start = time.perf_counter()
    if args.kind == "parameter":
        x = _values(args.sr_lower, args.sr_upper, args.k, args.scale)
        y = _values(args.somega_lower, args.somega_upper, args.k, args.scale)
        result = run_parameter_scan(
            x,
            y,
            dt=args.dt,
            total_time=args.time,
            window_time=args.window,
            sample_time=args.sample_time,
            project_sticking=args.project_sticking,
            field_scale=args.field_scale,
        )
        xscale = yscale = args.scale
    else:
        x = _values(args.v0_lower, args.v0_upper, args.k, "linear")
        y = _values(args.s0_lower, args.s0_upper, args.k, "linear")
        result = run_initial_scan(
            x,
            y,
            sr=args.sr,
            somega=args.somega,
            dt=args.dt,
            total_time=args.time,
            window_time=args.window,
            sample_time=args.sample_time,
            project_sticking=args.project_sticking,
            field_scale=args.field_scale,
        )
        xscale = yscale = "linear"

    save(result, args.output)
    plot(result, args.output.with_suffix(".png"), xscale=xscale, yscale=yscale)
    print(summarize(result))
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {args.output.with_suffix('.png').resolve()}")
    print(f"Wall time: {time.perf_counter()-start:.3f}s")


if __name__ == "__main__":
    main()
