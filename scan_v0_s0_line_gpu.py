"""High-resolution one-dimensional initial-state basin scan on CUDA/NVRTC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RUNS_DIR, SCRATCH_DIR, image_output_path
from scan_sr_somega_gpu import (
    DEFAULT_REFERENCE,
    METRIC_INDEX,
    METRIC_NAMES,
    _get_cuda_module,
    _get_cupy,
    _time_arguments,
)


# Configured cut through the regular stripe region.
SR = 2.5
SOMEGA = 2.25447
V0 = 4.25
S0_LOWER = -6.80
S0_UPPER = -6.75
K = 8192
DT = 1.0e-3
TOTAL_TIME = 50000.0
OBSERVATION_WINDOW = 4000.0
SAMPLE_TIME = 0.2
BLOCK_SIZE = 256
ATTRACTOR_THRESHOLD = 0.8
USE_KAHAN = True

DATA_PATH = RUNS_DIR / "initial-conditions" / "line-scan" / "v0_4p25_s0_m6p80_m6p75_k8192.npz"
FIGURE_PATH = SCRATCH_DIR / "initial-conditions" / "line-scan" / "v0_4p25_s0_m6p80_m6p75_k8192.png"


def contiguous_intervals(x: np.ndarray, selected: np.ndarray) -> list[tuple[float, float, int]]:
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, cuts)
    half_step = 0.5 * float(x[1] - x[0]) if x.size > 1 else 0.0
    return [
        (float(x[group[0]] - half_step), float(x[group[-1]] + half_step), int(group.size))
        for group in groups
    ]


def main(use_kahan: bool = USE_KAHAN) -> None:
    steps, window_steps, sample_stride = _time_arguments(
        DT, TOTAL_TIME, OBSERVATION_WINDOW, SAMPLE_TIME
    )
    s0 = np.linspace(S0_LOWER, S0_UPPER, K, dtype=np.float32)
    cp = _get_cupy()
    print(f"Loading CUDA module from source with NVRTC ({'Kahan FP32' if use_kahan else 'ordinary FP32'})...", flush=True)
    kernel = _get_cuda_module(cp, use_kahan=use_kahan).get_function("AnalyzeInitialLine")
    status_gpu = cp.full(K, 4, dtype=cp.int32)
    metrics_gpu = cp.full((K, len(METRIC_NAMES)), cp.nan, dtype=cp.float32)

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    kernel(
        ((K + BLOCK_SIZE - 1) // BLOCK_SIZE,),
        (BLOCK_SIZE,),
        (
            np.float32(SR), np.float32(SOMEGA), np.float32(V0), cp.asarray(s0), np.int32(K),
            np.float32(DT), np.int32(steps), np.int32(window_steps), np.int32(sample_stride),
            np.float32(DEFAULT_REFERENCE.alpha), np.float32(DEFAULT_REFERENCE.delta),
            np.float32(DEFAULT_REFERENCE.gamma_g), np.float32(DEFAULT_REFERENCE.gamma),
            np.float32(DEFAULT_REFERENCE.gamma_m), status_gpu, metrics_gpu,
        ),
    )
    stop.record()
    stop.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000.0
    status = cp.asnumpy(status_gpu)
    metrics = cp.asnumpy(metrics_gpu)
    valid = status == 0
    if np.any(valid & ~np.all(np.isfinite(metrics), axis=1)):
        raise RuntimeError("successful CUDA trajectories contain non-finite metrics")

    phase_mean = metrics[:, METRIC_INDEX["mean_phase_rate"]]
    phase_std = metrics[:, METRIC_INDEX["std_phase_rate"]]
    red = valid & (phase_mean > ATTRACTOR_THRESHOLD)
    intervals = contiguous_intervals(s0, red)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        s0=s0,
        status=status,
        metrics=metrics,
        metric_names=np.asarray(METRIC_NAMES),
        red_attractor=red,
        sr=np.asarray(SR),
        somega=np.asarray(SOMEGA),
        v0=np.asarray(V0),
        dt=np.asarray(DT),
        total_time=np.asarray(TOTAL_TIME),
        observation_window=np.asarray(OBSERVATION_WINDOW),
        sample_time=np.asarray(SAMPLE_TIME),
        attractor_threshold=np.asarray(ATTRACTOR_THRESHOLD),
        kahan=np.asarray(use_kahan),
    )

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(s0[valid], phase_mean[valid], ".", ms=1.8, color="#9e0142")
    axes[0].axhline(ATTRACTOR_THRESHOLD, color="0.25", ls="--", lw=1.0)
    axes[0].set_ylabel(r"$\langle\dot\phi\rangle_{\rm end}$")
    axes[0].set_title(r"Final-window phase velocity mean, $\dot\phi=1-v-\dot\theta$")
    axes[1].plot(s0[valid], phase_std[valid], ".", ms=1.8, color="#5e2b97")
    axes[1].set_ylabel(r"$\sigma_{\dot\phi,\rm end}$")
    axes[1].set_title("Final-window phase velocity standard deviation")
    axes[2].step(s0, red.astype(np.int8), where="mid", color="#b40426", lw=0.8)
    axes[2].set_yticks((0, 1), labels=("reverse rolling", "co-moving"))
    axes[2].set_ylim(-0.15, 1.15)
    axes[2].set_ylabel("attractor")
    axes[2].set_xlabel(r"initial slip $S_0=v_0+\alpha\omega_0$")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        rf"1D basin cut: $s_r={SR:g}$, $s_\omega={SOMEGA:g}$, $v_0={V0:g}$, "
        rf"$k={K}$, $T={TOTAL_TIME:g}$; {'Kahan FP32' if use_kahan else 'ordinary FP32'}"
    )
    figure_path = image_output_path(FIGURE_PATH)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    widths = np.asarray([upper - lower for lower, upper, _ in intervals])
    print(f"Finished in {elapsed:.3f} s: {np.count_nonzero(valid)}/{K} succeeded")
    print(f"grid spacing={float(s0[1] - s0[0]):.9g}; co-moving points={np.count_nonzero(red)} ({np.mean(red):.3%})")
    print(f"co-moving intervals={len(intervals)}")
    if widths.size:
        print(
            f"interval width min/median/max={widths.min():.9g}/"
            f"{np.median(widths):.9g}/{widths.max():.9g}"
        )
    for lower, upper, count in intervals[:20]:
        print(f"  [{lower:.9g}, {upper:.9g}]  width={upper-lower:.9g}  samples={count}")
    if len(intervals) > 20:
        print(f"  ... {len(intervals) - 20} more intervals")
    print(f"Saved {DATA_PATH.resolve()}")
    print(f"Saved {figure_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", action="store_true", help="use ordinary FP32 instead of Kahan FP32")
    main(use_kahan=not parser.parse_args().ordinary)
