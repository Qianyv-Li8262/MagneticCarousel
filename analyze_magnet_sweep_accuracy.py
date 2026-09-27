# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Diagnose startup scaling, late phase offsets, and time-step sensitivity."""

from __future__ import annotations

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from compare_magnet_counts_exact import COUNTS, DATA_DIR, PLOT_DIR, SUMMARY_DIR, read_rows


def refined_velocity(count: int, factor: int) -> np.ndarray | None:
    path = DATA_DIR / f"N{count}_dt_div_{factor}.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as source:
        return np.array([float(row["v_2d_equivalent"]) for row in csv.DictReader(source)])


def main():
    reference = read_rows(COUNTS[0])
    t = reference["time_1d"]
    v1 = reference["v_1d"]
    peaks, _ = find_peaks(v1[t >= 200], distance=10, prominence=0.03)
    period = float(np.polyfit(np.arange(len(peaks)), t[t >= 200][peaks], 1)[0])
    spline = CubicSpline(t, v1)
    startup = t <= 10
    late = (t >= 900) & (t <= 998)
    rows = []
    for count in COUNTS:
        data = read_rows(count)
        v2 = data["v_2d_equivalent"]

        def aligned_rms(delay: float) -> float:
            return float(np.sqrt(np.mean((v2[late] - spline(t[late] + delay)) ** 2)))

        grid = np.linspace(-period / 2, period / 2, 271)
        index = int(np.argmin([aligned_rms(delay) for delay in grid]))
        result = minimize_scalar(
            aligned_rms, method="bounded",
            bounds=(grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]),
            options={"xatol": 1e-8},
        )
        half = refined_velocity(count, 2)
        row = {
            "magnets": count,
            "startup_0_10_rms": float(np.sqrt(np.mean((v2[startup] - v1[startup]) ** 2))),
            "late_900_998_raw_rms": float(np.sqrt(np.mean((v2[late] - v1[late]) ** 2))),
            "late_phase_shift_time": float(result.x),
            "late_900_998_aligned_rms": float(result.fun),
            "late_900_998_step_halving_rms": (
                float(np.sqrt(np.mean((v2[late] - half[late]) ** 2))) if half is not None else math.nan
            ),
        }
        rows.append(row)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / "accuracy_diagnostics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = np.array([row["magnets"] for row in rows])
    start_error = np.array([row["startup_0_10_rms"] for row in rows])
    raw = np.array([row["late_900_998_raw_rms"] for row in rows])
    aligned = np.array([row["late_900_998_aligned_rms"] for row in rows])
    shift = np.array([row["late_phase_shift_time"] for row in rows])
    step = np.array([row["late_900_998_step_halving_rms"] for row in rows])
    slope = np.polyfit(np.log(n[n >= 256]), np.log(start_error[n >= 256]), 1)[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    axes[0, 0].loglog(n, start_error, "o-", label="2D − 1D RMS, t=0–10")
    guide_n = n[n >= 128]
    axes[0, 0].loglog(guide_n, start_error[n.tolist().index(512)] * (512 / guide_n) ** 2,
                      "k--", label="N⁻² guide")
    axes[0, 0].set(title=f"Startup scaling (N≥256 fit: N^{slope:.2f})", ylabel="RMS velocity difference")
    axes[0, 0].legend()

    axes[0, 1].loglog(n, raw, "o-", label="same clock time")
    axes[0, 1].loglog(n, aligned, "o-", label="after optimal phase shift")
    axes[0, 1].set(title="Late periodic state, t=900–998", ylabel="RMS velocity difference")
    axes[0, 1].legend()

    axes[1, 0].scatter(n, shift)
    axes[1, 0].axhline(0, color="0.5", lw=0.7)
    axes[1, 0].set(title="Equivalent phase shift, wrapped to ±half a period",
                   ylabel="Shift in 1D time")

    present = np.isfinite(step)
    axes[1, 1].semilogy(n[present], raw[present], "o", label="2D vs 1D")
    axes[1, 1].semilogy(n[present], step[present], "s", label="2D base vs half step")
    axes[1, 1].set(title="Integration-step sensitivity, t=900–998",
                   ylabel="RMS velocity difference")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(n, [str(count) for count in n], rotation=45)
        axis.set_xlabel("Magnet count N")
        axis.grid(alpha=0.25)
    fig.suptitle("Exact-field magnet sweep: geometry, phase, and numerical accuracy")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "accuracy_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    zoom = (t >= 900) & (t <= 920)
    for axis, count in zip(axes, (1024, 4096)):
        data = read_rows(count)
        delay = next(row["late_phase_shift_time"] for row in rows if row["magnets"] == count)
        axis.plot(t[zoom], v1[zoom], "k--", lw=1.5, label="1D at same time")
        axis.plot(t[zoom], data["v_2d_equivalent"][zoom], color="tab:blue", lw=1.8,
                  label=f"2D, N={count}")
        axis.plot(t[zoom], spline(t[zoom] + delay), color="tab:orange", lw=1.2,
                  label=f"1D shifted by {delay:+.3f}")
        axis.set_ylabel("Equivalent tangential speed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("1D time")
    fig.suptitle("Late trajectory: the apparent discrepancy is mainly phase")
    fig.savefig(PLOT_DIR / "late_phase_alignment.png", dpi=180)
    plt.close(fig)
    print(f"Reference period ≈ {period:.6f}; startup power slope ≈ {slope:.3f}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
