"""Build exact-field 2D models for N=8..4096 and compare with the 1D trajectory.

Run with the Python configured in CMakeUserPresets.json::

    python compare_magnet_counts_exact.py

Each N has its own build tree and simulation process, so identically named
Python extensions cannot be confused or reused between magnet counts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from animate_one_trajectory import scaled_model_parameters, timestep_for_magnets
from project_paths import RESULTS_DIR, RUNS_DIR, SCRATCH_DIR


SOURCE_DIR = Path(__file__).resolve().parent
COUNTS = tuple(2**power for power in range(3, 13))
DT_1D = 0.001
SAMPLE_TIME = 0.2
TOTAL_TIME = 1000.0
STARTUP_TIME = 100.0
LATE_WINDOW = 100.0
DATA_DIR = RUNS_DIR / "magnet-sweep-exact"
PLOT_DIR = SCRATCH_DIR / "magnet-sweep-exact"
SUMMARY_DIR = RESULTS_DIR / "magnet-sweep-exact"


def load_extension(path: Path):
    candidates = list(path.glob("hysteresis*.pyd"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one hysteresis extension in {path}, found {len(candidates)}")
    spec = importlib.util.spec_from_file_location("hysteresis", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(count: int) -> Path:
    tree = SOURCE_DIR / "build" / "magnet-sweep" / f"exact-{count}"
    subprocess.run(
        ["cmake", "--preset", "local-cpu", "-B", str(tree),
         f"-DIYPT16_MAGNET_NUMBER={count}", "-DIYPT16_FIELD_MODE=exact", "-DBUILD_TESTING=OFF"],
        cwd=SOURCE_DIR, check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(tree), "--target", "hysteresis_fp64", "-j", "4"],
        cwd=SOURCE_DIR, check=True,
    )
    return tree / "extensions" / "fp64"


def simulate_reference(total_time: float):
    module1 = load_extension(SOURCE_DIR.parent / "src" / "build" / "release" / "extensions" / "fp64")
    p1_values = (
        0.33333333 * 0.428571,
        1.7212357e-4 / 0.428571,
        0.16560356 / 0.1**2,
        489.39639 / 0.1**2,
        11011.419 / 0.428571**2 / 0.1**2,
    )
    params = module1.SolverParams(*p1_values)
    state = np.zeros(8, dtype=module1.real_dtype)
    mode = sign = None
    rows = [{"time_1d": 0.0, "x_1d": 0.0, "v_1d": 0.0}]
    for index in range(1, round(total_time / SAMPLE_TIME) + 1):
        state, mode, sign = module1.evolve_with_discrete_mode(
            state, DT_1D, round(SAMPLE_TIME / DT_1D), params, mode, sign)
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite 1D reference at t={index * SAMPLE_TIME:g}")
        rows.append({"time_1d": index * SAMPLE_TIME,
                     "x_1d": float(state[1]), "v_1d": float(state[2])})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "reference_1d.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(count: int, total_time: float) -> dict:
    module2 = load_extension(SOURCE_DIR / "build" / "magnet-sweep" / f"exact-{count}" / "extensions" / "fp64")
    if module2.magnet_number != count or module2.field_mode != "exact":
        raise RuntimeError(f"Incorrect 2D build: N={module2.magnet_number}, field={module2.field_mode}")
    with (DATA_DIR / "reference_1d.csv").open(newline="", encoding="utf-8") as source:
        reference = list(csv.DictReader(source))
    p2 = module2.SolverParams(*scaled_model_parameters(count))
    radius0 = count / 2
    dt2 = timestep_for_magnets(count)
    two_steps = round(SAMPLE_TIME / (radius0 * dt2))
    if not math.isclose(two_steps * radius0 * dt2, SAMPLE_TIME):
        raise RuntimeError("Sample time is not aligned with both integrators")
    samples = round(total_time / SAMPLE_TIME)
    if not math.isclose(samples * SAMPLE_TIME, total_time) or samples <= 0:
        raise ValueError("total_time must be a positive multiple of the sample time")

    state2 = np.zeros(module2.state_size, dtype=module2.real_dtype)
    state2[1], state2[5] = radius0, 1.0
    mode2 = direction2 = None
    last_angle = 0.0
    unwrapped_angle = 0.0
    rows = []

    def record(index: int):
        nonlocal last_angle, unwrapped_angle
        x, y, vx, vy = map(float, state2[1:5])
        radius = math.hypot(x, y)
        angle = math.atan2(y, x)
        if index:
            unwrapped_angle += (angle - last_angle + math.pi) % (2 * math.pi) - math.pi
        last_angle = angle
        tangent = (x * vy - y * vx) / radius
        rows.append({
            "time_1d": index * SAMPLE_TIME,
            "x_1d": float(reference[index]["x_1d"]),
            "v_1d": float(reference[index]["v_1d"]),
            "x_2d_equivalent": radius0 * unwrapped_angle,
            "v_2d_equivalent": tangent / radius0,
            "radius_offset": radius - radius0,
            "radial_velocity": (x * vx + y * vy) / radius,
            "sliding_2d": int(bool(mode2)),
        })

    record(0)
    for index in range(1, samples + 1):
        state2, mode2, direction2 = module2.evolve_trajectory(
            state2, dt2, two_steps, p2, mode2, direction2, (index - 1) * two_steps)
        if not np.isfinite(state2).all():
            raise RuntimeError(f"Non-finite state at N={count}, t={index * SAMPLE_TIME:g}")
        record(index)
        if index % 100 == 0:
            print(f"N={count}: {index}/{samples} samples", flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"N{count}.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    window = [row for row in rows if row["time_1d"] > total_time - LATE_WINDOW]
    summary = {
        "magnets": count, "field_mode": module2.field_mode,
        "total_time_1d": total_time, "dt_1d": DT_1D, "dt_2d": dt2,
        "final_v_1d": rows[-1]["v_1d"], "final_v_2d": rows[-1]["v_2d_equivalent"],
        "final_v_difference": rows[-1]["v_2d_equivalent"] - rows[-1]["v_1d"],
        "late_mean_v_1d": float(np.mean([row["v_1d"] for row in window])),
        "late_mean_v_2d": float(np.mean([row["v_2d_equivalent"] for row in window])),
        "late_rms_v_difference": float(np.sqrt(np.mean([
            (row["v_2d_equivalent"] - row["v_1d"]) ** 2 for row in window]))),
        "final_radius_offset": rows[-1]["radius_offset"],
        "late_mean_radius_offset": float(np.mean([row["radius_offset"] for row in window])),
        "data_file": str(path),
    }
    (DATA_DIR / f"N{count}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_rows(count: int) -> dict[str, np.ndarray]:
    with (DATA_DIR / f"N{count}.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def plot_traces(counts: tuple[int, ...], total_time: float, start: float, stop: float, name: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(counts)))
    reference = read_rows(counts[0])
    mask = (reference["time_1d"] >= start) & (reference["time_1d"] <= stop)
    axes[0].plot(reference["time_1d"][mask], reference["v_1d"][mask], "k--", lw=2, label="1D")
    axes[1].plot(reference["time_1d"][mask], reference["x_1d"][mask], "k--", lw=2)
    for count, color in zip(counts, colors):
        data = read_rows(count)
        t = data["time_1d"][mask]
        axes[0].plot(t, data["v_2d_equivalent"][mask], color=color, lw=1.1, label=f"N={count}")
        axes[1].plot(t, data["x_2d_equivalent"][mask], color=color, lw=1.1)
        axes[2].plot(t, (data["v_2d_equivalent"] - data["v_1d"])[mask], color=color, lw=1.1)
        axes[3].plot(t, data["radius_offset"][mask], color=color, lw=1.1)
    axes[0].set_ylabel("Tangential speed / R")
    axes[1].set_ylabel("Tangential displacement")
    axes[2].set_ylabel("2D − 1D speed")
    axes[3].set_ylabel("Radius − R")
    axes[3].set_xlabel("1D time  (t₁ = R t₂)")
    axes[0].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Exact magnetic field: {name.replace('_', ' ')} trajectories (1D time {start:g}–{stop:g}; total {total_time:g})")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_final(summary: list[dict], total_time: float):
    counts = np.array([item["magnets"] for item in summary])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    axes[0].plot(counts, [item["final_v_2d"] for item in summary], "o-", label="2D final")
    axes[0].axhline(summary[0]["final_v_1d"], color="k", linestyle="--", label="1D final")
    axes[0].set_ylabel("Final v / R")
    axes[0].legend()
    axes[1].plot(counts, [item["late_mean_v_2d"] for item in summary], "o-", label="2D late mean")
    axes[1].axhline(summary[0]["late_mean_v_1d"], color="k", linestyle="--", label="1D late mean")
    axes[1].set_ylabel("Last 100 time units: mean v / R")
    axes[1].legend()
    axes[2].plot(counts, [item["late_rms_v_difference"] for item in summary], "o-")
    axes[2].set_ylabel("Last 100: RMS(2D − 1D)")
    axes[2].set_xlabel("Magnet count N")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(counts, [str(count) for count in counts])
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Exact magnetic field: final-state comparison at 1D time {total_time:g}")
    fig.savefig(PLOT_DIR / "final_state_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=int, choices=COUNTS, help=argparse.SUPPRESS)
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    args = parser.parse_args()
    if (not math.isfinite(args.total_time) or args.total_time < LATE_WINDOW or
            not math.isclose(round(args.total_time / SAMPLE_TIME) * SAMPLE_TIME, args.total_time)):
        parser.error("--total-time must be >= 100 and a multiple of 0.2")
    if args.reference:
        simulate_reference(args.total_time)
        return
    if args.single is not None:
        print(json.dumps(simulate(args.single, args.total_time), indent=2), flush=True)
        return

    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reference",
                    "--total-time", str(args.total_time)], cwd=SOURCE_DIR, check=True)
    for count in COUNTS:
        print(f"Building exact field for N={count}", flush=True)
        build(count)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single", str(count),
                        "--total-time", str(args.total_time)], cwd=SOURCE_DIR, check=True)

    summary = [json.loads((DATA_DIR / f"N{count}.json").read_text(encoding="utf-8")) for count in COUNTS]
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot_traces(COUNTS, args.total_time, 0, min(STARTUP_TIME, args.total_time), "startup")
    plot_traces(COUNTS, args.total_time, args.total_time - LATE_WINDOW, args.total_time, "late_state")
    plot_final(summary, args.total_time)
    print(f"Data: {DATA_DIR}\nSummary: {SUMMARY_DIR / 'summary.csv'}\nPlots: {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    main()
