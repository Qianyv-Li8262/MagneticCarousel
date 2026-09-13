"""Scan initial velocity near the locked/unlocked basin boundary.

The C++ solver wraps its stored phase modulo 2*pi.  This script therefore
never differentiates the stored phase.  It diagnoses phase locking from the
exact kinematic rate

    phase_rate = 1 - v - omega.

Each trajectory is written incrementally to its own CSV file.  Evolution
stops as soon as a 100-frame window confirms either phase locking or sustained
recovery of the frame-mean velocity.  A summary CSV and diagnostic figure are
written after all trajectories have finished.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RUNS_DIR, fp64_module_dir


# ============================ Scan configuration ============================
# Keep the ten sampling points explicit so that changing the scan never
# requires touching the detection code.  The provisional critical point is
# midway between the previously resolved values 0.18922 and 0.18923.
VC_ESTIMATE = 0.189225
V0_SAMPLES = np.asarray(
    [
        0.189125,
        0.189175,
        0.189200,
        0.189215,
        0.189220,
        0.189230,
        0.189235,
        0.189250,
        0.189275,
        0.189325,
    ],
    dtype=np.float64,
)

SR = 0.8
SOMEGA = 0.6
S0 = 0.0

DT = 1.0e-3
FRAME_TIME = 50.0
MAX_TIME = 300_000.0
CONFIRM_FRAMES = 100

# Locked-state tests.  All must hold over the same confirmation window.
LOCKED_SPEED_ERROR_TOL = 1.0e-3
LOCKED_SPEED_SPREAD_TOL = 2.0e-3
LOCKED_SPEED_SLOPE_TOL = 2.0e-7
LOCKED_PHASE_RATE_RMS_TOL = 3.0e-3
LOCKED_SLIP_MAX_TOL = 1.0e-8

# Unlocked recovery tests.  The second half of the window must lie above the
# first half, the fitted trend must be positive, and phase winding must be
# clearly nonzero.  These window tests tolerate small physical oscillations.
RECOVERY_SPEED_SLOPE_MIN = 2.0e-7
RECOVERY_HALF_RISE_MIN = 1.0e-3
RECOVERY_FROM_MIN_MIN = 2.0e-3
UNLOCKED_PHASE_RATE_MEAN_MIN = 2.0e-2
UNLOCKED_SLIP_MAX_TOL = 1.0e-6

# Reference nondimensional parameters.
ALPHA_REF = 0.33333333
DELTA_REF = 1.7212357e-4
GAMMA_G_REF = 0.16560356
GAMMA_REF = 489.39639
GAMMA_M_REF = 11011.419
# ===========================================================================


TRAJECTORY_HEADER = (
    "frame",
    "v0",
    "t",
    "x",
    "instantaneous_v",
    "theta_wrapped",
    "instantaneous_omega",
    "phase_wrapped",
    "q_x",
    "q_y",
    "frame_mean_v",
    "instantaneous_phase_rate",
    "slip",
    "sliding",
)

SUMMARY_HEADER = (
    "v0",
    "v0_minus_vc_estimate",
    "outcome",
    "event_time",
    "decision_time",
    "confirmation_time",
    "frames",
    "final_t",
    "final_frame_mean_v",
    "final_instantaneous_v",
    "final_instantaneous_omega",
    "final_phase_rate",
    "final_slip",
    "minimum_frame_mean_v",
    "minimum_time",
    "trajectory_csv",
    "message",
)


@dataclass
class Detection:
    outcome: str
    event_time: float
    decision_time: float
    message: str


@dataclass
class TrajectoryResult:
    summary: dict[str, object]
    times: np.ndarray
    frame_mean_v: np.ndarray
    phase_rate: np.ndarray


def load_fp64(module_dir: Path):
    """Load the separately built FP64 extension, never a nearby FP32 build."""
    module_path = str(module_dir.resolve())
    sys.path.insert(0, module_path)
    try:
        hysteresis = importlib.import_module("hysteresis")
    finally:
        sys.path.remove(module_path)
    if np.dtype(hysteresis.real_dtype) != np.dtype(np.float64):
        raise RuntimeError(
            f"Expected an FP64 module, loaded {hysteresis.real_dtype} "
            f"from {hysteresis.__file__}"
        )
    return hysteresis


def model_parameters(hysteresis):
    alpha = ALPHA_REF * SR
    inverse_frequency_squared = 1.0 / SOMEGA**2
    params = hysteresis.SolverParams(
        alpha,
        DELTA_REF / SR,
        GAMMA_G_REF * inverse_frequency_squared,
        GAMMA_REF * inverse_frequency_squared,
        GAMMA_M_REF / (SR**2) * inverse_frequency_squared,
    )
    return alpha, params


def make_initial_state(v0: float, alpha: float) -> np.ndarray:
    state = np.zeros(8, dtype=np.float64)
    state[2] = v0
    state[4] = (S0 - v0) / alpha
    return state


def linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    centered_t = times - np.mean(times)
    denominator = float(centered_t @ centered_t)
    if denominator == 0.0:
        return 0.0
    return float(centered_t @ (values - np.mean(values)) / denominator)


def detect_outcome(
    times,
    frame_mean_v,
    phase_rate,
    slip,
    minimum_index: int,
    locked_velocity: float,
) -> Detection | None:
    """Return a confirmed online classification, or None to keep evolving."""
    if len(frame_mean_v) < CONFIRM_FRAMES:
        return None

    start = len(frame_mean_v) - CONFIRM_FRAMES
    window_t = np.asarray(times[-CONFIRM_FRAMES:], dtype=np.float64)
    window_v = np.asarray(frame_mean_v[-CONFIRM_FRAMES:], dtype=np.float64)
    window_phase_rate = np.asarray(phase_rate[-CONFIRM_FRAMES:], dtype=np.float64)
    window_slip = np.asarray(slip[-CONFIRM_FRAMES:], dtype=np.float64)
    speed_slope = linear_slope(window_t, window_v)

    locked = (
        abs(float(np.mean(window_v)) - locked_velocity) <= LOCKED_SPEED_ERROR_TOL
        and float(np.ptp(window_v)) <= LOCKED_SPEED_SPREAD_TOL
        and abs(speed_slope) <= LOCKED_SPEED_SLOPE_TOL
        and float(np.sqrt(np.mean(window_phase_rate**2))) <= LOCKED_PHASE_RATE_RMS_TOL
        and float(np.max(np.abs(window_slip))) <= LOCKED_SLIP_MAX_TOL
    )
    if locked:
        return Detection(
            outcome="locked",
            event_time=float(window_t[0]),
            decision_time=float(window_t[-1]),
            message="100-frame velocity plateau and phase-rate lock confirmed",
        )

    half = CONFIRM_FRAMES // 2
    half_rise = float(np.mean(window_v[half:]) - np.mean(window_v[:half]))
    recovered_from_minimum = float(window_v[-1] - frame_mean_v[minimum_index])
    minimum_precedes_late_half = minimum_index < start + half
    unlocked = (
        minimum_precedes_late_half
        and speed_slope >= RECOVERY_SPEED_SLOPE_MIN
        and half_rise >= RECOVERY_HALF_RISE_MIN
        and recovered_from_minimum >= RECOVERY_FROM_MIN_MIN
        and float(np.mean(window_phase_rate)) >= UNLOCKED_PHASE_RATE_MEAN_MIN
        and float(np.max(np.abs(window_slip))) <= UNLOCKED_SLIP_MAX_TOL
    )
    if unlocked:
        return Detection(
            outcome="unlocked",
            event_time=float(times[minimum_index]),
            decision_time=float(window_t[-1]),
            message="100-frame sustained velocity recovery and phase winding confirmed",
        )
    return None


def trajectory_filename(v0: float) -> str:
    return f"trajectory_v0_{v0:.9f}.csv"


def run_trajectory(
    hysteresis,
    params,
    alpha: float,
    v0: float,
    steps_per_frame: int,
    output_dir: Path,
) -> TrajectoryResult:
    state = make_initial_state(v0, alpha)
    locked_velocity = -alpha / (1.0 - alpha)
    output_path = output_dir / trajectory_filename(v0)

    times: list[float] = []
    frame_velocities: list[float] = []
    phase_rates: list[float] = []
    slips: list[float] = []
    minimum_index = -1
    detection: Detection | None = None
    error_message = ""
    previous_x = float(state[1])

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(TRAJECTORY_HEADER)
        writer.writerow(
            (
                0,
                v0,
                float(state[0]),
                float(state[1]),
                float(state[2]),
                float(state[3]),
                float(state[4]),
                float(state[5]),
                float(state[6]),
                float(state[7]),
                math.nan,
                1.0 - float(state[2]) - float(state[4]),
                float(state[2]) + alpha * float(state[4]),
                0,
            )
        )

        frame = 0
        while float(state[0]) < MAX_TIME:
            frame += 1
            try:
                state, sliding = hysteresis.evolve_with_mode(state, DT, steps_per_frame, params)
            except Exception as exc:  # preserve partial CSV and continue the scan
                error_message = f"{type(exc).__name__}: {exc}"
                break

            t = float(state[0])
            x = float(state[1])
            velocity = float(state[2])
            omega = float(state[4])
            frame_velocity = (x - previous_x) / FRAME_TIME
            previous_x = x
            current_phase_rate = 1.0 - velocity - omega
            current_slip = velocity + alpha * omega

            times.append(t)
            frame_velocities.append(frame_velocity)
            phase_rates.append(current_phase_rate)
            slips.append(current_slip)
            if minimum_index < 0 or frame_velocity < frame_velocities[minimum_index]:
                minimum_index = len(frame_velocities) - 1

            writer.writerow(
                (
                    frame,
                    v0,
                    t,
                    x,
                    velocity,
                    float(state[3]),
                    omega,
                    float(state[5]),
                    float(state[6]),
                    float(state[7]),
                    frame_velocity,
                    current_phase_rate,
                    current_slip,
                    int(sliding),
                )
            )

            detection = detect_outcome(
                times,
                frame_velocities,
                phase_rates,
                slips,
                minimum_index,
                locked_velocity,
            )
            if detection is not None:
                break

            if frame % 1000 == 0:
                print(
                    f"  v0={v0:.9f}: t={t:.0f}, frame_mean_v={frame_velocity:+.6f}, "
                    f"phase_rate={current_phase_rate:+.6f}",
                    flush=True,
                )

    time_array = np.asarray(times)
    velocity_array = np.asarray(frame_velocities)
    phase_rate_array = np.asarray(phase_rates)
    if velocity_array.size:
        minimum_value = float(velocity_array[minimum_index])
        minimum_time = float(time_array[minimum_index])
        final_frame_velocity = float(velocity_array[-1])
        final_phase_rate = float(phase_rate_array[-1])
        final_slip = float(slips[-1])
    else:
        minimum_value = math.nan
        minimum_time = math.nan
        final_frame_velocity = math.nan
        final_phase_rate = math.nan
        final_slip = math.nan

    if detection is not None:
        outcome = detection.outcome
        event_time = detection.event_time
        decision_time = detection.decision_time
        message = detection.message
    elif error_message:
        outcome = "error"
        event_time = math.nan
        decision_time = float(state[0])
        message = error_message
    else:
        outcome = "undecided"
        event_time = math.nan
        decision_time = float(state[0])
        message = f"no criterion confirmed before MAX_TIME={MAX_TIME:g}"

    summary = {
        "v0": v0,
        "v0_minus_vc_estimate": v0 - VC_ESTIMATE,
        "outcome": outcome,
        "event_time": event_time,
        "decision_time": decision_time,
        "confirmation_time": decision_time - event_time if np.isfinite(event_time) else math.nan,
        "frames": len(times),
        "final_t": float(state[0]),
        "final_frame_mean_v": final_frame_velocity,
        "final_instantaneous_v": float(state[2]),
        "final_instantaneous_omega": float(state[4]),
        "final_phase_rate": final_phase_rate,
        "final_slip": final_slip,
        "minimum_frame_mean_v": minimum_value,
        "minimum_time": minimum_time,
        "trajectory_csv": output_path.name,
        "message": message,
    }
    return TrajectoryResult(summary, time_array, velocity_array, phase_rate_array)


def save_summary(path: Path, results: list[TrajectoryResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(result.summary for result in results)


def plot_results(path: Path, results: list[TrajectoryResult], locked_velocity: float) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), constrained_layout=True)
    colors = plt.get_cmap("coolwarm")(np.linspace(0.05, 0.95, len(results)))

    for result, color in zip(results, colors):
        summary = result.summary
        label = f"{float(summary['v0']):.6f} ({summary['outcome']})"
        if result.times.size:
            axes[0].plot(result.times, result.frame_mean_v, color=color, lw=1.2, label=label)
            axes[1].plot(result.times, result.phase_rate, color=color, lw=1.1)

    axes[0].axhline(locked_velocity, color="0.2", ls="--", lw=1.2, label=rf"$v_*={locked_velocity:.6f}$")
    axes[0].set(xlabel="$t$", ylabel="frame-mean $v$", title="Velocity histories")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].axhline(0.0, color="0.2", ls="--", lw=1.0)
    axes[1].set(
        xlabel="$t$",
        ylabel=r"sampled $\dot\phi=1-v-\omega$",
        title="Phase locking versus winding",
    )

    outcome_styles = {
        "locked": ("tab:blue", "o"),
        "unlocked": ("tab:orange", "s"),
        "undecided": ("0.4", "x"),
        "error": ("tab:red", "X"),
    }
    for outcome, (color, marker) in outcome_styles.items():
        selected = [r.summary for r in results if r.summary["outcome"] == outcome]
        if not selected:
            continue
        distances = np.asarray([abs(float(row["v0_minus_vc_estimate"])) for row in selected])
        event_times = np.asarray([float(row["event_time"]) for row in selected])
        finite = np.isfinite(event_times) & (distances > 0.0)
        if np.any(finite):
            axes[2].loglog(
                distances[finite],
                event_times[finite],
                linestyle="none",
                marker=marker,
                color=color,
                ms=7,
                label=outcome,
            )
    axes[2].set(
        xlabel=r"$|v_0-v_{0,c}^{\rm est}|$",
        ylabel="detected event time",
        title="Critical-slowing diagnostic",
    )
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    figure.suptitle(
        rf"Near-boundary scan: $s_r={SR:g}$, $s_\omega={SOMEGA:g}$, "
        rf"$S_0={S0:g}$, $v_{{0,c}}^{{\rm est}}={VC_ESTIMATE:.6f}$"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def validate_configuration() -> int:
    if V0_SAMPLES.ndim != 1 or V0_SAMPLES.size != 10:
        raise ValueError("V0_SAMPLES must be a one-dimensional array containing exactly 10 points")
    if not np.all(np.isfinite(V0_SAMPLES)) or np.unique(V0_SAMPLES).size != V0_SAMPLES.size:
        raise ValueError("V0_SAMPLES must contain ten distinct finite values")
    if DT <= 0.0 or FRAME_TIME <= 0.0 or MAX_TIME <= 0.0:
        raise ValueError("DT, FRAME_TIME, and MAX_TIME must be positive")
    steps_per_frame = round(FRAME_TIME / DT)
    if not np.isclose(steps_per_frame * DT, FRAME_TIME, rtol=0.0, atol=1.0e-12):
        raise ValueError("FRAME_TIME must be an integer multiple of DT")
    if CONFIRM_FRAMES < 2 or CONFIRM_FRAMES % 2 != 0:
        raise ValueError("CONFIRM_FRAMES must be an even integer of at least two")
    return steps_per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=fp64_module_dir())
    parser.add_argument(
        "--output-dir", type=Path, default=RUNS_DIR / "critical-slowing" / "fp64-trajectories"
    )
    args = parser.parse_args()

    steps_per_frame = validate_configuration()
    hysteresis = load_fp64(args.module_dir)
    alpha, params = model_parameters(hysteresis)
    locked_velocity = -alpha / (1.0 - alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"FP64 module: {hysteresis.__file__}")
    print(f"V0 samples: {V0_SAMPLES}")
    print(
        f"frame_time={FRAME_TIME:g}, confirmation_time={CONFIRM_FRAMES * FRAME_TIME:g}, "
        f"max_time={MAX_TIME:g}",
        flush=True,
    )

    results: list[TrajectoryResult] = []
    for index, v0 in enumerate(V0_SAMPLES, start=1):
        print(f"[{index}/{V0_SAMPLES.size}] Starting v0={v0:.9f}", flush=True)
        result = run_trajectory(
            hysteresis,
            params,
            alpha,
            float(v0),
            steps_per_frame,
            args.output_dir,
        )
        results.append(result)
        summary = result.summary
        print(
            f"  -> {summary['outcome']}: event_t={float(summary['event_time']):g}, "
            f"decision_t={float(summary['decision_time']):g}",
            flush=True,
        )

    summary_path = args.output_dir / "critical_slowing_summary.csv"
    figure_path = args.output_dir / "critical_slowing.png"
    save_summary(summary_path, results)
    plot_results(figure_path, results, locked_velocity)
    print(f"Saved {summary_path.resolve()}")
    print(f"Saved {figure_path.resolve()}")


if __name__ == "__main__":
    main()
