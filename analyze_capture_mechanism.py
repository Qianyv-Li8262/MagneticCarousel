"""Summarize and plot the CPU dump evidence for the striped basin mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent


def wrapped_angle(values: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * values))


def first_time_below(frame: pd.DataFrame, threshold: float, after: float = 10.0) -> float | None:
    selected = frame.loc[(frame["time"] > after) & (frame["qnorm"] < threshold), "time"]
    return None if selected.empty else float(selected.iloc[0])


def initial_sticking_time(frame: pd.DataFrame) -> float | None:
    selected = frame.loc[frame["sliding_to_sticking_count"] > 0, "time"]
    return None if selected.empty else float(selected.iloc[0])


def accepted_rhs(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = frame["accepted"].astype(str).str.lower().isin(("true", "1"))
    return frame.loc[accepted].copy()


def late_stats(path: Path, start: float = 10_000.0) -> dict[str, float]:
    frame = pd.read_csv(path / "trajectory_overview.csv")
    late = frame.loc[frame["time"] >= start]
    return {
        "v_mean": float(late["xdot"].mean()),
        "omega_mean": float(late["theta_dot"].mean()),
        "phidot_mean": float(late["phase_rate"].mean()),
        "phidot_std": float(late["phase_rate"].std(ddof=0)),
        "qnorm_mean": float(late["qnorm"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=SOURCE_DIR / "runs" / "trajectories" / "capture-mechanism",
        help="directory containing the CPU dump runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCE_DIR / "scratch" / "trajectories" / "capture-mechanism" / "capture_mechanism_cpu_fp64.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SOURCE_DIR / "results" / "trajectories" / "capture-mechanism" / "capture_mechanism_summary.json",
    )
    args = parser.parse_args()

    root = args.root
    inside = pd.read_csv(root / "inside-fp64-fine" / "dump_states.csv")
    outside = pd.read_csv(root / "outside-fp64-fine" / "dump_states.csv")
    inside_initial = pd.read_csv(root / "inside-fp64-initial-full" / "dump_states.csv")
    outside_initial = pd.read_csv(root / "outside-fp64-initial-full" / "dump_states.csv")
    inside_rhs = accepted_rhs(pd.read_csv(root / "inside-fp64-switch-full" / "rhs.csv"))
    outside_rhs = accepted_rhs(pd.read_csv(root / "outside-fp64-switch-full" / "rhs.csv"))
    inside_changes = pd.read_csv(root / "inside-fp64-fine" / "branch_changes.csv")
    outside_changes = pd.read_csv(root / "outside-fp64-fine" / "branch_changes.csv")

    physical_inside_changes = inside_changes.loc[inside_changes["t_stage"] > 1.0]
    physical_outside_changes = outside_changes.loc[outside_changes["t_stage"] > 1.0]
    first_relu_crossing = (
        None if physical_outside_changes.empty else float(physical_outside_changes.iloc[0]["t_stage"])
    )
    velocity_difference = np.abs(inside["xdot"].to_numpy() - outside["xdot"].to_numpy())
    separation_indices = np.flatnonzero(velocity_difference > 0.5)
    separation_index = int(separation_indices[0]) if len(separation_indices) else None
    separation_time = None if separation_index is None else float(inside.iloc[separation_index]["time"])
    if separation_index is None:
        separation_phase_offset = None
        separation_q_angle_offset = None
    else:
        phase_inside = inside.iloc[separation_index]["phase"] - inside.iloc[separation_index]["theta"]
        phase_outside = outside.iloc[separation_index]["phase"] - outside.iloc[separation_index]["theta"]
        q_angle_inside = np.arctan2(inside.iloc[separation_index]["qy"], inside.iloc[separation_index]["qx"])
        q_angle_outside = np.arctan2(outside.iloc[separation_index]["qy"], outside.iloc[separation_index]["qx"])
        separation_phase_offset = float(wrapped_angle(np.array([phase_inside - phase_outside]))[0])
        separation_q_angle_offset = float(wrapped_angle(np.array([q_angle_inside - q_angle_outside]))[0])

    summary = {
        "parameters": {"v0": 4.25, "sr": 2.5, "somega": 2.25447, "dt": 0.001},
        "trajectory_separation": {
            "first_abs_velocity_difference_above_0.5": separation_time,
            "body_frame_phase_offset_at_that_time": separation_phase_offset,
            "q_angle_offset_at_that_time": separation_q_angle_offset,
        },
        "inside": {
            "s0": -6.798795,
            "sliding_to_sticking_time": initial_sticking_time(inside_initial),
            "relu_crossings_after_t1": int(len(physical_inside_changes)),
            "first_qnorm_below_0.29999": first_time_below(inside, 0.29999),
            "late_fp64": late_stats(root / "inside-fp64"),
            "late_fp32": late_stats(root / "inside-fp32"),
        },
        "outside": {
            "s0": -6.797,
            "sliding_to_sticking_time": initial_sticking_time(outside_initial),
            "relu_crossings_after_t1": int(len(physical_outside_changes)),
            "first_relu_crossing": first_relu_crossing,
            "first_qnorm_below_0.29999": first_time_below(outside, 0.29999),
            "first_qnorm_below_0.29": first_time_below(outside, 0.29),
            "late_fp64": late_stats(root / "outside-fp64"),
            "late_fp32": late_stats(root / "outside-fp32"),
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11), layout="constrained")
    colors = {"inside": "tab:red", "outside": "tab:blue"}

    for label, frame in (("inside band", inside), ("adjacent gap", outside)):
        key = "inside" if label.startswith("inside") else "outside"
        axes[0, 0].plot(frame["time"], frame["xdot"], lw=0.9, color=colors[key], label=label)
        axes[0, 1].plot(frame["time"], frame["phase_rate"], lw=0.9, color=colors[key], label=label)
        axes[1, 0].plot(frame["time"], frame["qnorm"], lw=0.9, color=colors[key], label=label)

    axes[0, 0].set(title="Centroid velocity", xlabel="time", ylabel=r"$v=\dot{x}$")
    axes[0, 1].set(title=r"Phase velocity $\dot{\phi}=1-v-\dot{\theta}$", xlabel="time", ylabel=r"$\dot{\phi}$")
    axes[1, 0].set(title="Hysteresis-state radius", xlabel="time", ylabel=r"$|q|$")
    axes[1, 0].axhline(0.3, ls="--", lw=0.8, color="0.35", label="saturation radius")
    axes[1, 0].set_ylim(0.235, 0.302)

    for label, rhs in (("inside band", inside_rhs), ("adjacent gap", outside_rhs)):
        key = "inside" if label.startswith("inside") else "outside"
        axes[1, 1].plot(rhs["t_stage"], rhs["relu_test"], lw=0.8, color=colors[key], label=label)
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    if first_relu_crossing is not None:
        axes[1, 1].axvline(first_relu_crossing, color="tab:blue", ls="--", lw=0.9)
    axes[1, 1].set(
        title=r"Later desaturation: first hysteresis-branch crossing",
        xlabel="time",
        ylabel=r"$q\cdot\dot{H}_{e,\mathrm{test}}$",
        xlim=(6089.0, 6095.0),
    )

    circle_angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[2, 0].plot(0.3 * np.cos(circle_angle), 0.3 * np.sin(circle_angle), "--", color="0.5", lw=0.8)
    for label, rhs in (("inside band", inside_rhs), ("adjacent gap", outside_rhs)):
        key = "inside" if label.startswith("inside") else "outside"
        axes[2, 0].plot(rhs["qx"], rhs["qy"], lw=0.8, color=colors[key], label=label)
    axes[2, 0].set(
        title=r"Magnetic internal state near the first branch crossing",
        xlabel=r"$q_x$",
        ylabel=r"$q_y$",
        aspect="equal",
    )

    for label, frame in (("inside band", inside), ("adjacent gap", outside)):
        key = "inside" if label.startswith("inside") else "outside"
        zoom = frame.loc[(frame["time"] >= 2680.0) & (frame["time"] <= 2820.0)]
        axes[2, 1].plot(zoom["time"], zoom["xdot"], lw=0.9, color=colors[key], label=label)
    if separation_time is not None:
        axes[2, 1].axvline(separation_time, color="0.25", ls="--", lw=0.8)
    axes[2, 1].set(
        title=r"Resonant capture: one trajectory closes around $v=1$",
        xlabel="time",
        ylabel=r"$v=\dot{x}$",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(
        "CPU FP64 dump: neighboring initial states select different attractors\n"
        r"$v_0=4.25,\ s_r=2.5,\ s_\omega=2.25447$; inside $S_0=-6.798795$, gap $S_0=-6.797$",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
