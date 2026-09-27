# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Build a Poincare section directly from trajectory dump CSV files.

The default section is the positive crossing

    phi = phase - theta = 2*pi*n,  dphi/dt > 0,

where ``phase`` is the lab-field phase t-x stored by the dump kernel.  State
values at the crossing are linearly interpolated between adjacent CSV rows.
"""

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
DEFAULT_RUNS = (
    SOURCE_DIR / "runs" / "trajectories" / "capture-mechanism" / "inside-fp64-fine",
    SOURCE_DIR / "runs" / "trajectories" / "capture-mechanism" / "outside-fp64-fine",
)
REQUIRED = {"time", "phase", "theta", "phase_rate", "xdot", "theta_dot", "qx", "qy", "qnorm"}


def load_states(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    detailed = run_dir / "dump_states.csv"
    overview = run_dir / "trajectory_overview.csv"
    source = detailed if detailed.is_file() else overview
    if not source.is_file():
        raise FileNotFoundError(f"no dump_states.csv or trajectory_overview.csv in {run_dir}")
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    frame = frame.sort_values("time", kind="stable").drop_duplicates("time").reset_index(drop=True)
    if len(frame) < 2 or np.any(np.diff(frame["time"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError(f"{source} needs strictly increasing time samples")
    return frame, source


def interpolate_crossings(frame: pd.DataFrame, transient: float, direction: str) -> pd.DataFrame:
    time = frame["time"].to_numpy(dtype=float)
    # phase and theta are individually wrapped by the integrator.  Form the
    # relative angle on the unit circle first, then unwrap it continuously.
    wrapped_phi = np.angle(np.exp(1j * (frame["phase"].to_numpy() - frame["theta"].to_numpy())))
    phi = np.unwrap(wrapped_phi)
    two_pi = 2.0 * np.pi
    section_index = np.floor(phi / two_pi).astype(np.int64)
    jump = np.diff(section_index)
    if direction == "positive":
        candidates = np.flatnonzero(jump > 0)
    elif direction == "negative":
        candidates = np.flatnonzero(jump < 0)
    else:
        candidates = np.flatnonzero(jump != 0)

    rows: list[dict[str, float | int]] = []
    values = {name: frame[name].to_numpy(dtype=float) for name in ("xdot", "theta_dot", "phase_rate", "qx", "qy", "qnorm")}
    for left in candidates:
        right = left + 1
        step = 1 if jump[left] > 0 else -1
        # Sampling should normally contain one crossing.  This loop also
        # handles coarser CSV files that span several turns in one interval.
        for level_index in range(section_index[left] + step, section_index[right] + step, step):
            level = level_index * two_pi
            denominator = phi[right] - phi[left]
            if denominator == 0.0:
                continue
            fraction = (level - phi[left]) / denominator
            if not 0.0 <= fraction <= 1.0:
                continue
            crossing_time = time[left] + fraction * (time[right] - time[left])
            if crossing_time < transient:
                continue
            row: dict[str, float | int] = {
                "crossing": len(rows),
                "turn": int(level_index),
                "time": float(crossing_time),
                "direction": int(step),
            }
            for name, array in values.items():
                row[name] = float(array[left] + fraction * (array[right] - array[left]))
            row["qangle"] = float(np.arctan2(row["qy"], row["qx"]))
            rows.append(row)
    return pd.DataFrame(rows)


def label_for(run_dir: Path) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return rf"$S_0={metadata['s0']:g}$"
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="dump directories; defaults to the current inside/outside FP64 pair")
    parser.add_argument("--transient", type=float, default=2500.0, help="discard crossings earlier than this time")
    parser.add_argument("--direction", choices=("positive", "negative", "both"), default="positive")
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCE_DIR / "scratch" / "trajectories" / "capture-mechanism" / "poincare_section_fp64.png",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=SOURCE_DIR / "results" / "trajectories" / "capture-mechanism" / "poincare",
    )
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()

    if not np.isfinite(args.transient) or args.transient < 0.0:
        parser.error("--transient must be finite and non-negative")
    run_dirs = tuple(path.resolve() for path in (args.runs or DEFAULT_RUNS))
    sections: list[tuple[str, pd.DataFrame, Path]] = []
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in run_dirs:
        states, source = load_states(run_dir)
        section = interpolate_crossings(states, args.transient, args.direction)
        if section.empty:
            raise RuntimeError(f"no requested section crossings found in {source}")
        section_path = args.csv_dir / f"{run_dir.name}_poincare.csv"
        section.to_csv(section_path, index=False)
        sections.append((label_for(run_dir), section, section_path))
        print(f"{run_dir.name}: {len(section)} crossings from {source.name} -> {section_path}")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (label, section, _) in zip(colors, sections):
        age = np.linspace(0.2, 1.0, len(section))
        axes[0, 0].scatter(section["qx"], section["qy"], s=8, color=color, alpha=age, label=label, rasterized=True)
        axes[0, 1].scatter(section["xdot"], section["qx"], s=8, color=color, alpha=age, label=label, rasterized=True)
        axes[1, 0].scatter(section["time"], section["xdot"], s=7, color=color, alpha=0.55, label=label, rasterized=True)
        if len(section) >= 2:
            axes[1, 1].scatter(
                section["xdot"].iloc[:-1], section["xdot"].iloc[1:], s=8, color=color, alpha=0.55, label=label, rasterized=True
            )

    circle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[0, 0].plot(0.3 * np.cos(circle), 0.3 * np.sin(circle), "--", color="0.5", lw=0.8, label=r"$|q|=0.3$")
    axes[0, 0].set(title="Poincare section in hysteresis state", xlabel=r"$q_x$", ylabel=r"$q_y$", aspect="equal")
    axes[0, 1].set(title="Coupled mechanical-magnetic section", xlabel=r"$v$", ylabel=r"$q_x$")
    axes[1, 0].set(title="Successive section hits", xlabel="crossing time", ylabel=r"$v_n$")
    axes[1, 1].set(title="Return map", xlabel=r"$v_n$", ylabel=r"$v_{n+1}$")
    limits = axes[1, 1].get_xlim()
    lower = min(limits[0], axes[1, 1].get_ylim()[0])
    upper = max(limits[1], axes[1, 1].get_ylim()[1])
    axes[1, 1].plot([lower, upper], [lower, upper], "--", color="0.5", lw=0.8)
    axes[1, 1].set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal", adjustable="box")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    direction_text = {"positive": r"$\dot\phi>0$", "negative": r"$\dot\phi<0$", "both": "both directions"}[args.direction]
    figure.suptitle(
        rf"Poincare section: $\phi=t-x-\theta=2\pi n$, {direction_text}; $t\geq{args.transient:g}$"
        "\n"
        "Crossings linearly interpolated from CPU dump CSV",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
