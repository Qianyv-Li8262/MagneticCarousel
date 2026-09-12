"""Plot the Cartesian phase portrait (|q|, d|q|/dt) from a dump directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BRANCH_COLORS = {0: "tab:blue", 1: "tab:red"}


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def accepted_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(("true", "false", "1", "0"))
    if invalid.any():
        raise ValueError("rhs.csv contains an unrecognized value in the accepted column")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def valid_branch(series: pd.Series, path: Path) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = ~numeric.isin((0, 1))
    if invalid.any():
        raise ValueError(f"{path.name} contains a ReLU branch other than 0 or 1")
    return numeric.to_numpy(dtype=np.int8)


def from_rhs(path: Path) -> pd.DataFrame:
    rhs = pd.read_csv(path)
    require_columns(rhs, {"step", "t_stage", "qx", "qy", "qdotx", "qdoty", "branch", "accepted"}, path)
    rhs = rhs.loc[accepted_mask(rhs["accepted"])].copy()
    if rhs.empty:
        raise ValueError(f"{path.name} contains no accepted RHS rows")
    sort_columns = [name for name in ("t_stage", "step", "role", "stage") if name in rhs]
    rhs = rhs.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    qx = rhs["qx"].to_numpy(dtype=float)
    qy = rhs["qy"].to_numpy(dtype=float)
    qdotx = rhs["qdotx"].to_numpy(dtype=float)
    qdoty = rhs["qdoty"].to_numpy(dtype=float)
    qnorm = np.hypot(qx, qy)
    scale = max(1.0, float(np.nanmax(qnorm)))
    nonsingular = qnorm > np.finfo(float).eps * scale
    rate = np.full_like(qnorm, np.nan)
    rate[nonsingular] = (qx[nonsingular] * qdotx[nonsingular] + qy[nonsingular] * qdoty[nonsingular]) / qnorm[
        nonsingular
    ]

    result = pd.DataFrame(
        {
            "step": rhs["step"].to_numpy(dtype=np.int64),
            "time": rhs["t_stage"].to_numpy(dtype=float),
            "qnorm": qnorm,
            "qnorm_rate": rate,
            "branch": valid_branch(rhs["branch"], path),
            "source": "accepted_rhs",
        }
    )
    if "step_time" in rhs:
        result.insert(2, "step_time", rhs["step_time"].to_numpy(dtype=float))
    return result.loc[np.isfinite(result["qnorm_rate"])].reset_index(drop=True)


def from_states(path: Path) -> pd.DataFrame:
    states = pd.read_csv(path)
    require_columns(states, {"step", "time", "qx", "qy", "last_branch"}, path)
    if len(states) < 2:
        raise ValueError(f"{path.name} needs at least two rows for finite differencing")
    states = states.sort_values(["time", "step"], kind="stable").reset_index(drop=True)
    time = states["time"].to_numpy(dtype=float)
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path.name} time must be finite and strictly increasing")

    qnorm = np.hypot(states["qx"].to_numpy(dtype=float), states["qy"].to_numpy(dtype=float))
    edge_order = 2 if len(states) >= 3 else 1
    rate = np.gradient(qnorm, time, edge_order=edge_order)
    branch = pd.to_numeric(states["last_branch"], errors="coerce").where(lambda values: values.isin((0, 1)))
    branch = branch.ffill().bfill()
    if branch.isna().any():
        raise ValueError(f"{path.name} contains no valid last_branch value")

    return pd.DataFrame(
        {
            "step": states["step"].to_numpy(dtype=np.int64),
            "time": time,
            "qnorm": qnorm,
            "qnorm_rate": rate,
            "branch": branch.to_numpy(dtype=np.int8),
            "source": "state_central_difference",
        }
    )


def load_phase_data(input_dir: Path, source: str) -> tuple[pd.DataFrame, str]:
    rhs_path = input_dir / "rhs.csv"
    state_path = input_dir / "dump_states.csv"
    if source == "rhs" or (source == "auto" and rhs_path.is_file()):
        return from_rhs(rhs_path), "accepted RHS: (q · qdot) / |q|"
    if source == "states" or source == "auto":
        if not state_path.is_file():
            raise FileNotFoundError(f"missing state file: {state_path}")
        return from_states(state_path), "state samples: finite difference of |q|"
    raise AssertionError(f"unexpected source {source}")


def downsample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def title_from_metadata(input_dir: Path) -> str:
    path = input_dir / "run_metadata.json"
    if not path.is_file():
        return "Radial q phase portrait"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return (
            rf"$v_0={metadata['v0']:g},\ S_0={metadata['s0']:g},\ "
            rf"s_r={metadata['sr']:g},\ s_\omega={metadata['somega']:g}$"
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "Radial q phase portrait"


def plot_phase_portrait(data: pd.DataFrame, output: Path, title: str, subtitle: str, dpi: int, max_points: int) -> None:
    indices = downsample_indices(len(data), max_points)
    qnorm = data["qnorm"].to_numpy(dtype=float)[indices]
    rate = data["qnorm_rate"].to_numpy(dtype=float)[indices]
    branch = data["branch"].to_numpy(dtype=np.int8)[indices]
    if len(qnorm) < 2:
        raise ValueError("fewer than two finite phase-space points remain")

    points = np.column_stack((qnorm, rate))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_branch = branch[1:]

    figure, axis = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    for value in (0, 1):
        selected = segment_branch == value
        if selected.any():
            axis.add_collection(
                LineCollection(
                    segments[selected],
                    colors=BRANCH_COLORS[value],
                    linewidths=1.0,
                    alpha=0.78,
                    rasterized=len(segments) > 50000,
                )
            )
    axis.scatter(
        [qnorm[0]],
        [rate[0]],
        s=85,
        marker="o",
        color="limegreen",
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )
    axis.scatter(
        [qnorm[-1]],
        [rate[-1]],
        s=100,
        marker="X",
        color="black",
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
    axis.autoscale()
    axis.margins(0.04)
    axis.set_xlabel(r"$|q|$")
    axis.set_ylabel(r"$d|q|/dt$")
    axis.grid(alpha=0.23)
    axis.legend(
        handles=(
            Line2D([], [], color=BRANCH_COLORS[1], linewidth=2.0, label="ReLU x"),
            Line2D([], [], color=BRANCH_COLORS[0], linewidth=2.0, label="ReLU 0"),
            Line2D([], [], marker="o", linestyle="", color="limegreen", label="start"),
            Line2D([], [], marker="X", linestyle="", color="black", label="end"),
        ),
        loc="best",
    )
    axis.set_title(f"{title}\n{subtitle}; plotted points={len(qnorm):,}/{len(data):,}")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory produced by dump_one_trajectory.py")
    parser.add_argument(
        "--source",
        choices=("auto", "rhs", "states"),
        default="auto",
        help="auto prefers accepted rhs.csv and falls back to state finite differences",
    )
    parser.add_argument("--output", type=Path, default=None, help="PNG path; defaults inside input_dir")
    parser.add_argument("--derived-csv", type=Path, default=None, help="derived CSV path; defaults inside input_dir")
    parser.add_argument("--max-plot-points", type=int, default=200000)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_plot_points < 2:
        parser.error("--max-plot-points must be at least 2")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    output = (args.output or input_dir / "q_radial_phase_space.png").resolve()
    derived_csv = (args.derived_csv or input_dir / "q_radial_phase_space.csv").resolve()
    existing = [path for path in (output, derived_csv) if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output already exists: {existing[0]} (use --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    derived_csv.parent.mkdir(parents=True, exist_ok=True)

    data, method = load_phase_data(input_dir, args.source)
    if data.empty:
        raise ValueError("no finite phase-space data remain after processing")
    data.to_csv(derived_csv, index=False)
    plot_phase_portrait(
        data,
        output,
        title_from_metadata(input_dir),
        method,
        dpi=args.dpi,
        max_points=args.max_plot_points,
    )
    print(f"wrote {output}")
    print(f"wrote {derived_csv}")
    print(f"source={method}; rows={len(data):,}")


if __name__ == "__main__":
    main()
