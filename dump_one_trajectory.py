"""Run one deterministic CPU trajectory, dump a chosen time window, and plot it."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RUNS_DIR, dump_module_dir


ALPHA_REF = 0.33333333
DELTA_REF = 1.7212357e-4
GAMMA_G_REF = 0.16560356
GAMMA_REF = 489.39639
GAMMA_M_REF = 11011.419

OVERVIEW_COLUMNS = (
    "step",
    "time",
    "integrator_t",
    "x",
    "xdot",
    "theta",
    "theta_dot",
    "phase",
    "phase_rate",
    "qx",
    "qy",
    "qnorm",
    "slip",
    "sliding",
    "slip_sign",
)

DUMP_COLUMNS = (
    *OVERVIEW_COLUMNS,
    "rhs_count",
    "zero_branch_count",
    "x_branch_count",
    "branch_switch_count",
    "min_relu_test",
    "max_relu_test",
    "closest_relu_test",
    "last_branch",
    "sliding_to_sticking_count",
    "sliding_reverse_count",
    "sticking_to_sliding_count",
)

BRANCH_CHANGE_COLUMNS = (
    "step",
    "step_time",
    "t_stage",
    "qx",
    "qy",
    "relu_test",
    "previous_branch",
    "new_branch",
    "role",
    "stage",
    "sliding",
)

RHS_COLUMNS = (
    "step",
    "step_time",
    "t_stage",
    "qx",
    "qy",
    "qdotx",
    "qdoty",
    "h0_dot_x",
    "h0_dot_y",
    "he_dot_test_x",
    "he_dot_test_y",
    "relu_test",
    "branch",
    "role",
    "stage",
    "sliding",
    "accepted",
)


def finite_positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{name} must be finite and positive")


def aligned_step(parser: argparse.ArgumentParser, name: str, value: float, dt: float) -> int:
    if not math.isfinite(value) or value < 0.0:
        parser.error(f"{name} must be finite and non-negative")
    ratio = value / dt
    step = round(ratio)
    grid_time = step * dt
    tolerance = max(1.0e-12, 64.0 * np.finfo(np.float64).eps * max(1.0, abs(value), abs(grid_time)))
    if abs(value - grid_time) > tolerance:
        parser.error(f"{name}={value:g} is not aligned with dt={dt:g}; nearest grid time is {grid_time:.17g}")
    if step > np.iinfo(np.int32).max:
        parser.error(f"{name} requires {step} steps, exceeding the current int32 step limit")
    return int(step)


def load_module(module_dir: Path):
    module_dir = module_dir.resolve()
    if not module_dir.is_dir():
        raise FileNotFoundError(f"module directory does not exist: {module_dir}")
    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module("hysteresis_dump")
    finally:
        sys.path.remove(str(module_dir))
    return module


def scaled_parameters(module, sr: float, somega: float):
    alpha = ALPHA_REF * sr
    inverse_somega_squared = 1.0 / (somega * somega)
    params = module.SolverParams(
        alpha,
        DELTA_REF / sr,
        GAMMA_G_REF * inverse_somega_squared,
        GAMMA_REF * inverse_somega_squared,
        GAMMA_M_REF / (sr * sr) * inverse_somega_squared,
    )
    return alpha, params


def initial_state(module, v0: float, s0: float, alpha: float) -> np.ndarray:
    state = np.zeros(8, dtype=module.real_dtype)
    state[2] = v0
    state[4] = (s0 - v0) / alpha
    return state


def require_current_extension(module, source_dir: Path) -> None:
    extension = Path(module.__file__).resolve()
    sources = (
        source_dir / "1dmodel-hysteresis.hpp",
        source_dir / "1dmodel-hysteresis-dump.hpp",
        source_dir / "1dmodel-hysteresis.cpp",
        source_dir / "pybind_dump.cpp",
    )
    newer = [source.name for source in sources if source.stat().st_mtime > extension.stat().st_mtime]
    if newer:
        raise RuntimeError(
            f"loaded {extension.name}, but these sources are newer: {', '.join(newer)}; "
            "rebuild with `python src/build_dump.py --fp64`"
        )


def enrich_state_table(table: dict[str, np.ndarray], alpha: float, dt: float) -> dict[str, np.ndarray]:
    enriched = dict(table)
    # The integer grid is authoritative. Do not use accumulated integrator_t for alignment.
    enriched["time"] = np.asarray(enriched["step"], dtype=np.float64) * dt
    enriched["phase_rate"] = 1.0 - enriched["xdot"] - enriched["theta_dot"]
    enriched["slip"] = enriched["xdot"] + alpha * enriched["theta_dot"]
    return enriched


def scalar_for_csv(value):
    return value.item() if isinstance(value, np.generic) else value


def write_csv(path: Path, table: dict[str, np.ndarray], columns: Iterable[str]) -> None:
    columns = tuple(columns)
    missing = [name for name in columns if name not in table]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    lengths = {len(np.asarray(table[name])) for name in columns}
    if len(lengths) != 1:
        raise ValueError(f"{path.name}: columns have inconsistent lengths")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(
            tuple(scalar_for_csv(value) for value in row)
            for row in zip(*(table[name] for name in columns), strict=True)
        )


def mark_dump_window(axes, start_time: float, end_time: float) -> None:
    for axis in axes:
        axis.axvspan(start_time, end_time, color="tab:orange", alpha=0.13, label="dump window")
        axis.axvline(start_time, color="tab:orange", linestyle="--", linewidth=1.1)
        axis.axvline(end_time, color="tab:red", linestyle="--", linewidth=1.1)
        axis.grid(alpha=0.25)


def plot_overview(path: Path, data: dict[str, np.ndarray], start_time: float, end_time: float, title: str) -> None:
    time = data["time"]
    figure, axes = plt.subplots(4, 1, figsize=(12.0, 9.0), sharex=True, layout="constrained")
    series = (
        (data["x"], r"$x(t)$", "tab:blue"),
        (data["theta"], r"$\theta(t)$", "tab:purple"),
        (data["phase_rate"], r"$\dot{\phi}=1-v-\omega$", "tab:red"),
        (data["xdot"], r"$v=\dot{x}$", "tab:green"),
    )
    for axis, (values, label, color) in zip(axes, series, strict=True):
        axis.plot(time, values, color=color, linewidth=1.25)
        axis.set_ylabel(label)
    mark_dump_window(axes, start_time, end_time)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel(r"grid time $t=\mathrm{step}\,\Delta t$")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def q_angle(qx: np.ndarray, qy: np.ndarray, qnorm: np.ndarray) -> np.ndarray:
    angle = np.arctan2(qy, qx).astype(np.float64, copy=False)
    scale = max(1.0, float(np.nanmax(qnorm))) if qnorm.size else 1.0
    angle = np.array(angle, copy=True)
    angle[qnorm <= np.finfo(np.float64).eps * scale] = np.nan
    return angle


def plot_detail(
    path: Path,
    data: dict[str, np.ndarray],
    branch_changes: dict[str, np.ndarray],
    start_time: float,
    end_time: float,
    title: str,
) -> None:
    time = data["time"]
    figure, axes = plt.subplots(7, 1, figsize=(12.0, 13.5), sharex=True, layout="constrained")
    series = (
        (data["x"], r"$x(t)$", "tab:blue"),
        (data["theta"], r"$\theta(t)$", "tab:purple"),
        (data["phase_rate"], r"$\dot{\phi}=1-v-\omega$", "tab:red"),
        (data["xdot"], r"$v=\dot{x}$", "tab:green"),
        (data["qnorm"], r"$|q|$", "tab:brown"),
        (q_angle(data["qx"], data["qy"], data["qnorm"]), r"$\arg(q)$", "tab:cyan"),
    )
    for axis, (values, label, color) in zip(axes[:-1], series, strict=True):
        axis.plot(time, values, color=color, linewidth=1.25)
        axis.set_ylabel(label)
    axes[5].set_ylim(-np.pi, np.pi)
    axes[5].set_yticks((-np.pi, 0.0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))

    valid_branch = data["last_branch"] >= 0
    axes[-1].step(time[valid_branch], data["last_branch"][valid_branch], where="post", color="0.2", linewidth=1.2)
    if len(branch_changes["step_time"]):
        axes[-1].scatter(
            branch_changes["step_time"],
            branch_changes["new_branch"],
            c=branch_changes["new_branch"],
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=16,
            zorder=3,
            label="accepted branch change",
        )
    axes[-1].set_ylabel("ReLU branch")
    axes[-1].set_yticks((0, 1), labels=("zero", "x"))
    axes[-1].set_ylim(-0.2, 1.2)
    if len(branch_changes["step_time"]):
        axes[-1].legend(loc="best")

    mark_dump_window(axes, start_time, end_time)
    padding = max((end_time - start_time) * 0.01, np.finfo(float).eps)
    axes[-1].set_xlim(start_time - padding, end_time + padding)
    axes[-1].set_xlabel(r"grid time $t=\mathrm{step}\,\Delta t$")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def output_paths(output_dir: Path, full_mode: bool) -> list[Path]:
    names = [
        "run_metadata.json",
        "trajectory_overview.csv",
        "dump_states.csv",
        "branch_changes.csv",
        "trajectory_overview.png",
        "dump_detail.png",
    ]
    if full_mode:
        names.append("rhs.csv")
    return [output_dir / name for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=dump_module_dir())
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "trajectories" / "trajectory_dump")
    parser.add_argument("--v0", type=float, required=True)
    parser.add_argument("--s0", type=float, default=0.0)
    parser.add_argument("--sr", type=float, required=True)
    parser.add_argument("--somega", type=float, required=True)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--dump-start-time", type=float, required=True)
    parser.add_argument("--dump-end-time", type=float, required=True)
    parser.add_argument("--dump-every-steps", type=int, default=1000)
    parser.add_argument("--overview-every-steps", type=int, default=10000)
    parser.add_argument("--mode", choices=("summary", "full"), default="summary")
    parser.add_argument("--include-rejected-trials", action="store_true")
    parser.add_argument("--max-records", type=int, default=2_000_000, help="maximum rows per returned table")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in ("v0", "s0"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    finite_positive(parser, "--sr", args.sr)
    finite_positive(parser, "--somega", args.somega)
    finite_positive(parser, "--dt", args.dt)
    if args.dump_every_steps <= 0 or args.overview_every_steps <= 0:
        parser.error("dump and overview intervals must be positive")
    if args.max_records <= 0:
        parser.error("--max-records must be positive")
    if args.include_rejected_trials and args.mode != "full":
        parser.error("--include-rejected-trials requires --mode full")
    start_step = aligned_step(parser, "--dump-start-time", args.dump_start_time, args.dt)
    end_step = aligned_step(parser, "--dump-end-time", args.dump_end_time, args.dt)
    if end_step <= start_step:
        parser.error("--dump-end-time must be greater than --dump-start-time")

    output_dir = args.output_dir.resolve()
    # Include the optional full-mode file so a summary overwrite cannot leave stale RHS data behind.
    paths = output_paths(output_dir, True)
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output already exists: {existing[0]} (use --overwrite to replace this run's files)")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    module = load_module(args.module_dir)
    require_current_extension(module, source_dir)
    dtype = np.dtype(module.real_dtype)
    if dtype != np.dtype(np.float64):
        print(f"warning: loaded {dtype}; FP64 is recommended for theoretical analysis", file=sys.stderr)
    alpha, params = scaled_parameters(module, args.sr, args.somega)
    state = initial_state(module, args.v0, args.s0, alpha)
    full_mode = args.mode == "full"

    final_state, sliding, trace = module.evolve(
        state,
        args.dt,
        end_step,
        params,
        dump=True,
        dump_every=args.dump_every_steps,
        overview_every=args.overview_every_steps,
        start_step=start_step,
        end_step=end_step,
        log_branch_changes=True,
        log_rhs_stages=full_mode,
        log_rejected_trials=full_mode and args.include_rejected_trials,
        max_records=args.max_records,
    )
    if trace["truncated"]:
        raise RuntimeError("trace reached --max-records; increase the limit or narrow the dump window")

    overview = enrich_state_table(trace["overview"], alpha, args.dt)
    detail = enrich_state_table(trace["steps"], alpha, args.dt)
    branch_changes = dict(trace["branch_changes"])
    branch_changes["step_time"] = np.asarray(branch_changes["step"], dtype=np.float64) * args.dt
    rhs = dict(trace["rhs"])
    rhs["step_time"] = np.asarray(rhs["step"], dtype=np.float64) * args.dt
    write_csv(output_dir / "trajectory_overview.csv", overview, OVERVIEW_COLUMNS)
    write_csv(output_dir / "dump_states.csv", detail, DUMP_COLUMNS)
    write_csv(output_dir / "branch_changes.csv", branch_changes, BRANCH_CHANGE_COLUMNS)
    if full_mode:
        write_csv(output_dir / "rhs.csv", rhs, RHS_COLUMNS)

    grid_start_time = start_step * args.dt
    grid_end_time = end_step * args.dt
    title = rf"$v_0={args.v0:g},\ S_0={args.s0:g},\ s_r={args.sr:g},\ s_\omega={args.somega:g}$"
    plot_overview(output_dir / "trajectory_overview.png", overview, grid_start_time, grid_end_time, title)
    plot_detail(output_dir / "dump_detail.png", detail, branch_changes, grid_start_time, grid_end_time, title)

    metadata = {
        "module": str(Path(module.__file__).resolve()),
        "real_dtype": str(dtype),
        "dt": args.dt,
        "v0": args.v0,
        "s0": args.s0,
        "sr": args.sr,
        "somega": args.somega,
        "parameters": {
            "alpha": alpha,
            "delta": DELTA_REF / args.sr,
            "gamma_g": GAMMA_G_REF / args.somega**2,
            "gamma": GAMMA_REF / args.somega**2,
            "gamma_m": GAMMA_M_REF / (args.sr**2 * args.somega**2),
        },
        "dump": {
            "start_step": start_step,
            "end_step": end_step,
            "start_time": grid_start_time,
            "end_time": grid_end_time,
            "dump_every_steps": args.dump_every_steps,
            "overview_every_steps": args.overview_every_steps,
            "mode": args.mode,
            "include_rejected_trials": bool(args.include_rejected_trials),
            "max_records_per_table": args.max_records,
        },
        "row_counts": {
            "overview": len(overview["step"]),
            "dump_states": len(detail["step"]),
            "branch_changes": len(branch_changes["step"]),
            "rhs": len(trace["rhs"]["step"]),
        },
        "branch_encoding": {"-1": "not observed", "0": "zero branch", "1": "x branch"},
        "rk_role_encoding": {"0": "full trial", "1": "pre-event", "2": "post-event"},
        "rk_stage_encoding": {"0": "start", "1": "midpoint"},
        "final_state": [float(value) for value in final_state],
        "final_sliding": bool(sliding),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"wrote trajectory dump to {output_dir}")
    print(
        f"overview rows={metadata['row_counts']['overview']}, "
        f"dump rows={metadata['row_counts']['dump_states']}, "
        f"branch changes={metadata['row_counts']['branch_changes']}, "
        f"rhs rows={metadata['row_counts']['rhs']}"
    )


if __name__ == "__main__":
    main()
