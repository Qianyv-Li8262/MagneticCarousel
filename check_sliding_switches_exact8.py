"""Exercise 2D contact-mode switches with the FP64 exact eight-magnet field.

Build first with::

    cmake --preset local-cpu -B build/sliding-exact8 -DIYPT16_MAGNET_NUMBER=8 -DIYPT16_FIELD_MODE=exact -DBUILD_TESTING=OFF
    cmake --build build/sliding-exact8 --target hysteresis_fp64 -j 4
    python check_sliding_switches_exact8.py

The saved summary compares each time step to the finest run on a common time
grid. Switches are observed at step boundaries, so their times have dt resolution.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from animate_one_trajectory import scaled_model_parameters  # noqa: E402


DT_VALUES = (0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625)
TOTAL_TIME = 1.0
CASES = {
    "forward_capture": {"slip_y": 1.0},
    "backward_capture": {"slip_y": -1.0},
    "radial_capture": {"slip_x": 1.0},
    "rolling_release_recapture": {"q_y": -0.1},
}
EXPECTED_EVENTS = {
    "forward_capture": (("sliding", "rolling"),),
    "backward_capture": (("sliding", "rolling"),),
    "radial_capture": (("sliding", "rolling"),),
    "rolling_release_recapture": (("rolling", "sliding"), ("sliding", "rolling")),
}


def load_module(module_dir: Path):
    candidates = list(module_dir.resolve().glob("hysteresis*.pyd"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one FP64 extension in {module_dir}, found {len(candidates)}")
    spec = importlib.util.spec_from_file_location("hysteresis", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.magnet_number, module.field_mode, module.state_size, np.dtype(module.real_dtype)) != (
        8, "exact", 15, np.dtype(np.float64)
    ):
        raise RuntimeError("This experiment requires the 8-magnet FP64 exact-field 2D extension")
    return module


def initial_state(module, params, case: dict[str, float]):
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    state[1] = 4.0  # magnet ring radius is N/2
    state[5] = 1.0  # identity attitude
    state[10] = -case.get("slip_x", 0.0) / params.alpha
    state[9] = case.get("slip_y", 0.0) / params.alpha
    state[13] = case.get("q_y", 0.0)
    return state


def slip_vector(state: np.ndarray, alpha: float):
    return np.array((state[3] - alpha * state[10], state[4] + alpha * state[9]))


def run(module, params, case, dt: float):
    steps = round(TOTAL_TIME / dt)
    if not np.isclose(steps * dt, TOTAL_TIME, rtol=0, atol=1e-12):
        raise ValueError("Total time must align with every dt")
    state = initial_state(module, params, case)
    sliding = bool(np.linalg.norm(slip_vector(state, params.alpha)) > 1e-10)
    direction = None
    rows = []
    events = []

    def record(step):
        rows.append((step * dt, *map(float, state[1:5]),
                     float(np.linalg.norm(slip_vector(state, params.alpha))), int(sliding)))

    record(0)
    for step in range(steps):
        old_mode = sliding
        state, sliding, direction = module.evolve_trajectory(
            state, dt, 1, params, None if direction is None else old_mode,
            direction, step
        )
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite state at step {step + 1}, dt={dt}")
        if sliding != old_mode:
            events.append({"time": (step + 1) * dt,
                           "from": "sliding" if old_mode else "rolling",
                           "to": "sliding" if sliding else "rolling"})
        record(step + 1)
    return np.asarray(rows), events


def write_trace(path: Path, trace: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("time", "x", "y", "vx", "vy", "slip_norm", "sliding"))
        writer.writerows(trace)


def plot_case(path: Path, case_name: str, traces):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
    for dt, (trace, events) in traces.items():
        label = f"dt={dt:g}"
        axes[0].plot(trace[:, 1], trace[:, 2], label=label, lw=1.2)
        axes[1].plot(trace[:, 0], trace[:, 5], label=label, lw=1.1)
        if dt == DT_VALUES[-1]:
            for event in events:
                axes[1].axvline(event["time"], color="0.35", lw=0.8, ls="--")
    axes[0].set(xlabel="x", ylabel="y", title=case_name.replace("_", " "))
    axes[1].set(xlabel="time", ylabel="contact slip magnitude")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path,
                        default=ROOT / "build/sliding-exact8/extensions/fp64")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/sliding-switches-exact8")
    args = parser.parse_args()
    module = load_module(args.module_dir)
    params = module.SolverParams(*scaled_model_parameters(8, sr=0.428571, somega=0.1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"module": str(Path(module.__file__).resolve()),
               "field": "exact", "magnets": 8, "precision": "float64",
               "params": {name: float(getattr(params, name)) for name in
                          ("alpha", "delta", "gamma_g", "gamma", "gamma_m")},
               "total_time": TOTAL_TIME, "cases": {}}

    for case_name, case in CASES.items():
        traces = {}
        for dt in DT_VALUES:
            trace, events = run(module, params, case, dt)
            actual = tuple((event["from"], event["to"]) for event in events)
            if actual != EXPECTED_EVENTS[case_name]:
                raise AssertionError(f"{case_name}, dt={dt}: unexpected transitions {actual}")
            traces[dt] = trace, events
            write_trace(args.output_dir / f"{case_name}_dt_{dt:g}.csv", trace)
        reference = traces[DT_VALUES[-1]][0]
        comparison = {}
        for dt, (trace, events) in traces.items():
            stride = round(dt / DT_VALUES[-1])
            reference_aligned = reference[::stride]
            comparison[str(dt)] = {
                "events": events,
                "max_position_error_vs_finest": float(np.max(np.linalg.norm(
                    trace[:, 1:3] - reference_aligned[:, 1:3], axis=1))),
                "max_velocity_error_vs_finest": float(np.max(np.linalg.norm(
                    trace[:, 3:5] - reference_aligned[:, 3:5], axis=1))),
                "mode_mismatch_samples_vs_finest": int(np.count_nonzero(
                    trace[:, 6] != reference_aligned[:, 6])),
                "final_position": list(map(float, trace[-1, 1:3])),
            }
        summary["cases"][case_name] = {"initial": case, "steps": comparison}
        position_errors = [comparison[str(dt)]["max_position_error_vs_finest"] for dt in DT_VALUES]
        if any(coarse <= fine for coarse, fine in zip(position_errors[:-2], position_errors[1:-1])):
            raise AssertionError(f"{case_name}: position error did not decrease with step size")
        plot_case(args.output_dir / f"{case_name}.png", case_name, traces)
        print(case_name, json.dumps(comparison), flush=True)

    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
