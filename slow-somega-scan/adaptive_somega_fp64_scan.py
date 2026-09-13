"""Refine the s_omega locking boundary until adjacent FP64 values remain."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
from project_paths import SOMEGA_FP64_LOG_DIR, extension_dir  # noqa: E402

LOG_DIR = SOMEGA_FP64_LOG_DIR
sys.path.insert(0, str(extension_dir("fp64")))
import hysteresis  # noqa: E402


REFERENCE = hysteresis.SolverParams(
    0.33333333,
    1.7212357e-4,
    0.16560356,
    489.39639,
    11011.419,
)
FIXED_SR = 2.8
DT = 0.001
MAX_CONFIRM = 80000
THREADS = 24
TARGET_COUNT = 101
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?\s+")


def read_rows(path: Path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    for line in raw.decode(encoding).splitlines():
        if not NUMBER.match(line):
            continue
        fields = line.split()
        if len(fields) == 3:
            yield float(fields[0]), int(fields[1]), float(fields[2])


def current_bracket() -> tuple[float, float]:
    locked, unlocked = [], []
    for path in LOG_DIR.glob("slow_*.log"):
        for value, classification, transition_time in read_rows(path):
            if classification == 1 and transition_time >= 0:
                locked.append(value)
            elif classification == 0:
                unlocked.append(value)
    lower = max(locked)
    upper = min(value for value in unlocked if value > lower)
    return lower, upper


def next_index() -> int:
    indices = []
    for path in LOG_DIR.glob("slow_fine*.log"):
        suffix = path.stem.removeprefix("slow_fine")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=0) + 1


def save_log(path: Path, values: np.ndarray, answer: np.ndarray, times: np.ndarray, elapsed: float):
    frame_time = hysteresis.critical_slow_confirm_evolve_steps * DT
    lines = [
        f"Module: {hysteresis.__file__}",
        f"Mode: scan somega, fixed sr={FIXED_SR:g}; initial state: all zero",
        f"dt={DT:g}, frame_time={frame_time:g}, max_confirm={MAX_CONFIRM}, max_time={MAX_CONFIRM * frame_time:g}",
        f"lock_confirmation_time={hysteresis.critical_slow_lock_confirm_window * frame_time:g}, phase_rate_tolerance={hysteresis.critical_slow_phase_rate_tolerance:g}",
        f"{len(values)} trajectories, {THREADS} requested threads, {elapsed:.3f} s",
        "Columns: somega, classification, transition_time",
    ]
    lines.extend(
        f"{value:.17g}\t{int(classification)}\t{time:.17g}"
        for value, classification, time in zip(values, answer, times)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if np.dtype(hysteresis.real_dtype) != np.dtype(np.float64):
        raise RuntimeError(f"FP64 module required, got {hysteresis.real_dtype}")
    index = next_index()
    while True:
        lower, upper = current_bracket()
        successor = float(np.nextafter(lower, np.inf))
        print(
            f"bracket=[{lower:.17g}, {upper:.17g}], width={upper-lower:.18e}, "
            f"ulp={np.spacing(lower):.18e}",
            flush=True,
        )
        if successor >= upper:
            print("Stopped: locked and unlocked endpoints are adjacent FP64 values.")
            break

        values = np.unique(np.linspace(lower, upper, TARGET_COUNT, dtype=np.float64))
        interior = values[(values > lower) & (values < upper)]
        if interior.size == 0:
            print("Stopped: no representable interior FP64 value.")
            break
        values = np.concatenate(([lower], interior, [upper]))
        answer = np.empty(values.size, dtype=np.int32)
        times = np.empty(values.size, dtype=np.float64)
        start = perf_counter()
        hysteresis.calc_slow_cpu(
            answer,
            times,
            values,
            FIXED_SR,
            hysteresis.CriticalSlowScanMode.SOMEGA,
            DT,
            MAX_CONFIRM,
            REFERENCE,
            THREADS,
        )
        elapsed = perf_counter() - start
        output = LOG_DIR / f"slow_fine{index}.log"
        save_log(output, values, answer, times, elapsed)
        print(
            f"saved {output.name}: n={values.size}, locked={np.count_nonzero(answer == 1)}, "
            f"unlocked={np.count_nonzero(answer == 0)}, errors={np.count_nonzero(answer < 0)}, "
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )
        new_lower, new_upper = current_bracket()
        if new_lower == lower and new_upper == upper:
            print("Stopped: scan did not shrink the bracket.")
            break
        index += 1


if __name__ == "__main__":
    main()
