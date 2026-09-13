"""Evaluate every FP64 value in the last monotone (fine9) bracket."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from project_paths import SOMEGA_FP64_LOG_DIR, extension_dir  # noqa: E402

LOG_DIR = SOMEGA_FP64_LOG_DIR
sys.path.insert(0, str(extension_dir("fp64")))
import hysteresis  # noqa: E402

from adaptive_somega_fp64_scan import DT, FIXED_SR, MAX_CONFIRM, REFERENCE, THREADS, read_rows, save_log  # noqa: E402


def main():
    source = LOG_DIR / "slow_fine9.log"
    data = list(read_rows(source))
    locked = [x for x, c, t in data if c == 1 and t >= 0]
    unlocked = [x for x, c, _ in data if c == 0]
    lower = max(locked)
    upper = min(x for x in unlocked if x > lower)
    lower_bits = np.asarray(lower, dtype=np.float64).view(np.uint64).item()
    upper_bits = np.asarray(upper, dtype=np.float64).view(np.uint64).item()
    bits = np.arange(lower_bits, upper_bits + 1, dtype=np.uint64)
    values = bits.view(np.float64)
    answer = np.empty(values.size, dtype=np.int32)
    times = np.empty(values.size, dtype=np.float64)
    start = perf_counter()
    hysteresis.calc_slow_cpu(
        answer, times, values, FIXED_SR, hysteresis.CriticalSlowScanMode.SOMEGA,
        DT, MAX_CONFIRM, REFERENCE, THREADS,
    )
    elapsed = perf_counter() - start
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output = LOG_DIR / "slow_fp64_exhaustive.log"
    save_log(output, values, answer, times, elapsed)
    print(f"range=[{lower:.17g}, {upper:.17g}], values={values.size}, elapsed={elapsed:.3f}s")
    print(f"locked={np.count_nonzero(answer == 1)}, unlocked={np.count_nonzero(answer == 0)}")
    transitions = np.flatnonzero(answer[1:] != answer[:-1]) + 1
    print(f"classification transitions={len(transitions)}")
    for index in transitions:
        print(f"  ulp_index={index}, value={values[index]:.17g}, {answer[index-1]}->{answer[index]}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
