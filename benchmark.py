"""Time X steps of the current CPU build. Run with the build's Python environment."""

import argparse
import importlib.machinery
import importlib.util
from time import perf_counter

import numpy as np

from project_paths import extension_dir


# Edit these values, then run: python src2d/benchmark.py
X = 1000000
DT = 1.0e-3

# Same physical parameters as animate_one_trajectory.py.
SR = 0.428571
SOMEGA = 0.1

# [disk angle, pos x/y, speed x/y, quaternion w/x/y/z,
#  omega x/y/z, hysteresis q x/y/z]
# At rest at radius 4, with an identity attitude quaternion (8-magnet model).
INITIAL_STATE = [
    0.0,
    4.0, 0.0,
    0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
]


def main():
    parser = argparse.ArgumentParser(description="Time a complete C++ 2D trajectory")
    parser.add_argument("--steps", type=int, default=X)
    parser.add_argument("--magnets", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.steps <= 2**31 - 1:
        raise ValueError("X must be a positive int32 step count")
    if args.magnets < 2 or args.magnets % 2:
        raise ValueError("magnet count must be positive and even")

    # Search only the selected build; never silently load another installed module.
    directory = extension_dir("fp64")
    spec = importlib.machinery.PathFinder.find_spec("hysteresis", [str(directory)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No compatible hysteresis build in {directory}; use the build's Python")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = np.array(INITIAL_STATE, dtype=module.real_dtype)
    state[1] = args.magnets / 2
    params = module.SolverParams(
        0.33333333 * SR,
        1.7212357e-4 / SR,
        2.64965696 / SOMEGA**2,
        7830.34224 / SOMEGA**2,
        176182.704 / (SR**2 * SOMEGA**2),
    )

    # Zero steps initializes any lazy LUT without advancing the initial state.
    module.evolve_trajectory(state, DT, 0, params)
    start = perf_counter()
    final_state, _, _ = module.evolve_trajectory(state, DT, args.steps, params)
    elapsed = perf_counter() - start

    if not np.isfinite(final_state).all():
        raise RuntimeError("Solver returned a non-finite final state")
    print(f"Build: {module.__file__}")
    print(f"Field: {module.field_mode}; dtype: {module.real_dtype}")
    print(f"{args.steps} steps: {elapsed:.6f} s ({elapsed / args.steps * 1e6:.3f} us/step)")


if __name__ == "__main__":
    main()
