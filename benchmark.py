# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Time X steps of the current CPU build. Run with the build's Python environment."""

import argparse
import importlib.machinery
import importlib.util
from time import perf_counter

import numpy as np

from project_paths import extension_dir
from animate_one_trajectory import scaled_model_parameters, timestep_for_magnets


# Edit these values, then run: python src2d/benchmark.py
X = 1000000

# [disk angle, pos x/y, speed x/y, quaternion w/x/y/z,
#  omega x/y/z, hysteresis q x/y/z]
# At rest with an identity attitude quaternion. The radius is set below.
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
    parser.add_argument("--magnets", type=int, default=None,
                        help="check the compiled magnet count")
    args = parser.parse_args()
    if not 0 < args.steps <= 2**31 - 1:
        raise ValueError("X must be a positive int32 step count")

    # Search only the selected build; never silently load another installed module.
    directory = extension_dir("fp64")
    spec = importlib.machinery.PathFinder.find_spec("hysteresis", [str(directory)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No compatible hysteresis build in {directory}; use the build's Python")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    magnet_count = module.magnet_number
    if args.magnets is not None and args.magnets != magnet_count:
        raise ValueError(f"--magnets={args.magnets} does not match the compiled count {magnet_count}")
    dt = timestep_for_magnets(magnet_count)
    state = np.array(INITIAL_STATE, dtype=module.real_dtype)
    state[1] = magnet_count / 2
    params = module.SolverParams(*scaled_model_parameters(magnet_count))

    # Zero steps initializes any lazy LUT without advancing the initial state.
    module.evolve_trajectory(state, dt, 0, params)
    start = perf_counter()
    final_state, _, _ = module.evolve_trajectory(state, dt, args.steps, params)
    elapsed = perf_counter() - start

    if not np.isfinite(final_state).all():
        raise RuntimeError("Solver returned a non-finite final state")
    print(f"Build: {module.__file__}")
    print(f"Field: {module.field_mode}; dtype: {module.real_dtype}")
    print(f"Magnets: {magnet_count}; dt: {dt:g}")
    print(f"{args.steps} steps: {elapsed:.6f} s ({elapsed / args.steps * 1e6:.3f} us/step)")


if __name__ == "__main__":
    main()
