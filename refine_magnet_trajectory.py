# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Repeat one exact-field magnet trajectory with a smaller integration step.

The sampling grid stays at 0.2 in 1D time, so results can be compared row by
row with compare_magnet_counts_exact.py's original trajectories.
"""

from __future__ import annotations

import argparse
import csv
import math

import numpy as np

from animate_one_trajectory import scaled_model_parameters, timestep_for_magnets
from compare_magnet_counts_exact import COUNTS, DATA_DIR, SAMPLE_TIME, TOTAL_TIME, load_extension, SOURCE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, choices=COUNTS)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    if args.factor < 2:
        parser.error("factor must be >= 2")
    count = args.count
    module = load_extension(SOURCE_DIR / "build" / "magnet-sweep" / f"exact-{count}" / "extensions" / "fp64")
    if module.magnet_number != count or module.field_mode != "exact":
        raise RuntimeError("Wrong exact-field extension")
    radius0 = count / 2
    dt = timestep_for_magnets(count) / args.factor
    steps_per_sample = round(SAMPLE_TIME / (radius0 * dt))
    if not math.isclose(steps_per_sample * radius0 * dt, SAMPLE_TIME):
        raise RuntimeError("Sample time is not aligned with refined integration step")
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    state[1], state[5] = radius0, 1
    params = module.SolverParams(*scaled_model_parameters(count))
    sliding = direction = None
    rows = []
    for index in range(round(TOTAL_TIME / SAMPLE_TIME) + 1):
        if index:
            state, sliding, direction = module.evolve_trajectory(
                state, dt, steps_per_sample, params, sliding, direction,
                (index - 1) * steps_per_sample)
        x, y, vx, vy = map(float, state[1:5])
        radius = math.hypot(x, y)
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite state at sample {index}")
        rows.append({"time_1d": index * SAMPLE_TIME,
                     "v_2d_equivalent": (x * vy - y * vx) / (radius * radius0),
                     "radius_offset": radius - radius0})
        if index and index % 1000 == 0:
            print(f"N={count}, refinement={args.factor}: {index}/{round(TOTAL_TIME / SAMPLE_TIME)}", flush=True)
    path = DATA_DIR / f"N{count}_dt_div_{args.factor}.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path, flush=True)


if __name__ == "__main__":
    main()
