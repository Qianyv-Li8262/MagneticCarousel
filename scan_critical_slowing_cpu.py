"""Example CPU array launch. Build first with: python src/build.py --fp64.

Run: python src/run_critical_slow_cpu.py --threads 8 --count 32 --max-confirm 200
All trajectories share params/dt/max_confirm; v0[i] is supplied explicitly.
No array may be mutated/resized by another Python thread during the call.
"""
import argparse
from time import perf_counter

import hysteresis
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--max-confirm', type=int, default=1600)
    args = parser.parse_args()
    if args.count < 0:
        parser.error('--count must be nonnegative')
    if not hasattr(hysteresis, 'calc_slow_cpu'):
        raise RuntimeError('Build the FP64 module first: python src/build.py --fp64')

    # Data preparation only: replace with any explicit FP64 velocity samples.
    v0 = np.linspace(0.1890, 0.1894, args.count, dtype=np.float64)
    answer = np.empty(v0.size, dtype=np.int32)
    times = np.empty(v0.size, dtype=np.float64)
    sr, somega = 0.8, 0.6
    params = hysteresis.SolverParams(0.33333333 * sr, 1.7212357e-4 / sr,
                                    0.16560356 / somega**2, 489.39639 / somega**2,
                                    11011.419 / sr**2 / somega**2)
    start = perf_counter()
    hysteresis.calc_slow_cpu(answer, times, v0, 0.001, args.max_confirm, params, args.threads)
    print(f'{v0.size} trajectories, {args.threads} requested threads, {perf_counter() - start:.3f} s')
    print('Columns: v0, classification, transition_time')
    print(np.column_stack((v0, answer, times)))


if __name__ == '__main__':
    main()
