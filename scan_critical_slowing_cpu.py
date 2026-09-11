"""Example CPU array launch. Build first with: python src/build.py --fp64.

Run: python src/scan_critical_slowing_cpu.py --threads 24 --count 500
All trajectories share params/dt/max_confirm; v0[i] is supplied explicitly.
No array may be mutated/resized by another Python thread during the call.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import hysteresis
import matplotlib.pyplot as plt
import numpy as np


def plot_results(v0, answer, times):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for classification, label in ((1, '1: locked'), (2, '2: rising')):
        valid = (answer == classification) & np.isfinite(times) & (times >= 0)
        # Keep every scan position: NaNs break lines at missing/other-class points.
        axis.plot(v0, np.where(valid, times, np.nan), 'o-', label=label, markersize=4)
    axis.set(xlabel=r'Initial velocity $v_0$', ylabel='Transition time',
             title='Critical slowing: transition time versus initial velocity')
    axis.ticklabel_format(axis='x', style='plain', useOffset=False)
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--max-confirm', type=int, default=None,
                        help='maximum sampling frames (default: enough for 80000 time units)')
    parser.add_argument('--no-show', action='store_true', help='do not open the plot window')
    parser.add_argument('--save-plot', type=Path, help='save the result figure')
    args = parser.parse_args()
    if args.count < 0:
        parser.error('--count must be nonnegative')
    if not hasattr(hysteresis, 'calc_slow_cpu'):
        raise RuntimeError('Build the FP64 module first: python src/build.py --fp64')
    dt = 0.001
    frame_time = hysteresis.critical_slow_confirm_evolve_steps * dt
    max_confirm = args.max_confirm if args.max_confirm is not None else round(80000.0 / frame_time)

    # Data preparation only: replace with any explicit FP64 velocity samples.
    v0 = np.linspace(0.18900, 0.18922, args.count, dtype=np.float64)
    answer = np.empty(v0.size, dtype=np.int32)
    times = np.empty(v0.size, dtype=np.float64)
    sr, somega = 0.8, 0.6
    params = hysteresis.SolverParams(0.33333333 * sr, 1.7212357e-4 / sr,
                                    0.16560356 / somega**2, 489.39639 / somega**2,
                                    11011.419 / sr**2 / somega**2)
    start = perf_counter()
    print(f'Module: {hysteresis.__file__}', flush=True)
    print(f'dt={dt:g}, frame_time={frame_time:g}, max_confirm={max_confirm}, '
          f'max_time={max_confirm * frame_time:g}', flush=True)
    print(f'lock_confirmation_time={hysteresis.critical_slow_lock_confirm_window * frame_time:g}, '
          f'recovery_confirmation_time={hysteresis.critical_slow_confirm_window * frame_time:g}', flush=True)
    hysteresis.calc_slow_cpu(answer, times, v0, dt, max_confirm, params, args.threads)
    print(f'{v0.size} trajectories, {args.threads} requested threads, {perf_counter() - start:.3f} s')
    print('Columns: v0, classification, transition_time')
    for velocity, classification, transition_time in zip(v0, answer, times):
        print(f'{velocity:.17g}\t{int(classification)}\t{transition_time:.17g}')
    sys.stdout.flush()
    figure = plot_results(v0, answer, times)
    if args.save_plot is not None:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_plot, dpi=180)
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
