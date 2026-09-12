"""Scan critical slowing versus one scale parameter on the FP64 CPU backend.

Build first with::

    python src/build.py --fp64

Two scan modes are available; every trajectory starts from the all-zero state::

    # Fix sr and scan somega
    python src/scan_critical_slowing_cpu.py --mode somega --fixed 0.8

    # Fix somega and scan sr
    python src/scan_critical_slowing_cpu.py --mode sr --fixed 0.6

No array may be mutated or resized by another Python thread during the call.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import hysteresis
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_ALPHA = 0.33333333
REFERENCE_DELTA = 1.7212357e-4
REFERENCE_GAMMA_G = 0.16560356
REFERENCE_GAMMA = 489.39639
REFERENCE_GAMMA_M = 11011.419

MODE_DEFAULTS = {
    # mode: (fixed value, scan lower, scan upper)
    'somega': (0.8, 0.1, 3.0),
    'sr': (0.6, 0.1, 4.5),
}


def plot_results(scan_values, answer, times, mode, fixed):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    valid = (answer == 1) & np.isfinite(times) & (times >= 0)
    # Keep every scan position: NaNs break lines at non-locking/error points.
    axis.plot(scan_values, np.where(valid, times, np.nan), 'o-', label='phase locked', markersize=4)
    fixed_name = 'sr' if mode == 'somega' else 'somega'
    labels = {'sr': r's_r', 'somega': r's_\omega'}
    axis.set(
        xlabel=rf'${labels[mode]}$',
        ylabel='Phase-locking time',
        title=rf'Phase-locking time versus ${labels[mode]}$ (fixed ${labels[fixed_name]}={fixed:g}$)',
    )
    axis.ticklabel_format(axis='x', style='plain', useOffset=False)
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=('somega', 'sr'), default='somega',
        help='parameter to scan: somega fixes sr; sr fixes somega (default: somega)',
    )
    parser.add_argument('--fixed', type=float, help='fixed value of the other scale parameter')
    parser.add_argument('--lower', type=float, help='lower endpoint of the scanned scale')
    parser.add_argument('--upper', type=float, help='upper endpoint of the scanned scale')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument(
        '--max-confirm', type=int, default=None,
        help='maximum sampling frames (default: enough for 80000 time units)',
    )
    parser.add_argument('--no-show', action='store_true', help='do not open the plot window')
    parser.add_argument('--save-plot', type=Path, help='save the result figure')
    args = parser.parse_args()

    default_fixed, default_lower, default_upper = MODE_DEFAULTS[args.mode]
    fixed = default_fixed if args.fixed is None else args.fixed
    lower = default_lower if args.lower is None else args.lower
    upper = default_upper if args.upper is None else args.upper
    if args.count < 0:
        parser.error('--count must be nonnegative')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    if not np.isfinite(fixed) or fixed <= 0.0:
        parser.error('--fixed must be finite and positive')
    if not np.isfinite(lower) or lower <= 0.0:
        parser.error('--lower must be finite and positive')
    if not np.isfinite(upper) or upper <= 0.0:
        parser.error('--upper must be finite and positive')
    if upper < lower:
        parser.error('--upper must be greater than or equal to --lower')
    if args.max_confirm is not None and args.max_confirm < 0:
        parser.error('--max-confirm must be nonnegative')
    if not hasattr(hysteresis, 'calc_slow_cpu') or not hasattr(hysteresis, 'CriticalSlowScanMode'):
        raise RuntimeError('Build the updated FP64 module first: python src/build.py --fp64')

    dt = 0.001
    frame_time = hysteresis.critical_slow_confirm_evolve_steps * dt
    max_confirm = args.max_confirm if args.max_confirm is not None else round(80000.0 / frame_time)
    scan_values = np.linspace(lower, upper, args.count, dtype=np.float64)
    answer = np.empty(scan_values.size, dtype=np.int32)
    times = np.empty(scan_values.size, dtype=np.float64)
    reference = hysteresis.SolverParams(
        REFERENCE_ALPHA,
        REFERENCE_DELTA,
        REFERENCE_GAMMA_G,
        REFERENCE_GAMMA,
        REFERENCE_GAMMA_M,
    )
    scan_mode = (
        hysteresis.CriticalSlowScanMode.SOMEGA
        if args.mode == 'somega'
        else hysteresis.CriticalSlowScanMode.SR
    )
    fixed_name = 'sr' if args.mode == 'somega' else 'somega'

    start = perf_counter()
    print(f'Module: {hysteresis.__file__}', flush=True)
    print(f'Mode: scan {args.mode}, fixed {fixed_name}={fixed:g}; initial state: all zero', flush=True)
    print(
        f'dt={dt:g}, frame_time={frame_time:g}, max_confirm={max_confirm}, '
        f'max_time={max_confirm * frame_time:g}',
        flush=True,
    )
    print(
        f'lock_confirmation_time={hysteresis.critical_slow_lock_confirm_window * frame_time:g}, '
        f'phase_rate_tolerance={hysteresis.critical_slow_phase_rate_tolerance:g}',
        flush=True,
    )
    hysteresis.calc_slow_cpu(
        answer, times, scan_values, fixed, scan_mode, dt, max_confirm, reference, args.threads,
    )
    print(f'{scan_values.size} trajectories, {args.threads} requested threads, {perf_counter() - start:.3f} s')
    print(f'Columns: {args.mode}, classification, transition_time')
    for scale, classification, transition_time in zip(scan_values, answer, times):
        print(f'{scale:.17g}\t{int(classification)}\t{transition_time:.17g}')
    sys.stdout.flush()

    figure = plot_results(scan_values, answer, times, args.mode, fixed)
    if args.save_plot is not None:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_plot, dpi=180)
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
