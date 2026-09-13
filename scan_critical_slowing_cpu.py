"""Scan critical slowing versus one scale parameter on the CPU backend.

Build first with::

    python src/build.py --fp64

For double-double arithmetic (about 31 significant decimal digits), build and
select the companion module with::

    python src/build.py --dd
    python src/scan_critical_slowing_cpu.py --dd ...

Two scan modes are available; every trajectory starts from the all-zero state::

    # Fix sr and scan somega
    python src/scan_critical_slowing_cpu.py --mode somega --fixed 0.8

    # Fix somega and scan sr
    python src/scan_critical_slowing_cpu.py --mode sr --fixed 0.6

No array may be mutated or resized by another Python thread during the call.
"""

import argparse
from decimal import Decimal, InvalidOperation, localcontext
import importlib
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from project_paths import extension_dir


REFERENCE_ALPHA = '0.33333333'
REFERENCE_DELTA = '1.7212357e-4'
REFERENCE_GAMMA_G = '0.16560356'
REFERENCE_GAMMA = '489.39639'
REFERENCE_GAMMA_M = '11011.419'

MODE_DEFAULTS = {
    # mode: (fixed value, scan lower, scan upper)
    'somega': ('0.8', '0.1', '3.0'),
    'sr': ('0.6', '0.1', '4.5'),
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
        title=rf'Phase-locking time versus ${labels[mode]}$ (fixed ${labels[fixed_name]}={fixed}$)',
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
    parser.add_argument(
        '--dd', action='store_true',
        help='use the double-double CPU module and preserve decimal command-line inputs',
    )
    parser.add_argument('--fixed', help='fixed value of the other scale parameter')
    parser.add_argument('--lower', help='lower endpoint of the scanned scale')
    parser.add_argument('--upper', help='upper endpoint of the scanned scale')
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
    fixed_text = default_fixed if args.fixed is None else args.fixed
    lower_text = default_lower if args.lower is None else args.lower
    upper_text = default_upper if args.upper is None else args.upper
    try:
        fixed_decimal = Decimal(fixed_text)
        lower_decimal = Decimal(lower_text)
        upper_decimal = Decimal(upper_text)
    except InvalidOperation:
        parser.error('--fixed, --lower and --upper must be decimal numbers')
    if args.count < 0:
        parser.error('--count must be nonnegative')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    if not fixed_decimal.is_finite() or fixed_decimal <= 0:
        parser.error('--fixed must be finite and positive')
    if not lower_decimal.is_finite() or lower_decimal <= 0:
        parser.error('--lower must be finite and positive')
    if not upper_decimal.is_finite() or upper_decimal <= 0:
        parser.error('--upper must be finite and positive')
    if upper_decimal < lower_decimal:
        parser.error('--upper must be greater than or equal to --lower')
    if args.max_confirm is not None and args.max_confirm < 0:
        parser.error('--max-confirm must be nonnegative')

    module_name = 'hysteresis_dd' if args.dd else 'hysteresis'
    module_dir = extension_dir('dd' if args.dd else 'fp64')
    if module_dir.is_dir():
        sys.path.insert(0, str(module_dir))
    try:
        hysteresis = importlib.import_module(module_name)
    except ImportError as error:
        build_flag = '--dd' if args.dd else '--fp64'
        raise RuntimeError(f'Build the {module_name} module first: python src/build.py {build_flag}') from error
    kernel_name = 'calc_slow_cpu_dd' if args.dd else 'calc_slow_cpu'
    if not hasattr(hysteresis, kernel_name) or not hasattr(hysteresis, 'CriticalSlowScanMode'):
        build_flag = '--dd' if args.dd else '--fp64'
        raise RuntimeError(f'Build the updated module first: python src/build.py {build_flag}')

    dt_text = '0.001'
    dt = float(dt_text)
    frame_time = hysteresis.critical_slow_confirm_evolve_steps * dt
    max_confirm = args.max_confirm if args.max_confirm is not None else round(80000.0 / frame_time)
    if args.dd:
        with localcontext() as context:
            context.prec = 50
            if args.count == 0:
                scan_input = []
            elif args.count == 1:
                scan_input = [str(lower_decimal)]
            else:
                step = (upper_decimal - lower_decimal) / (args.count - 1)
                scan_input = [str(lower_decimal + step * index) for index in range(args.count)]
                scan_input[-1] = str(upper_decimal)
        plot_values = np.asarray([float(value) for value in scan_input], dtype=np.float64)
        reference_values = (
            REFERENCE_ALPHA, REFERENCE_DELTA, REFERENCE_GAMMA_G,
            REFERENCE_GAMMA, REFERENCE_GAMMA_M,
        )
        fixed = fixed_text
    else:
        fixed = float(fixed_decimal)
        lower = float(lower_decimal)
        upper = float(upper_decimal)
        if not all(np.isfinite(value) for value in (fixed, lower, upper)):
            parser.error('--fixed, --lower and --upper must fit in FP64 unless --dd is used')
        scan_input = np.linspace(lower, upper, args.count, dtype=np.float64)
        plot_values = scan_input
        reference_values = tuple(float(value) for value in (
            REFERENCE_ALPHA, REFERENCE_DELTA, REFERENCE_GAMMA_G,
            REFERENCE_GAMMA, REFERENCE_GAMMA_M,
        ))
    reference = hysteresis.SolverParams(*reference_values)
    scan_mode = (
        hysteresis.CriticalSlowScanMode.SOMEGA
        if args.mode == 'somega'
        else hysteresis.CriticalSlowScanMode.SR
    )
    fixed_name = 'sr' if args.mode == 'somega' else 'somega'

    start = perf_counter()
    print(f'Module: {hysteresis.__file__}', flush=True)
    backend_name = getattr(hysteresis, 'real_backend', 'double')
    print(f'Backend: {backend_name}', flush=True)
    print(f'Mode: scan {args.mode}, fixed {fixed_name}={fixed}; initial state: all zero', flush=True)
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
    if args.dd:
        answer, times = hysteresis.calc_slow_cpu_dd(
            scan_input, fixed_text, scan_mode, dt_text, max_confirm, reference, args.threads,
        )
        answer = np.asarray(answer)
        times = np.asarray(times)
        printed_scales = scan_input
    else:
        answer = np.empty(scan_input.size, dtype=np.int32)
        times = np.empty(scan_input.size, dtype=np.float64)
        hysteresis.calc_slow_cpu(
            answer, times, scan_input, fixed, scan_mode, dt, max_confirm, reference, args.threads,
        )
        printed_scales = (f'{scale:.17g}' for scale in scan_input)
    print(f'{len(scan_input)} trajectories, {args.threads} requested threads, {perf_counter() - start:.3f} s')
    print(f'Columns: {args.mode}, classification, transition_time')
    for scale, classification, transition_time in zip(printed_scales, answer, times):
        print(f'{scale}\t{int(classification)}\t{transition_time:.17g}')
    sys.stdout.flush()

    figure = plot_results(plot_values, answer, times, args.mode, fixed)
    if args.save_plot is not None:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_plot, dpi=180)
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
