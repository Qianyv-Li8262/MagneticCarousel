# SPDX-License-Identifier: AGPL-3.0-only
"""Run the current CPU build and overlay its trajectory on the 56c6 experiment.

Run with the daily Python. Edit the constants below for release/plot settings.
Material, rolling and field parameters live in kernels/headers and require a
CMake rebuild. This script never imports the isolated fitting executable.
The optional --no-show switch saves the same figure without opening a window.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from animate_one_trajectory import load_hysteresis, read_build_config

ROOT = Path(__file__).resolve().parent

# ===== Edit release conditions here: all lengths/speeds below are physical SI =====
BALL_RADIUS_SCALE = 1.0                 # 6 mm reference ball radius
DISK_RPM = 200.0
INITIAL_RADIUS_MM = 64.17102990498384
INITIAL_AZIMUTH_RAD = 0.0
INITIAL_TANGENTIAL_SPEED_MM_S = 0.0     # positive = counterclockwise
INITIAL_RADIAL_SPEED_MM_S = 0.0         # positive = outward
INITIAL_SPIN_Z_RAD_S = 0.0
INITIAL_ATTITUDE_WXYZ = (1.0, 0.0, 0.0, 0.0)
INITIAL_JA_MEMORY = (0.0, 0.0, 0.0)     # dimensionless q; all eddy memories start at zero
DISK_PHASE_RAD = 0.58
# No holding/pre-magnetization, blowing or spin resets.

# ===== Integration and comparison settings =====
DT = 5.0e-4                            # dimensionless disk-angular time step
RUN_SECONDS = 240.0                    # physical seconds AFTER release
SAMPLE_CODE_TIME = 0.1                 # same sampling interval as the fitting runs
RADIUS_AVERAGE_SECONDS = 5.0           # full centered window, BOTH experiment and model
# Speed retains its 1-second average to preserve the startup-time comparison.
VIDEO_RELEASE_PTS_S = 4.9
SPACE_SCALE = 1.017099618690604          # physical length = original CV length * this
# SPACE_SCALE scales BOTH model radius and speed back into the CV coordinates.
# Changing INITIAL_RADIUS_MM does not silently change SPACE_SCALE.
VIDEO_NAME = '56c6fd6092d21301a3c7808b624152c1'
VIDEO_DIR = ROOT.parent / 'cv' / 'kinematics_latest6' / VIDEO_NAME
OUTPUT_DIR = ROOT / 'scratch' / 'video56c6-comparison'
CURVE_LABEL = 'Current build'
# Optional saved trajectory to overlay while tuning (same CSV schema as this script).
REFERENCE_CSV = None                   # e.g. ROOT / 'scratch' / 'saved_run.csv'
REFERENCE_SPACE_SCALE = 1.017099618690604
REFERENCE_LABEL = 'Reference run'


def initial_state(module, params, length_mm, omega):
    state = np.zeros(module.state_size, dtype=module.real_dtype)
    er = np.array([np.cos(INITIAL_AZIMUTH_RAD), np.sin(INITIAL_AZIMUTH_RAD)])
    et = np.array([-er[1], er[0]])
    state[0] = DISK_PHASE_RAD
    state[1:3] = INITIAL_RADIUS_MM / length_mm * er
    state[3:5] = (INITIAL_RADIAL_SPEED_MM_S * er + INITIAL_TANGENTIAL_SPEED_MM_S * et) / (length_mm * omega)
    attitude = np.asarray(INITIAL_ATTITUDE_WXYZ, dtype=float)
    if attitude.shape != (4,) or not np.isfinite(attitude).all() or np.linalg.norm(attitude) == 0:
        raise ValueError('INITIAL_ATTITUDE_WXYZ must be a finite nonzero quaternion')
    state[5:9] = attitude / np.linalg.norm(attitude)
    # Horizontal spin starts in pure rolling; normal spin is independently editable.
    state[9:11] = [-state[4] / params.alpha, state[3] / params.alpha]
    state[11] = INITIAL_SPIN_Z_RAD_S / omega
    state[12:15] = INITIAL_JA_MEMORY
    return state


def simulate(module, params, length_mm, omega):
    count = int(round(RUN_SECONDS * omega / DT))
    stride = max(1, int(round(SAMPLE_CODE_TIME / DT)))
    state = initial_state(module, params, length_mm, omega)
    samples = []; step = 0; sliding = direction = None; next_progress = 20.0
    while True:
        radius = np.linalg.norm(state[1:3]); velocity = state[3:5] * length_mm * omega
        er = state[1:3] / radius
        samples.append((step * DT / omega, np.linalg.norm(velocity),
            er[0] * velocity[1] - er[1] * velocity[0], er @ velocity,
            radius * length_mm, state[11] * omega, np.arctan2(state[2], state[1]),
            np.linalg.norm(state[12:15]), bool(sliding)))
        if step == count:
            break
        span = min(stride, count - step)
        state[0] = DISK_PHASE_RAD + step * DT
        state, sliding, direction = module.evolve_trajectory(
            state, DT, span, params, sliding, direction, step)
        step += span
        if step * DT / omega >= next_progress:
            print(f'  simulated {step * DT / omega:.1f} / {RUN_SECONDS:g} s', flush=True)
            next_progress += 20.0
    return pd.DataFrame(samples, columns=['time_s', 'speed_mm_s', 'vt_mm_s', 'vr_mm_s',
        'radius_mm', 'wz_rad_s', 'phi_rad', 'q_norm', 'sliding'])


def centered_time_mean(t, values, centers, window):
    """Full-window time-weighted means; never pad or shrink endpoint windows."""
    t = np.asarray(t); values = np.asarray(values); centers = np.asarray(centers)
    integral = np.r_[0., np.cumsum(np.diff(t) * (values[1:] + values[:-1]) / 2)]
    def primitive(x):
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        delta = x - t[i]
        slope = (values[i + 1] - values[i]) / (t[i + 1] - t[i])
        return integral[i] + values[i] * delta + .5 * slope * delta**2
    valid = (centers - window / 2 >= t[0]) & (centers + window / 2 <= t[-1])
    result = np.full(len(centers), np.nan)
    x = centers[valid]
    result[valid] = (primitive(x + window / 2) - primitive(x - window / 2)) / window
    return result


def experiment_bins(raw):
    bins = []
    for lo in np.arange(7., 72.):
        w = (raw.source_pts_s >= lo) & (raw.source_pts_s < lo + 1)
        v = w & (raw.derivative_valid == 1); r = w & (raw.position_valid == 1)
        if v.sum() >= 5 and r.sum() >= 5:
            bins.append((lo + .5, raw.com_speed_mm_s[v].mean(), raw.r_mm[r].mean()))
    bins = np.asarray(bins)
    good = (raw.position_valid == 1) & np.isfinite(raw.r_mm) & (raw.source_pts_s >= VIDEO_RELEASE_PTS_S)
    positions = raw.loc[good].sort_values('source_pts_s').groupby('source_pts_s', as_index=False).r_mm.mean()
    if np.diff(positions.source_pts_s).max() > .25:
        raise ValueError('Position gap exceeds .25 s; split the signal before radius averaging')
    bins[:, 2] = centered_time_mean(positions.source_pts_s, positions.r_mm, bins[:, 0], RADIUS_AVERAGE_SECONDS)
    return bins


def smooth(a, key, scale):
    if key == 'radius_mm':
        values = centered_time_mean(a.time_s, a[key], a.time_s, RADIUS_AVERAGE_SECONDS)
        valid = np.isfinite(values)
        return a.time_s.to_numpy()[valid] + VIDEO_RELEASE_PTS_S, values[valid] / scale
    n = max(1, round(1 / np.median(np.diff(a.time_s))))
    if len(a) < n:
        raise ValueError('Run at least one second for a 1-second mean')
    kernel = np.ones(n) / n
    return (np.convolve(a.time_s + VIDEO_RELEASE_PTS_S, kernel, 'valid'),
            np.convolve(a[key], kernel, 'valid') / scale)


def complete_laps(a):
    # Only use a consistently clockwise orbital interval, as in the fit report.
    w = a.time_s.to_numpy() > 15
    t = a.time_s.to_numpy()[w] + VIDEO_RELEASE_PTS_S
    phase = -np.unwrap(a.phi_rad.to_numpy())[w]
    if len(t) < 2 or np.any(np.diff(phase) <= 0):
        return np.empty((0, 4))
    levels = 2 * np.pi * np.arange(np.floor(phase[0] / (2 * np.pi)) + 1,
                                  np.floor(phase[-1] / (2 * np.pi)) + 1)
    cross = np.interp(levels, phase, t)
    return np.column_stack(((cross[1:] + cross[:-1]) / 2, np.diff(cross), cross[:-1], cross[1:]))


def plot_and_measure(a, target, video_laps, config, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    fig.suptitle('56c6 experiment vs current compiled model', fontsize=16)
    axes[0, 0].plot(target[:, 0], target[:, 1], 'k.', ms=4, label='Experiment: 1 s mean')
    axes[0, 1].plot(target[:, 0], target[:, 2], 'k.', ms=4, label=f'Experiment: {RADIUS_AVERAGE_SECONDS:g} s mean')
    curves = [(a, SPACE_SCALE, '#059669', CURVE_LABEL)]
    if REFERENCE_CSV is not None:
        curves.insert(0, (pd.read_csv(REFERENCE_CSV), REFERENCE_SPACE_SCALE, '#94a3b8', REFERENCE_LABEL))
    for frame, scale, color, label in curves:
        for axis, key in [(axes[0, 0], 'speed_mm_s'), (axes[0, 1], 'radius_mm')]:
            x, y = smooth(frame, key, scale); axis.plot(x, y, color=color, lw=1.7, label=label)
        laps = complete_laps(frame)
        if len(laps):
            axes[1, 0].plot(laps[:, 0], laps[:, 1], color=color, label=label)
        x, y = smooth(frame, 'radius_mm', scale)
        axes[1, 1].plot(x - VIDEO_RELEASE_PTS_S, y, color=color, label=label)
    w = (video_laps.start_pts_s >= 20) & (video_laps.end_pts_s <= 72)
    axes[1, 0].plot((video_laps.start_pts_s[w] + video_laps.end_pts_s[w]) / 2,
                    video_laps.period_s[w], 'k.', label='Experiment: complete laps')
    w = (video_laps.start_pts_s >= 35) & (video_laps.end_pts_s <= 72)
    period_target = float(video_laps.period_s[w].mean())
    axes[1, 0].axhline(period_target, color='gray', ls=':')
    axes[0, 0].set(xlim=(5, 42), xlabel='Video PTS (s)', ylabel='Speed / scale (mm/s)', title='Waiting and startup; release at 4.9 s')
    axes[0, 1].set(xlim=(5, 72), xlabel='Video PTS (s)', ylabel='Radius / scale (mm)', title=f'{RADIUS_AVERAGE_SECONDS:g} s radius mean; spatial scale {SPACE_SCALE:.6f}')
    axes[1, 0].set(xlim=(22, 72), ylim=(1.3, 2.6), xlabel='Video PTS (s)', ylabel='Complete-lap period (s)', title='Period in the observed interval')
    axes[1, 1].set(xlim=(0, RUN_SECONDS), xlabel='Time after release (s)', ylabel='Radius / scale (mm)', title='Long run: check for a delayed overshoot')
    for axis in axes.flat:
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / 'comparison.png', dpi=155)
    # Second view: scale the experimental spatial calibration into physical units.
    # This is the SAME fitted global factor, not a new radius-only fit; velocities
    # must receive the same factor because the video time axis is unchanged.
    physical_fig, physical_axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    physical_fig.suptitle(f'56c6: experiment scaled by {SPACE_SCALE:.6f} vs physical model', fontsize=16)
    physical_axes[0, 0].plot(target[:, 0], target[:, 1] * SPACE_SCALE, 'k.', ms=4, label='Experiment x spatial scale')
    physical_axes[0, 1].plot(target[:, 0], target[:, 2] * SPACE_SCALE, 'k.', ms=4, label=f'Experiment x scale: {RADIUS_AVERAGE_SECONDS:g} s mean')
    for frame, _, color, label in curves:
        for axis, key in [(physical_axes[0, 0], 'speed_mm_s'), (physical_axes[0, 1], 'radius_mm')]:
            x, y = smooth(frame, key, 1.0)
            axis.plot(x, y, color=color, lw=1.7, label=label)
        frame_laps = complete_laps(frame)
        if len(frame_laps):
            physical_axes[1, 0].plot(frame_laps[:, 0], frame_laps[:, 1], color=color, label=label)
        x, y = smooth(frame, 'radius_mm', 1.0)
        physical_axes[1, 1].plot(x - VIDEO_RELEASE_PTS_S, y, color=color, label=label)
    w = (video_laps.start_pts_s >= 20) & (video_laps.end_pts_s <= 72)
    physical_axes[1, 0].plot((video_laps.start_pts_s[w] + video_laps.end_pts_s[w]) / 2,
                            video_laps.period_s[w], 'k.', label='Experiment: complete laps')
    physical_axes[1, 0].axhline(period_target, color='gray', ls=':')
    physical_axes[1, 1].plot(target[:, 0] - VIDEO_RELEASE_PTS_S, target[:, 2] * SPACE_SCALE,
                            'k.', ms=3, label='Experiment x spatial scale')
    physical_axes[0, 0].set(xlim=(5, 42), xlabel='Video PTS (s)', ylabel='Physical speed (mm/s)', title='Experiment speed x scale; model unchanged')
    physical_axes[0, 1].set(xlim=(5, 72), xlabel='Video PTS (s)', ylabel='Physical radius (mm)', title=f'Both radii: {RADIUS_AVERAGE_SECONDS:g} s centered mean')
    physical_axes[1, 0].set(xlim=(22, 72), ylim=(1.3, 2.6), xlabel='Video PTS (s)', ylabel='Complete-lap period (s)', title='Time and periods remain unchanged')
    physical_axes[1, 1].set(xlim=(0, RUN_SECONDS), xlabel='Time after release (s)', ylabel='Physical radius (mm)', title='Long run in physical units')
    for axis in physical_axes.flat:
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    physical_fig.savefig(OUTPUT_DIR / 'comparison_scaled_experiment.png', dpi=155)
    # Metrics follow the displayed windows: speed 1 s, radius configurable.
    vm = []; rm = []
    for pts in target[:, 0]:
        w = (a.time_s >= pts - VIDEO_RELEASE_PTS_S - .5) & (a.time_s < pts - VIDEO_RELEASE_PTS_S + .5)
        vm.append(a.speed_mm_s[w].mean() / SPACE_SCALE)
        rm.append(centered_time_mean(a.time_s, a.radius_mm,
                  np.array([pts - VIDEO_RELEASE_PTS_S]), RADIUS_AVERAGE_SECONDS)[0] / SPACE_SCALE)
    vm = np.asarray(vm); rm = np.asarray(rm)
    laps = complete_laps(a); w = (laps[:, 2] >= 35) & (laps[:, 3] <= 72)
    quiet = (target[:, 0] >= 9) & (target[:, 0] < 18)
    startup = target[:, 0] < 35
    tail = a.time_s >= max(0, RUN_SECONDS - 20)
    _, radius = smooth(a, 'radius_mm', SPACE_SCALE)
    metrics = dict(experiment_period_s=period_target,
        model_period_s=float(laps[w, 1].mean()) if w.any() else None,
        experiment_quiet_speed_mm_s=float(target[quiet, 1].mean()),
        model_quiet_speed_mm_s=float(vm[quiet].mean()),
        startup_speed_rmse_mm_s=float(np.sqrt(np.mean((vm[startup] - target[startup, 1])**2))),
        radius_rmse_mm=float(np.sqrt(np.nanmean((rm - target[:, 2])**2))),
        overshoot_relative_last20s_mm=float(radius.max() - a.radius_mm[tail].mean() / SPACE_SCALE),
        sliding_fraction=float(a.sliding.mean()),run_end_s=float(a.time_s.iloc[-1]))
    settings = {key: value for key, value in globals().items() if key.isupper() and isinstance(value, (str, float, int, tuple, Path))}
    (OUTPUT_DIR / 'run_config.json').write_text(json.dumps(dict(settings=settings, build=config, metrics=metrics),
        indent=2, default=str), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'Figure: {OUTPUT_DIR / "comparison.png"}', flush=True)
    print(f'Scaled experiment: {OUTPUT_DIR / "comparison_scaled_experiment.png"}', flush=True)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(physical_fig)


def main(show=True):
    values = [DT, RUN_SECONDS, SAMPLE_CODE_TIME, SPACE_SCALE, INITIAL_RADIUS_MM, DISK_RPM,
              BALL_RADIUS_SCALE, RADIUS_AVERAGE_SECONDS]
    if not np.isfinite(values).all() or min(values) <= 0 or RUN_SECONDS < 72 - VIDEO_RELEASE_PTS_S:
        raise ValueError('Use finite positive settings and run through video PTS 72 s (at least 67.1 s after release)')
    if RADIUS_AVERAGE_SECONDS >= RUN_SECONDS:
        raise ValueError('Radius averaging window must be shorter than the simulation duration')
    raw = pd.read_csv(VIDEO_DIR / 'kinematics.csv')
    video_laps = pd.read_csv(VIDEO_DIR / 'orbit_periods.csv')
    config = read_build_config(kahan=False)
    module = load_hysteresis(kahan=False)
    somega = (DISK_RPM * 2 * np.pi / 60) / module.reference_disk_omega_rad_s
    params = module.SolverParams(sr=BALL_RADIUS_SCALE, somega=somega)
    units = module.physical_units(sr=BALL_RADIUS_SCALE, somega=somega)
    print('Using current FP64 build; material/field settings are recorded in run_config.json.', flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a = simulate(module, params, 1000 * units['length_unit_m'], units['disk_omega_rad_s'])
    a.to_csv(OUTPUT_DIR / 'trajectory.csv', index=False)
    plot_and_measure(a, experiment_bins(raw), video_laps, config, show)


if __name__ == '__main__':
    main(show='--no-show' not in sys.argv[1:])
