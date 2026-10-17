"""Compare time-weighted radius averages without running the simulation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from compare_video56c6 import VIDEO_DIR, VIDEO_RELEASE_PTS_S, SPACE_SCALE

WINDOWS_SECONDS = (1., 3., 5., 10.)
OUTPUT_DIR = Path(__file__).resolve().parent / 'scratch' / 'video56c6-radius-averages'


def centered_time_mean(t, radius, centers, window):
    # Integrate the piecewise-linear signal against actual video timestamps.
    # Every displayed point has a FULL window; no padded or shrinking end windows.
    integral = np.r_[0., np.cumsum(np.diff(t) * (radius[1:] + radius[:-1]) / 2)]
    def primitive(x):
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        delta = x - t[i]
        slope = (radius[i + 1] - radius[i]) / (t[i + 1] - t[i])
        return integral[i] + radius[i] * delta + .5 * slope * delta**2
    valid = (centers - window / 2 >= t[0]) & (centers + window / 2 <= t[-1])
    result = np.full(len(centers), np.nan)
    x = centers[valid]
    result[valid] = (primitive(x + window / 2) - primitive(x - window / 2)) / window
    return result


def main(show=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw = pd.read_csv(VIDEO_DIR / 'kinematics.csv')
    good = (raw.position_valid == 1) & np.isfinite(raw.r_mm) & (raw.source_pts_s >= VIDEO_RELEASE_PTS_S)
    a = raw.loc[good].sort_values('source_pts_s').groupby('source_pts_s', as_index=False).r_mm.mean()
    t = a.source_pts_s.to_numpy(); radius = a.r_mm.to_numpy() * SPACE_SCALE
    if np.diff(t).max() > .25:
        raise ValueError('A data gap exceeds .25 s; split the signal rather than bridge that gap')
    centers = np.arange(t[0], t[-1], 1 / 30)
    result = pd.DataFrame({'source_pts_s': centers})
    for window in WINDOWS_SECONDS:
        result[f'mean_{window:g}s_mm'] = centered_time_mean(t, radius, centers, window)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / 'radius_averages.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    fig.suptitle(f'56c6 experimental radius: longer time averages (spatial scale {SPACE_SCALE:.6f})')
    axes[0].plot(t, radius, color='#94a3b8', alpha=.22, lw=.6, label='Valid frame radii')
    colors = ['#3b82f6', '#e8790c', '#059669', '#be185d']
    for window, color in zip(WINDOWS_SECONDS, colors):
        values = result[f'mean_{window:g}s_mm']
        axes[0].plot(centers, values, color=color, lw=1.6 if window < 5 else 2.2,
                     alpha=.65 if window == 1 else 1, label=f'{window:g} s centered mean')
        if window >= 3:
            axes[1].plot(centers, values, color=color, lw=2, label=f'{window:g} s centered mean')
    blocks = []
    for lo in np.arange(30., 70., 5.):
        mid = lo + 2.5
        value = float(centered_time_mean(t, radius, np.array([mid]), 5.)[0])
        blocks.append(dict(start_pts_s=float(lo), end_pts_s=float(lo + 5), mean_radius_mm=value))
    axes[1].plot([z['start_pts_s'] + 2.5 for z in blocks], [z['mean_radius_mm'] for z in blocks],
                 'kD', ms=4, label='Non-overlapping 5 s blocks')
    axes[0].set(xlim=(4.9, 73.5), ylabel='Calibrated radius (mm)', title='Full record: large windows broaden the startup transition')
    axes[1].set(xlim=(30, 72), ylim=(74.5, 78.5), ylabel='Calibrated radius (mm)', title='Late-time mean radius; complete windows only')
    for axis in axes:
        axis.set_xlabel('Video PTS (s)'); axis.grid(alpha=.25); axis.legend(fontsize=9)
    fig.savefig(OUTPUT_DIR / 'radius_averages.png', dpi=160)
    metadata = dict(spatial_scale=SPACE_SCALE, windows_seconds=WINDOWS_SECONDS,
        method='Centered full-window, time-weighted mean of linearly interpolated valid positions; no edge padding. No simulation or detrending.',
        valid_time_range=[float(t[0]), float(t[-1])], max_sample_gap_s=float(np.diff(t).max()), blocks=blocks)
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main(show='--no-show' not in sys.argv[1:])
