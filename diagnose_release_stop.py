"""Inspect release and arrest using the existing, manifest-checked CPU build."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from animate_one_trajectory import (read_build_config, load_hysteresis,
    make_initial_state, scaled_model_parameters)
from project_paths import RUNS_DIR, SCRATCH_DIR


def run(module, phi0, dt, seconds=20.0):
    sr = 5 / 6  # experimental radius 5 mm; reference radius 6 mm
    values = scaled_model_parameters(module.magnet_number, sr=sr, somega=1)
    params = module.SolverParams(*values)
    omega_disk = 100 * 2 * np.pi / 60
    k = (module.magnet_number / 2) / .07
    state = make_initial_state(module, alpha=values[0], phi0=phi0)
    stride = round(.02 / dt)
    count = round(seconds * omega_disk / (stride * dt))
    rows = [np.r_[0., state, 0.]]
    sliding, direction = None, None
    for i in range(count):
        state, sliding, direction = module.evolve_trajectory(
            state, dt, stride, params, sliding, direction, i * stride)
        rows.append(np.r_[(i + 1) * stride * dt / omega_disk, state, float(sliding)])
    data = np.array(rows)
    pos = data[:, 2:4]
    vel = data[:, 4:6]
    radius = np.linalg.norm(pos, axis=1)
    angle = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    angle -= angle[0]
    angular_speed = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / radius**2 * omega_disk
    speed = np.linalg.norm(vel, axis=1) * omega_disk / k
    tail = data[:, 0] > data[-1, 0] - 5
    reversals = np.flatnonzero((angular_speed[:-1] < 0) & (angular_speed[1:] >= 0)
                              & (angle[:-1] < -.5))
    departing = np.flatnonzero(angle < -.1)
    initial_end = int(departing[0]) if departing.size else len(angle)
    summary = dict(phi0_rad=phi0, dt=dt, seconds=float(data[-1, 0]),
        max_forward_rad=float(np.max(angle)), min_angle_rad=float(np.min(angle)),
        initial_forward_excursion_rad=float(np.max(angle[:initial_end])),
        initial_forward_peak_s=float(data[np.argmax(angle[:initial_end]), 0]),
        final_angle_rad=float(angle[-1]),
        angle_at_1s_rad=float(np.interp(1., data[:, 0], angle)),
        first_return_s=float(data[reversals[0] + 1, 0]) if reversals.size else None,
        first_return_angle_rad=float(angle[reversals[0] + 1]) if reversals.size else None,
        last_5s_angle_span_rad=float(np.ptp(angle[tail])),
        last_5s_mean_speed_mm_s=float(np.mean(speed[tail]) * 1000),
        last_5s_max_speed_mm_s=float(np.max(speed[tail]) * 1000),
        radius_range_mm=[float(np.min(radius) / k * 1000), float(np.max(radius) / k * 1000)],
        last_5s_sliding_fraction=float(np.mean(data[tail, -1])))
    return summary, data, angle, angular_speed, speed, radius / k * 1000


def main():
    config = read_build_config(kahan=False)
    module = load_hysteresis()
    output = RUNS_DIR / 'release-stop-diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    plots = SCRATCH_DIR / 'release-stop-diagnostic'
    plots.mkdir(parents=True, exist_ok=True)
    summaries = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for phi0, dt, seconds in [(0., .001, 20), (0., .0005, 20), (np.pi / 2, .001, 20),
                     (np.pi, .001, 20), (3 * np.pi / 2, .001, 20),
                     (3 * np.pi / 2, .001, 100), (3 * np.pi / 2, .0005, 100)]:
        result = run(module, phi0, dt, seconds)
        summary, data, angle, angular_speed, speed, radius = result
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        tag = f'phi{phi0:.4f}_dt{dt:g}_T{seconds}'
        np.savez_compressed(output / f'{tag}.npz', state_table=data,
            angle_rad=angle, angular_speed_rad_s=angular_speed,
            speed_m_s=speed, radius_mm=radius)
        if dt == .001 and seconds == 20:
            label = f'initial azimuth {phi0:.2f} rad'
            axes[0].plot(data[:, 0], angle, label=label)
            axes[1].plot(data[:, 0], angular_speed, label=label)
            axes[2].plot(data[:, 0], radius, label=label)
        if phi0 == 0 and dt == .001:
            close_fig, close_axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained')
            early = data[:, 0] <= 2
            close_axes[0].plot(data[early, 0], angle[early])
            close_axes[0].axhline(-1, color='gray', linestyle='--', alpha=.5)
            close_axes[0].axhline(-2, color='gray', linestyle='--', alpha=.5)
            close_axes[0].set(xlabel='Physical time (s)', ylabel='Displacement (rad)', title='Release: first 2 seconds')
            close_axes[1].plot(data[:, 0], angle)
            close_axes[1].set(xlabel='Physical time (s)', ylabel='Displacement (rad)', title='Same trajectory: full 20 seconds')
            for ax in close_axes:
                ax.grid(alpha=.25)
            close_fig.suptitle('5 mm ball; 100 rpm disk; initial rest, azimuth 0, q=0; existing FP64 build')
            close_fig.savefig(plots / 'one_release.png', dpi=150)
            plt.close(close_fig)
    for ax, label in zip(axes, ['Displacement (rad)', 'Angular speed (rad/s)', 'Orbit radius (mm)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel('Physical time (s)')
    fig.suptitle('Existing FP64 build; 5 mm ball, 100 rpm disk; tilt 0.02 rad')
    fig.savefig(plots / 'release_trajectories.png', dpi=160)
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(dict(build=config,
        parameters=scaled_model_parameters(module.magnet_number, sr=5/6),
        runs=summaries), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
