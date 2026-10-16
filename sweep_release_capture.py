"""Compare capture with the original magnetism using isolated CPU builds.

Worker mode runs a manifest-checked snapshot; driver mode varies tilt/contact
friction in that snapshot without changing the production model.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'scratch' / 'legacy-release-source'
OUTPUT = ROOT / 'runs' / 'legacy-release-capture'


def worker(source, output, label, seconds, dt, phi_values, speed_values):
    sys.path.insert(0, str(source))
    from animate_one_trajectory import read_build_config, load_hysteresis, make_initial_state
    from scipy.ndimage import uniform_filter1d
    config = read_build_config(kahan=False)
    module = load_hysteresis()
    target = output / label
    target.mkdir(parents=True, exist_ok=True)
    summaries = []
    for phi0 in phi_values:
        for somega in speed_values:
            params = module.SolverParams(sr=5/6, somega=somega)
            disk_omega = 100 * 2 * np.pi / 60 * somega
            sample_steps = max(1, round(.002 * disk_omega / dt))
            sample_dt = sample_steps * dt / disk_omega
            chunks = round(seconds / sample_dt)
            state = make_initial_state(module, alpha=params.alpha, phi0=phi0)
            data = [np.r_[0., state, 0.]]
            sliding, direction = None, None
            failure = None
            for i in range(chunks):
                try:
                    state, sliding, direction = module.evolve_trajectory(state, dt,
                        sample_steps, params, sliding, direction, i * sample_steps)
                except RuntimeError as exc:
                    failure = dict(label=label, phi0_rad=phi0, rpm=100*somega,
                        dt=dt, status='solver_failed', last_sample_s=i*sample_dt, error=str(exc))
                    break
                data.append(np.r_[(i + 1) * sample_dt, state, float(sliding)])
            if failure is not None:
                summaries.append(failure)
                print(json.dumps(failure), flush=True)
                continue
            data = np.array(data)
            pos, vel = data[:, 2:4], data[:, 4:6]
            angle = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
            angle -= angle[0]
            # B^2 and instantaneous attraction repeat eight times per disk turn.
            # Smooth over TWO such periods; retain raw trace for all stop claims.
            window = max(1, round((.15 / somega) / sample_dt))
            smooth = uniform_filter1d(angle, window, mode='nearest')
            tail = (data[:, 0] > seconds - 5) & (data[:, 0] < seconds - .15/somega)
            omega = (pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0]) / np.sum(pos**2, axis=1) * disk_omega
            speed = np.linalg.norm(vel, axis=1) * disk_omega / (4/.07) * 1000
            downhill_phi = config['inclined_disk']['axis_azimuth_rad'] - np.pi/2
            equilibrium_phi = (phi0 + float(np.mean(smooth[tail]))) % (2*np.pi)
            low_offset = (equilibrium_phi-downhill_phi+np.pi) % (2*np.pi)-np.pi
            # First slow reversal after significant negative motion.
            smooth_speed = np.gradient(smooth, sample_dt)
            candidates = np.flatnonzero((smooth_speed[:-1]<0) & (smooth_speed[1:]>=0) & (smooth[:-1]<-.5))
            turn = int(candidates[0]+1) if len(candidates) else None
            record = dict(label=label, phi0_rad=phi0, rpm=100*somega, somega=somega,
                dt=dt, duration_s=float(data[-1, 0]),
                tilt_rad=config['inclined_disk']['angle_rad'],
                mu=float(config['model_parameters']['Mu'].removeprefix('REAL_LITERAL(').removesuffix(')')),
                max_forward_rad=float(np.max(angle)), min_displacement_rad=float(np.min(angle)),
                final_mean_displacement_rad=float(np.mean(smooth[tail])),
                final_mean_azimuth_rad=equilibrium_phi, offset_from_downhill_rad=low_offset,
                tail_raw_angle_span_rad=float(np.ptp(angle[tail])),
                tail_smooth_angle_span_rad=float(np.ptp(smooth[tail])),
                tail_mean_speed_mm_s=float(np.mean(speed[tail])),
                tail_sliding_fraction=float(np.mean(data[tail, -1])),
                first_slow_turn_s=float(data[turn,0]) if turn is not None else None,
                first_slow_turn_displacement_rad=float(smooth[turn]) if turn is not None else None)
            tag = f'phi{phi0:g}_rpm{100*somega:g}_dt{dt:g}'
            np.savez_compressed(target/f'{tag}.npz', state_table=data, displacement_rad=angle,
                smoothed_displacement_rad=smooth, angular_speed_rad_s=omega, speed_mm_s=speed)
            record['trace'] = str(target/f'{tag}.npz')
            summaries.append(record)
            print(json.dumps(record), flush=True)
    (target/'summary.json').write_text(json.dumps(dict(build=config,
        smoothing='two attraction periods (0.15/somega seconds), excludes end-filter edge',
        runs=summaries), indent=2), encoding='utf-8')


def driver(extra=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    header = SNAPSHOT/'kernels'/'headers'/'hysteresis_config.hpp'
    base = header.read_text(encoding='utf-8')
    variants = [('legacy_base', .02, .02, .38, .0010909091), ('legacy_mu05', .02, .05, .38, .0010909091),
                ('legacy_mu10', .02, .10, .38, .0010909091), ('legacy_tilt01_mu05', .01, .05, .38, .0010909091),
                ('legacy_tilt04_mu05', .04, .05, .38, .0010909091)]
    if extra:
        variants = [('legacy_mu01', .02, .01, .38, .0010909091),
                    ('legacy_tilt03_mu01', .03, .01, .38, .0010909091),
                    ('legacy_halfK', .02, .02, .38, .00054545455),
                    ('legacy_C06', .02, .02, .6, .0010909091)]
    for label, tilt, mu, cja, khys in variants:
        text = base
        for key, value in [('kTiltAngleRad', tilt), ('Mu', mu), ('C_JA', cja), ('K', khys)]:
            text, count = re.subn(rf'(CONFIG_CONST real_t {key}\s*=\s*)REAL_LITERAL\([^)]*\)',
                rf'\g<1>REAL_LITERAL({value})', text)
            assert count == 1
        if text != header.read_text(encoding='utf-8'):
            header.write_text(text, encoding='utf-8')
        build = subprocess.run(['cmake', '--build', '--preset', 'local-cpu-release',
                                '--target', 'hysteresis_fp64'], cwd=SNAPSHOT,
                               text=True, capture_output=True)
        (OUTPUT/f'{label}_build.log').write_text(build.stdout+build.stderr, encoding='utf-8')
        if build.returncode:
            raise RuntimeError(build.stdout+build.stderr)
        variant_output = OUTPUT/label
        variant_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(header, variant_output/'hysteresis_config.hpp')
        # Preserve executable binaries for replay, with original LUT path in manifest.
        shutil.copytree(SNAPSHOT/'build'/'release'/'extensions'/'fp64',
                        variant_output/'extension', dirs_exist_ok=True)
        print(f'Built {label}: tilt={tilt}, mu={mu}', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--source', str(SNAPSHOT), '--output', str(OUTPUT), '--label', label], check=True)
    # Leave the isolated snapshot on the original tilt/friction values.
    header.write_text(base, encoding='utf-8')
    subprocess.run(['cmake', '--build', '--preset', 'local-cpu-release', '--target', 'hysteresis_fp64'],
                   cwd=SNAPSHOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--extra', action='store_true')
    parser.add_argument('--source', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--label', default='legacy_base')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--phi', type=float, nargs='+', default=[0., np.pi/2, np.pi, 3*np.pi/2])
    parser.add_argument('--somega', type=float, nargs='+', default=[.5, 1., 1.5])
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.output, args.label, args.seconds, args.dt, args.phi, args.somega)
    else:
        driver(args.extra)
