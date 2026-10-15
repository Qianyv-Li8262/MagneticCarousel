"""Paired smaller Z-spin test using the measured short-video initial frame."""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/video-frame-initial-20261013'
OUT = ROOT / 'runs/video-frame-small-spin-20261013/paired-pilot'
SPINS = [-40, -20, -10, -6, -3, -1, 0, 1, 3, 6, 10, 20, 40]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build = json.loads((SOURCE / 'build.json').read_text())
    with (SOURCE / 'jobs.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        bases = [row for row in reader if float(row['wz_rad_s']) == 0
                 and float(row['held_turns']) == 2
                 and int(row['phase_index']) < 2]
    jobs = []
    for base in bases:
        for spin in SPINS:
            row = dict(base)
            row['wz_rad_s'] = str(spin)
            row['label'] = row['label'].replace('__z0__', f'__z{spin}__')
            jobs.append(row)
    with (OUT / 'jobs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(jobs)
    config = dict(source_jobs=str(SOURCE / 'jobs.csv'), build=build,
                  spins_rad_s=SPINS, jobs_per_tau=len(jobs),
                  taus_s=[15, 20], duration_s=1200, dt_normalized=0.0005,
                  blowing=False, paired_phases=True)
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    for tau in config['taus_s']:
        dest = OUT / f'tau{tau}'
        dest.mkdir(exist_ok=True)
        if (dest / 'results.csv').exists():
            with (dest / 'results.csv').open() as stream:
                count = sum(1 for _ in stream) - 1
            if count == len(jobs):
                print(f'tau {tau} already complete', flush=True)
                continue
            raise RuntimeError(f'Incomplete existing output: {dest}')
        print(f'Starting tau={tau}, {len(jobs)} jobs', flush=True)
        with (dest / 'progress.log').open('w') as log:
            subprocess.run([build['executable'], '--jobs', str(OUT / 'jobs.csv'),
                            '--out', str(dest), '--tau', str(tau), '--workers', '24',
                            '--seconds', '1200'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f'Finished tau={tau}', flush=True)


if __name__ == '__main__':
    main()
