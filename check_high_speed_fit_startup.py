"""Check whether faster releases reach another stable branch; no forcing."""
from pathlib import Path
import json, subprocess, sys
root = Path(__file__).resolve().parent
out = root/'runs/laser-tilt-20261009/high_speed_fit_startup'
out.mkdir(parents=True, exist_ok=True)
records = []
for v0 in [-.125, -.2]:
    for mu, lam in [(.00005, .0054), (.001, .0012)]:
        tag = f'_highfit_start_v{abs(v0):g}_h{mu:g}_l{lam:g}'
        args = [sys.executable, str(root/'laser_tilt_experiment.py'), '--rpm', '200', '--rolling', f'--v0={v0}', '--force-mn', '0', '--push-seconds', '0', '--free-seconds', '360', '--angles', '0.002777749200346311', '--mu-low', '.00001', '--mu-high', str(mu), '--rolling-lambda', str(lam), '--tag', tag]
        with (out/f'{tag}.log').open('w', encoding='utf-8') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads((root/'runs/laser-tilt-20261009'/f'rpm200_diam12_dt0.0005{tag}'/'summary.json').read_text())[0]
        records.append(record)
        (out/'summary.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(json.dumps({'v0': v0, 'mu': mu, 'lambda': lam, 'status': record['status'], 'metrics': record.get('metrics')}), flush=True)
