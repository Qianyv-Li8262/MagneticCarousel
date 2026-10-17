"""Build isolated cache variants; interleave timings and check equal outputs."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--extra',action='store_true')
args=parser.parse_args()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
env = os.environ.copy()
preset = json.loads((ROOT/'CMakeUserPresets.json').read_text())['configurePresets'][0]
for key, value in preset['environment'].items():
    env[key] = value.replace('$penv{PATH}', env.get('PATH',''))
compiler = preset['cacheVariables']['CMAKE_CXX_COMPILER']
lut = ROOT/'build/profile-avx/field-lut/field_lut_fp64.bin'
commands=[]
for mode in range(3):
    command=[compiler,'-std=c++17','-O3','-DNDEBUG','-mavx2','-DUSE_FP64','-DFIELD_EVAL_MODE=1',
             f'-DLUT_CACHE_MODE={mode}',f'-I{ROOT / "kernels/headers"}',str(HERE/'bench_cpu.cpp'),
             '-o',str(HERE/f'cpu_cache{mode}.exe')]
    commands.append(command)
    subprocess.run(command,env=env,check=True)
    print(f'built mode {mode}',flush=True)
rows=[]
cases={'same_cell':2_000_000,'path':2_000_000,'random_cells':2_000_000,'trajectory':1_000_000}
if args.extra:
    cases={'trajectory_sliding':1_000_000,'trajectory_rolling':1_000_000}
suffix='_extra' if args.extra else ''
# Rotate execution order to distribute drift; no concurrent timed runs.
for repeat in range(9):
    for case,count in cases.items():
        for mode in [(repeat+j)%3 for j in range(3)]:
            output=subprocess.check_output([str(HERE/f'cpu_cache{mode}.exe'),str(lut),case,str(count),'1729'],env=env,text=True)
            row=dict(case=case,mode=mode,repeat=repeat,count=count,**json.loads(output))
            rows.append(row)
        print(f'round {repeat+1}/9 {case}',flush=True)
    (HERE/f'cpu_raw{suffix}.json').write_text(json.dumps(rows,indent=2))
summary=[]
for case in cases:
    baseline=next(r['values'] for r in rows if r['case']==case and r['mode']==0)
    for mode in range(3):
        selected=[r for r in rows if r['case']==case and r['mode']==mode]
        times=[r['ns'] for r in selected]
        error=max(abs(a-b) for r in selected for a,b in zip(r['values'],baseline))
        summary.append(dict(case=case,mode=mode,median_ns=statistics.median(times),min_ns=min(times),
                            max_ns=max(times),max_abs_output_diff=error,hit_rate=selected[0].get('hit_rate')))
report=dict(compiler=compiler,commands=commands,lut=str(lut),affinity_logical_cpu=2,
            rounds=9,summary=summary)
(HERE/f'cpu_summary{suffix}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2),flush=True)
