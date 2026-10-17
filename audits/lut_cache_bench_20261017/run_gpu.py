"""Time the existing production scan kernel, FP64, on a 64x64 initial grid."""
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=2000)
parser.add_argument('--skip-build',action='store_true')
args=parser.parse_args()
suffix=f'_steps{args.steps}' if args.steps!=2000 else ''

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
env=os.environ.copy()
preset=json.loads((ROOT/'CMakeUserPresets.json').read_text())['configurePresets'][0]
for key,value in preset['environment'].items():
    env[key]=value.replace('$penv{PATH}',env.get('PATH',''))
nvcc=r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe'
host=preset['cacheVariables']['CMAKE_CUDA_HOST_COMPILER']
lut=ROOT/'build/profile-avx/field-lut/field_lut_fp64.bin'
commands=[]
for mode in range(3):
    command=[nvcc,'-std=c++17','-O3','-arch=sm_120','-ccbin',host,'-DUSE_FP64','-DFIELD_EVAL_MODE=1',
        f'-DLUT_CACHE_MODE={mode}','-Xptxas=-v','-Xcompiler=/utf-8',f'-I{ROOT/"kernels/headers"}',f'-I{ROOT/"kernels"}',
        str(HERE/'bench_gpu.cu'),'-o',str(HERE/f'gpu_cache{mode}.exe')]
    commands.append(command)
    if args.skip_build:
        continue
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (HERE/f'gpu_build{mode}.log').write_text(result.stdout)
    if result.returncode:
        print(result.stdout,flush=True)
        raise SystemExit(result.returncode)
    print(f'built GPU mode {mode}',flush=True)
rows=[]
for repeat in range(3):
    for mode in [(repeat+j)%3 for j in range(3)]:
        output=subprocess.check_output([str(HERE/f'gpu_cache{mode}.exe'),str(lut),str(args.steps),str(HERE/f'gpu_output{mode}{suffix}.bin')],env=env,text=True)
        row=dict(mode=mode,repeat=repeat,**json.loads(output))
        rows.append(row)
        print(row,flush=True)
    (HERE/f'gpu_raw{suffix}.json').write_text(json.dumps(rows,indent=2))
summary=[]
double_count=64*64*(51+20)
baseline=(HERE/f'gpu_output0{suffix}.bin').read_bytes()
basevalues=struct.unpack(f'<{double_count}d',baseline[:double_count*8])
for mode in range(3):
    times=[ms for row in rows if row['mode']==mode for ms in row['ms']]
    data=(HERE/f'gpu_output{mode}{suffix}.bin').read_bytes()
    values=struct.unpack(f'<{double_count}d',data[:double_count*8])
    if not all(__import__('math').isfinite(x) for x in values): raise RuntimeError('nonfinite output')
    summary.append(dict(mode=mode,median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
        max_abs_output_diff=max(abs(a-b) for a,b in zip(values,basevalues)),
        max_scaled_output_diff=max(abs(a-b)/(1+abs(b)) for a,b in zip(values,basevalues)),
        status_equal=data[double_count*8:]==baseline[double_count*8:],
        failed_trajectories=max(row['failed_trajectories'] for row in rows if row['mode']==mode)))
(HERE/f'gpu_summary{suffix}.json').write_text(json.dumps(dict(commands=commands,grid=[64,64],steps=args.steps,dt=0.0005,
    precision='fp64',block=[8,16],axis_range=[0.25,1.0],summary=summary),indent=2))
print(json.dumps(summary,indent=2),flush=True)
