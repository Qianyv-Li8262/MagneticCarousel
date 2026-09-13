"""Diagnose the current model without changing production files.

Examples (in the CuPy environment):
    python diagnose_hysteresis_gpu.py --field-scale 0.1 --time 4000
    python diagnose_hysteresis_gpu.py --fp64 --k 16 --time 100

FP64 is an isolated reference calculation, not a production CUDA ABI change.
Results go to tmp/diagnostics. Each row records status, completed time, velocity,
angular velocity, |q|, max |q|, last-state N, max |phase_dot|, qx, qy, phase, x,
minimum and maximum rolling-to-sliding event fractions. Status 1 means N < 0
(possibly at an RK stage), 2 means undefined sliding direction, 3 nonfinite.
Success only means completion; it does not certify numerical convergence.
"""
from pathlib import Path
import argparse
import json
import time

import cupy as cp
import numpy as np

from project_paths import RUNS_DIR

ROOT = Path(__file__).resolve().parent
INSTRUMENTATION = r'''
__device__ double low[16384], high[16384];
__device__ void debug_transit(double v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    low[i]=fmin(low[i],v); high[i]=fmax(high[i],v);
}
'''
KERNEL = r'''
extern "C" __global__ void diagnose(float dt, int steps, int k, double *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=k*k) return;
    float sr=0.1f+1.9f*(i%k)/(k-1), sw=0.1f+1.9f*(i/k)/(k-1);
    solver_params ref={0.33333333f,1.7212357e-4f,0.16560356f,489.39639f,11011.419f};
    solver_params p=scaled_params(sr,sw,ref);
    low[i]=1; high[i]=0;
    state s={}; bool sliding=false; float sign=0;
    int initial=initialize_gpu_mode(s,p,sliding,sign), flag=initial==0 ? 0 : 4, n=0;
    real_t ss=sign, maxq=0, maxphase=0;
    for(;n<steps && flag==0;++n){
        flag=fwd1step(s,dt,p,sliding,ss);
        if(!finite_state(s)) flag=3;
        maxq=fmax(maxq,length(s.q));
        maxphase=fmax(maxphase,fabs(1-s.xdot-s.theta_dot));
    }
    vec2 h={H_EXT*real_cos(s.phase),H_EXT*real_sin(s.phase)};
    double N=p.gamma_g-p.gamma/(2+MU_R)*(-(MU_R-1)*H_EXT*H_EXT+(1-C)/(1-XI_JA)*(s.q*h));
    double *o=out+i*14;
    o[0]=flag; o[1]=(n-(flag!=0 && n>0))*(double)dt;
    o[2]=s.xdot; o[3]=s.theta_dot; o[4]=length(s.q); o[5]=maxq;
    o[6]=N; o[7]=maxphase; o[8]=s.q.x; o[9]=s.q.y;
    o[10]=s.phase; o[11]=s.x; o[12]=low[i]; o[13]=high[i];
}
'''


def source_for(field_scale, fp64, use_kahan):
    header=(ROOT/'1dmodel-hysteresis.hpp').read_text(encoding='utf-8')
    kahan_include='#include "kahan.hpp"'
    if kahan_include not in header:
        raise RuntimeError('Kahan header include changed; update diagnostic source assembly')
    header=header.replace(kahan_include,(ROOT/'kahan.hpp').read_text(encoding='utf-8'),1)
    field_line='#define H_EXT REAL_LITERAL(1.4661546e-1)'
    if field_line not in header:
        raise RuntimeError('H_EXT definition changed; update diagnostic override')
    header=header.replace(field_line, f'#define H_EXT REAL_LITERAL({.14661546*field_scale:.16e})')
    if fp64:
        header=header.replace('using real_t = float;', 'using real_t = double;')
        header=header.replace('#define REAL_LITERAL(value) value##f','#define REAL_LITERAL(value) value')
        for name in ('cos','fabs','remainder','sin','sqrt'):
            header=header.replace(f'#define REAL_{name.upper()}_IMPL {name}f',f'#define REAL_{name.upper()}_IMPL {name}')
    event_line='real_t dt_transit = dt * g_init / (g_init - g_final);'
    if event_line not in header:
        raise RuntimeError('Event interpolation changed; update diagnostic instrumentation')
    header=header.replace(event_line,event_line+' debug_transit(dt_transit/dt);')
    source=(ROOT/'1dmodel-hysteresis.cu').read_text(encoding='utf-8')
    source=source.replace('#include "1dmodel-hysteresis.hpp"',INSTRUMENTATION+header)
    # Retain shared initialization/scaling only, avoiding the production FP32 ABI.
    marker='DEVICE INLINE void update_velocity_minimum'
    if marker not in source:
        raise RuntimeError('CUDA helper layout changed; update diagnostic extraction')
    return source.split(marker)[0]+'\n}\n'+KERNEL


def run(field_scale, fp64, use_kahan, dt, total_time, k):
    options=('--std=c++17','-DUSE_KAHAN') if use_kahan else ('--std=c++17',)
    kernel=cp.RawModule(code=source_for(field_scale,fp64,use_kahan),options=options).get_function('diagnose')
    out=cp.zeros((k*k,14),dtype=cp.float64)
    start=time.perf_counter()
    kernel(((k*k+127)//128,),(128,),
           (np.float32(dt),np.int32(round(total_time/dt)),np.int32(k),out))
    data=out.get()
    flags,counts=np.unique(data[:,0],return_counts=True)
    summary=dict(field_scale=field_scale,fp64=fp64,kahan=use_kahan,dt=dt,total_time=total_time,k=k,
                 seconds=time.perf_counter()-start,
                 status_counts=dict(zip(map(int,flags),map(int,counts))),
                 time_quantiles=np.quantile(data[:,1],[0,.25,.5,.75,1]).tolist(),
                 max_q=float(data[:,5].max()),max_phase_rate=float(data[:,7].max()),
                 invalid_event_points=int(((data[:,12]<0)|(data[:,13]>1)).sum()))
    directory=RUNS_DIR/'diagnostics'
    directory.mkdir(parents=True,exist_ok=True)
    kahan_suffix='_kahan' if use_kahan else ''
    output=directory/f'field{field_scale}_fp{64 if fp64 else 32}{kahan_suffix}_dt{dt}_T{total_time}_k{k}.npz'
    np.savez(output,data=data,summary=json.dumps(summary))
    print(json.dumps(summary,indent=2))
    print(output)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field-scale',type=float,default=1.)
    parser.add_argument('--fp64',action='store_true')
    parser.add_argument('--kahan',action='store_true',
                        help='use compensated accumulators for persistent CUDA state')
    parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--time',type=float,default=100.)
    parser.add_argument('--k',type=int,default=16)
    args=parser.parse_args()
    if not (2<=args.k<=128):
        parser.error('k must be between 2 and 128')
    if not (np.isfinite(args.dt) and np.float32(args.dt)>0 and np.isfinite(args.time)
            and args.time>=0 and args.time/args.dt<2147483000
            and np.isfinite(args.field_scale) and 0<args.field_scale<=100):
        parser.error('invalid field scale/dt/time or step count overflow')
    run(args.field_scale,args.fp64,args.kahan,args.dt,args.time,args.k)
