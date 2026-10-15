"""Average radial-force balance on prescribed video-speed circular orbits.

Compile the current production header directly; use its verified production LUT.
Integrate JA memory on a prescribed pure-rolling orbit, without mechanical
trajectory integration, external pushing, or tuning any physical constants.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import animate_one_trajectory as animation

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/mean-centripetal-20261015'
SOURCE=ROOT/'kernels/tools/average_orbit_force.cpp'
BINARY=ROOT/'scratch/average_orbit_force.exe'
TARGETS=json.loads((ROOT.parent/'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())
COLUMNS=['radius_mm','period_s','rpm','spin_z_rad_s','mean_inward_magnetic_mN',
         'required_net_inward_mN','mean_rolling_generalized_inward_mN',
         'required_rolling_generalized_mN','mean_tangential_accel_m_s2',
         'mean_normal_load_mN','mean_total_magnetic_power_uW',
         'first_half_inward_mN','second_half_inward_mN']


def compile_tool():
    compiler=shutil.which('clang++')
    subprocess.run([compiler,'-std=c++17','-O3','-DUSE_FP64','-DFIELD_EVAL_MODE=1',
        '-DLUT_CACHE_MODE=0',f'-I{ROOT/"kernels/headers"}','-fuse-ld=lld',
        '-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',str(SOURCE),'-o',str(BINARY)],check=True)


def calculate(config,period,rpm,spin=0,dt=.0005,cycles=256,rmin=35,rmax=105,dr=.5):
    p=animation.scaled_model_parameters(config['magnet_number'],somega=rpm/100)
    result=subprocess.run([str(BINARY),config['lut'],str(dt),str(cycles),str(period),str(rpm),
        str(spin),str(rmin),str(rmax),str(dr),str(p[0]),str(p[2])],
        check=True,capture_output=True,text=True)
    return np.atleast_2d(np.loadtxt(io.StringIO(result.stdout),delimiter=',',skiprows=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=animation.read_build_config(kahan=False)
    if not args.skip_build:compile_tool()
    results=[]
    for target in TARGETS:
        period,rpm=target['ball_period_target_s'],target['disk_rpm']
        rows=calculate(config,period,rpm)
        np.savetxt(OUT/(target['video']+'.csv'),rows,delimiter=',',header=','.join(COLUMNS),comments='')
        fine=calculate(config,period,rpm,dt=.00025,cycles=512,rmin=70,rmax=85,dr=5)
        np.savetxt(OUT/(target['video']+'_fine.csv'),fine,delimiter=',',header=','.join(COLUMNS),comments='')
        samples=[dict(zip(COLUMNS,row)) for row in rows if row[0] in (35,40,50,60,65,70,75,80,85,90,95,100,105)]
        ratio=rows[:,6]/rows[:,7]
        ix=int(np.argmax(ratio))
        # Compare against the direct F=ma criterion, separately from rolling.
        roots=[]
        residual=rows[:,4]-rows[:,5]
        for i in range(len(rows)-1):
            if residual[i]*residual[i+1]<0:
                roots.append(float(np.interp(0,sorted([residual[i],residual[i+1]]),
                    [rows[i,0],rows[i+1,0]] if residual[i]<residual[i+1] else [rows[i+1,0],rows[i,0]])))
        spin_checks=[]
        for spin in (-2*np.pi,2*np.pi,86.84):
            checked=calculate(config,period,rpm,spin=spin,rmin=65,rmax=85,dr=5)
            spin_checks.extend(dict(zip(COLUMNS,row)) for row in checked)
        differences=[]
        for fine_row in fine:
            coarse=rows[np.argmin(abs(rows[:,0]-fine_row[0]))]
            differences.append(dict(radius_mm=float(fine_row[0]),
                inward_relative_difference=float(fine_row[4]/coarse[4]-1),
                generalized_relative_difference=float(fine_row[6]/coarse[6]-1)))
        record=dict(video=target['video'],period_s=period,rpm=rpm,samples=samples,
            radial_scan_mm=dict(minimum=35,maximum=105,step=.5),
            best_rolling_force_ratio=float(ratio[ix]),best_rolling_radius_mm=float(rows[ix,0]),
            direct_magnetic_F_equals_ma_radii_mm=roots,
            pure_rolling_radial_balance_exists=bool(ratio.max()>=1),
            spin_sensitivity=spin_checks,refinement=differences,
            periodic_mean_max_difference_mN=float(np.max(abs(rows[:,11]-rows[:,12]))))
        results.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('samples','spin_sensitivity')},indent=2),flush=True)
    # Solve the best-radius pure-rolling radial limit self-consistently in T.
    long=TARGETS[1]
    def deficit(period):
        a=calculate(config,period,long['disk_rpm'],rmin=78,rmax=86,dr=.5,cycles=128)
        return float(np.max(a[:,6]/a[:,7])-1)
    limit=brentq(deficit,2.5,3.2,xtol=2e-5)
    limit_rows=calculate(config,limit,long['disk_rpm'],rmin=78,rmax=86,dr=.25,cycles=512,dt=.00025)
    ix=np.argmax(limit_rows[:,6]/limit_rows[:,7])
    best_radius=float(limit_rows[ix,0])
    def fine_deficit(period):
        row=calculate(config,period,long['disk_rpm'],rmin=best_radius,rmax=best_radius,
                      dr=1,cycles=512,dt=.00025)[0]
        return float(row[6]/row[7]-1)
    limit=brentq(fine_deficit,limit-.005,limit+.005,xtol=1e-7)
    # At rest, this reduces exactly to the production held-ball memory solver.
    held=calculate(config,1e12,200,rmin=70,rmax=70,dr=1,cycles=256)[0,4]
    from audit_animation_period import Field
    field=Field(config)
    module=animation.load_hysteresis()
    params=module.SolverParams(*animation.scaled_model_parameters(8,somega=2))
    state=animation.make_initial_state(module,alpha=params.alpha)
    sums=[]
    field_period=np.pi/2
    for i in range(256*64):
        state=module.magnetize_held_ball(state,field_period/(64*50),50,params)
        if i>=128*64:sums.append(-field.evaluate(state)[1][0,0]*1000)
    record=dict(results=results,rolling_radial_limit_period_s=float(limit),
        rolling_radial_limit_radius_mm=float(limit_rows[ix,0]),
        limit_force_ratio_fine=float(fine_deficit(limit)+1),
        held_crosscheck=dict(native_mean_mN=float(held),production_held_mean_mN=float(np.mean(sums))),
        assumptions=['Horizontal disk and measured contact/rolling parameters from the current build.',
          'Prescribed reverse circular motion with pure rolling in the plane; spin_z=0 unless stated.',
          'Co-orbit frame q obeys q_dot=production_JA_rhs+(omega_ball-omega_orbit*z) cross q.',
          '256 source-field cycles; average final 128. Refinement doubles cycles and halves dt.',
          'Direct magnetic F=ma is only a necessary force-budget comparison if other horizontal forces are neglected.',
          'Rolling test includes the current magnetic torque and contact constraint via the actual production RHS.',
          'A radial root is not a full steady solution: tangential power, spin torque and stability must also balance.',
          'No force claim is made about arbitrary slipping or unmeasured spin states outside these tests.'],
        build=config,tool_sha256=animation.file_sha256(SOURCE),binary_sha256=animation.file_sha256(BINARY))
    (OUT/'summary.json').write_text(json.dumps(record,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for result in results:
        rows=np.loadtxt(OUT/(result['video']+'.csv'),delimiter=',',skiprows=1)
        axes[0].plot(rows[:,0],rows[:,4],label=f"Magnetic, T={result['period_s']:.4f} s")
        axes[0].plot(rows[:,0],rows[:,5],'--',label=f"Required net force, {result['video']}")
        axes[1].plot(rows[:,0],rows[:,6]/rows[:,7],label=f"T={result['period_s']:.4f} s")
    axes[1].axhline(1,color='black',ls='--',label='Radial rolling balance')
    axes[0].axhline(0,color='gray',lw=.8)
    for ax in axes:
        ax.axvspan(65,75,color='orange',alpha=.18,label='Video radius prior')
        ax.set(xlabel='Radius (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set(ylabel='Mean inward force (mN)',title='Current build: prescribed video-speed orbit')
    axes[1].set(ylabel='Available / required generalized inward force',title='Pure rolling, including magnetic torque')
    fig.savefig(OUT/'force_balance.png',dpi=180);plt.close(fig)
    print('LIMIT',limit,'radius',limit_rows[ix,0],'held check',record['held_crosscheck'],flush=True)


if __name__=='__main__':main()
