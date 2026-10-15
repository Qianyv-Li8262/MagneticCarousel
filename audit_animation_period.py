"""Audit the current animation itself against the locally measured video periods.

Runs animate_one_trajectory.animate and its actual frame callback with an Agg
canvas, preserving all solver settings. No alternative dynamics or old builds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import animate_one_trajectory as animation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/animation-field-period-20261015'
MASS = 7850 * 4 * np.pi / 3 * .006**3
OMEGA = 200 * 2 * np.pi / 60
LENGTH = .070 / 4
TARGETS = json.loads((ROOT.parent / 'cv/new_video_review/disk_rpm/ball_fit_targets.json').read_text())


def literal(value):
    return float(str(value).removeprefix('REAL_LITERAL(').removesuffix(')'))


class Field:
    """Reconstruct the verified production LUT and force formula in NumPy."""
    def __init__(self, config):
        self.config = config
        f = config['field_parameters']
        self.n = int(f['magnet_number'])
        self.nodes = int(f['kSplineNodes'])
        self.modes = int(f['kFieldFourierModeCount'])
        self.low = float(f['lut_lower_bound_ratio']) * self.n / 2
        self.dr = (float(f['lut_upper_bound_ratio']) * self.n / 2 - self.low) / (self.nodes - 1)
        self.lut = np.fromfile(config['lut'], dtype='<f8').reshape(self.modes, self.nodes, 4)
        p = config['model_parameters']
        self.beta = literal(p['Beta'])
        mur, alpha, c = (literal(p[k]) for k in ('Mu_r', 'Alpha_JA', 'C_JA'))
        self.D = (mur - 1) / (mur + 2)
        self.E = (1 - c) * (1 + alpha * (mur - 1)) / (mur + 2)

    def spline(self, left, right, u, offset):
        a, ma = left[..., offset], left[..., offset + 1]
        b, mb = right[..., offset], right[..., offset + 1]
        c1 = b - a - self.dr**2 * (2 * ma + mb) / 6
        c2, c3 = self.dr**2 * ma / 2, self.dr**2 * (mb - ma) / 6
        return (a + u * (c1 + u * (c2 + u * c3)),
                (c1 + u * (2 * c2 + 3 * c3 * u)) / self.dr,
                (2 * c2 + 6 * c3 * u) / self.dr**2)

    def evaluate(self, states):
        s = np.atleast_2d(states)
        r = np.linalg.norm(s[:, 1:3], axis=1)
        phi = np.arctan2(s[:, 2], s[:, 1])
        coordinate = (r - self.low) / self.dr
        idx = np.clip(np.floor(coordinate).astype(int), 0, self.nodes - 2)
        u = coordinate - idx
        av, ad, add = self.spline(self.lut[:, idx], self.lut[:, idx+1], u, 0)
        zv, zd, _ = self.spline(self.lut[:, idx], self.lut[:, idx+1], u, 2)
        m = ((2 * np.arange(self.modes) + 1) * self.n / 2)[:, None]
        theta = m * (phi - s[:, 0])
        co, si = np.cos(theta), np.sin(theta)
        hr, hp, hz = (-ad * co).sum(0), (m * av / r * si).sum(0), (-zv * co).sum(0)
        grr = (-add * co).sum(0)
        grp = (m * (ad / r - av / r**2) * si).sum(0)
        grz = (-zd * co).sum(0)
        gpp = ((m*m * av / r**2 - ad / r) * co).sum(0)
        gpz = (m * zv / r * si).sum(0)
        cp, sp = np.cos(phi), np.sin(phi)
        h = np.c_[cp*hr-sp*hp, sp*hr+cp*hp, hz] * self.beta
        gxx = cp*cp*grr-2*cp*sp*grp+sp*sp*gpp
        gxy = cp*sp*(grr-gpp)+(cp*cp-sp*sp)*grp
        gyy = sp*sp*grr+2*cp*sp*grp+cp*cp*gpp
        gxz, gyz = cp*grz-sp*gpz, sp*grz+cp*gpz
        grad = np.array([[gxx,gxy,gxz],[gxy,gyy,gyz],[gxz,gyz,-gxx-gyy]]).transpose(2,0,1)*self.beta
        # Quaternion rotates body q into the laboratory frame, exactly as in C++.
        qw, qv, q = s[:,5:6], s[:,6:9], s[:,12:15]
        cross = np.cross(qv, q)
        world_q = q + 2 * (qw * cross + np.cross(qv, cross))
        magnetization = 3 * (self.D * h - self.E * world_q)
        force = 4*np.pi*1e-7 * (MASS/7850) * 1.1e6**2 / LENGTH * np.einsum('nij,nj->ni',grad,magnetization)
        return h, force


def theory(config):
    field = Field(config)
    module = animation.load_hysteresis()
    params = module.SolverParams(*animation.scaled_model_parameters(8, somega=2))
    state = animation.prepare_initial_state(module, params, .0005, v0=-.05)
    # Validate the independent field reconstruction against production magnetic F.v.
    _, force = field.evaluate(state)
    native_power = module.magnetic_power(state, params)[0] * MASS * LENGTH**2 * OMEGA**3
    reconstructed_power = float(force[0,:2] @ (state[3:5]*LENGTH*OMEGA))
    if not np.isclose(native_power, reconstructed_power, rtol=2e-12, atol=1e-15):
        raise RuntimeError('LUT force reconstruction disagrees with production magnetic power')
    radial = np.linspace(60, 100, 161)
    phase = np.linspace(0, np.pi/2, 256, endpoint=False)
    rows=[]
    for radius in radial:
        states=np.zeros((len(phase),15));states[:,0]=phase;states[:,1]=radius*.001/LENGTH;states[:,5]=1
        h, force = field.evaluate(states)
        rows.append([radius, force[:,0].mean()*1000, -force[:,2].mean()*1000,
                     np.sqrt(np.mean(np.sum(h*h,axis=1))) * 1.1e6 * 4*np.pi*1e-7 * 1e4])
    rows=np.array(rows)
    targets=[]
    for t in TARGETS:
        period=t['ball_period_target_s'];r=.070;v=2*np.pi*r/period
        targets.append(dict(video=t['video'],rpm=t['disk_rpm'],period_s=period,
            period_2d=period*OMEGA,speed_m_s=v,centripetal_force_mN=MASS*v*v/r*1000,
            torque_free_pure_rolling_force_mN=1.4*MASS*v*v/r*1000,
            kinetic_energy_pure_rolling_uJ=.7*MASS*v*v*1e6))
    np.savetxt(OUT/'reversible_radial_curve.csv',rows,delimiter=',',comments='',
               header='radius_mm,outward_radial_force_mN,downward_force_mN,field_rms_G')
    record=dict(mass_kg=MASS,targets=targets,
        reconstruction_power_error_W=reconstructed_power-native_power,
        reversible_q0_phase_average=rows.tolist(),
        caveat='q=0 phase-averaged reversible reference only; neither a moving JA orbit nor a strict upper bound. Pure-rolling factor 1.4 assumes no magnetic torque along the required rolling acceleration.')
    (OUT/'theory.json').write_text(json.dumps(record,indent=2))
    print('THEORY',json.dumps(targets),flush=True)
    return field,record


def metrics(rows, seconds, blow_end):
    t=rows[:,0]/OMEGA;s=rows[:,1:16]
    r=np.linalg.norm(s[:,1:3],axis=1)*LENGTH
    angle=np.unwrap(np.arctan2(s[:,2],s[:,1]))
    tang=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/np.linalg.norm(s[:,1:3],axis=1)*LENGTH*OMEGA
    lo=max(blow_end/OMEGA+10,t[-1]-min(60,seconds/2))
    mask=t>=lo
    if mask.sum()<10:return dict(completed_s=float(t[-1]),status='insufficient force-free tail')
    wins=[]
    for indices in np.array_split(np.flatnonzero(mask),3):
        rate=np.polyfit(t[indices]-t[indices].mean(),angle[indices],1)[0]
        wins.append(dict(rate_rad_s=float(rate),period_s=float(-2*np.pi/rate) if rate<-.01 else None,
                         radius_mm=float(r[indices].mean()*1000)))
    rate=np.polyfit(t[mask]-t[mask].mean(),angle[mask],1)[0]
    rates=np.array([w['rate_rad_s'] for w in wins])
    stable=np.ptp(rates)/max(abs(rates.mean()),1e-8)<.01 and np.ptp([w['radius_mm'] for w in wins])<.5
    reverse=float(np.mean(tang[mask]<0))
    return dict(completed_s=float(t[-1]),period_s=float(-2*np.pi/rate) if rate<-.01 else None,
        radius_mean_mm=float(r[mask].mean()*1000),radius_range_mm=[float(r[mask].min()*1000),float(r[mask].max()*1000)],
        speed_rms_m_s=float(np.sqrt(np.mean(tang[mask]**2))),reverse_fraction=reverse,
        spin_z_rms_rad_s=float(np.sqrt(np.mean(s[mask,11]**2))*OMEGA),
        rolling_fraction=float(np.mean(rows[mask,16]==0)),windows=wins,
        stationary_reverse_orbit=bool(stable and reverse>.99),
        tail_start_s=float(lo))


def run_case(case, seconds, dt, field, config):
    key=case['name']+('_fine' if dt<.0005 else '')
    path=OUT/(key+'.json')
    inputs=dict(case=case,seconds=seconds,dt=dt,binary_sha256=config['binary_sha256'],
                lut_sha256=config['lut_sha256'],animation_sha256=animation.file_sha256(Path(animation.__file__)))
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['inputs']==inputs:
            print('CACHED',key,json.dumps(cached['metrics']),flush=True);return cached
    rows=[];began=time.perf_counter();error=None
    spf=round(.35/dt)
    options={k:v for k,v in case.items() if k!='name'}
    fig,ani=animation.animate(dt=dt,steps_per_frame=spf,fast=True,expected_config=config,
        sample_callback=lambda t,s,m:rows.append(np.r_[t,s,float(bool(m))]),**options)
    ani._init_draw()
    try:
        for frame in range(int(np.ceil(seconds*OMEGA/(5*spf*dt)))):
            ani._func(frame)
    except RuntimeError as exc:
        error=str(exc)
    ani._draw_was_started=True
    plt.close(fig)
    rows=np.array(rows)
    np.savez_compressed(OUT/(key+'.npz'),rows=rows)
    end=case.get('blow_start',0)+case.get('blow_duration',0)
    met=metrics(rows,seconds,end)
    sample=rows[rows[:,0]/OMEGA>=met.get('tail_start_s',0)]
    if len(sample):
        state=sample[:,1:16];_,force=field.evaluate(state)
        er=state[:,1:3]/np.linalg.norm(state[:,1:3],axis=1)[:,None]
        inward=-np.sum(force[:,:2]*er,axis=1)
        mag_acc=force/MASS
        speed=state[:,3:5]*LENGTH*OMEGA
        module=animation.load_hysteresis()
        params=module.SolverParams(*animation.scaled_model_parameters(8,somega=2))
        # Magnetic power includes magnetic spin torque, from the production API.
        power=np.array([module.magnetic_power(s,params) for s in state[::5]]) * MASS*LENGTH**2*OMEGA**3
        met.update(mean_inward_magnetic_force_mN=float(inward.mean()*1000),
                   mean_normal_load_mN=float((MASS*9.81-force[:,2]).mean()*1000),
                   mean_magnetic_total_power_uW=float(power[:,2].mean()*1e6),
                   rms_magnetic_force_mN=float(np.sqrt(np.mean(np.sum(force*force,axis=1)))*1000))
    record=dict(inputs=inputs,metrics=met,error=error,wall_s=time.perf_counter()-began)
    path.write_text(json.dumps(record,indent=2))
    print('RESULT',key,json.dumps(dict(**met,error=error,wall_s=record['wall_s'])),flush=True)
    return record


CASES=[dict(name='default',v0=-.05),dict(name='target_start',v0=-.12),
       dict(name='fast_start',v0=-.16),dict(name='gentle_blow',v0=-.05,blow_start=30,blow_duration=3),
       dict(name='long_blow',v0=-.05,blow_start=30,blow_duration=20),
       dict(name='strong_blow',v0=-.05,blow_start=30,blow_duration=20,blow_force=3),
       dict(name='target_nonrolling',v0=-.12,s0=-.12),
       dict(name='zero_start_long_blow',v0=0,blow_start=30,blow_duration=20)]


def plot(records, theory_record):
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    for record in records:
        case=record['inputs']['case'];key=case['name']+('_fine' if record['inputs']['dt']<.0005 else '')
        rows=np.load(OUT/(key+'.npz'))['rows'];t=rows[:,0]/OMEGA;s=rows[:,1:16]
        phi=np.unwrap(np.arctan2(s[:,2],s[:,1]));centres=[];periods=[]
        for lo in np.arange(0,t[-1]-5,5):
            use=(t>=lo)&(t<lo+5);rate=np.polyfit(t[use]-t[use].mean(),phi[use],1)[0]
            centres.append(lo+2.5);periods.append(-2*np.pi/rate if rate<-.01 else np.nan)
        axes[0].plot(centres,periods,label=key)
        axes[1].plot(t[::5],np.linalg.norm(s[::5,1:3],axis=1)*LENGTH*1000,label=key)
    periods=[q['period_s'] for q in theory_record['targets']]
    axes[0].axhspan(min(periods),max(periods),color='orange',alpha=.3,label='Video mean periods')
    axes[0].set(ylabel='Reverse orbital period (s)',ylim=(0,12),title='Actual animate_one_trajectory frame callback; force-free tail assessed')
    axes[1].axhspan(65,75,color='orange',alpha=.2)
    axes[1].set(ylabel='Orbit radius (mm)',xlabel='Seconds after release')
    curve=np.array(theory_record['reversible_q0_phase_average'])
    axes[2].plot(curve[:,0],-curve[:,1],label='Reversible q=0 phase-mean inward magnetic force')
    for target in theory_record['targets']:
        r=curve[:,0]*.001
        axes[2].plot(curve[:,0],MASS*r*(2*np.pi/target['period_s'])**2*1000,'--',label=f"m r w²: {target['video']}")
    axes[2].set(xlabel='Radius (mm)',ylabel='Inward force (mN)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--cases',nargs='*')
    parser.add_argument('--dt',type=float,default=.0005)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=animation.read_build_config(kahan=False)
    field,record=theory(config)
    records=[]
    for case in CASES:
        if args.cases and case['name'] not in args.cases:continue
        records.append(run_case(case,args.seconds,args.dt,field,config))
    (OUT/'latest_summary.json').write_text(json.dumps(dict(theory=record,runs=records,build=config),indent=2))
    plot(records,record)


if __name__=='__main__':main()
