"""Independent matched-period experiment; never changes production settings."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time
import numpy as np
import animate_one_trajectory as animation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/mu-moment-audit-20261010'
OUT.mkdir(parents=True, exist_ok=True)
BINARY = next((ROOT / 'scratch/laser-tilt-20261009-source/build').glob('hysteresis*.pyd'))
spec = importlib.util.spec_from_file_location('hysteresis', BINARY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
OMEGA = 200 * 2 * np.pi / 60
K = 4 / .07
TILT = .5 * np.arctan(4 / 930)
MREF = 4.222662956003016

def metrics(rows, start, end):
    used = rows[(rows[:, 0] >= start) & (rows[:, 0] <= end)]
    t = used[:, 0]; s = used[:, 1:16]
    phi = np.unwrap(np.arctan2(s[:, 2], s[:, 1]))
    radius = np.hypot(s[:, 1], s[:, 2])
    tang = -(s[:, 1]*s[:, 4]-s[:, 2]*s[:, 3])/radius*OMEGA/K*1000
    co = np.linalg.lstsq(np.c_[np.ones(len(t)), np.cos(phi), np.sin(phi), np.cos(2*phi), np.sin(2*phi)], tang, rcond=None)[0]
    rate = np.linalg.lstsq(np.c_[np.ones(len(t)), t-t.mean()], phi, rcond=None)[0][1]
    levels = -2*np.pi*np.arange(1, int(max(0, -(phi[-1]-phi[0]))/(2*np.pi))+1)
    angle = phi-phi[0]
    crossings=[]
    for level in levels:
        ids=np.flatnonzero(angle<=level)
        if len(ids) and ids[0]>0:
            i=ids[0];crossings.append(float(np.interp(level,angle[i-1:i+1][::-1],t[i-1:i+1][::-1])))
    return {'period_s':float(-2*np.pi/rate) if rate<0 else None,
            'crossing_period_s':float(np.mean(np.diff(crossings))) if len(crossings)>1 else None,
            'amplitude_percent':float(np.hypot(co[1],co[2])/co[0]*100),
            'amplitude2_percent':float(np.hypot(co[3],co[4])/co[0]*100),
            'phase_rad':float(np.arctan2(co[2],co[1])),
            'radius_mm':float(radius.mean()/K*1000),
            'radius_min_mm':float(radius.min()/K*1000),'radius_max_mm':float(radius.max()/K*1000),
            'mean_tangent_mm_s':float(tang.mean()),
            'reverse_fraction':float(np.mean(tang>0)),
            'rolling_fraction':float(np.mean(used[:,16]==0)),
            'window_s':[float(t[0]),float(t[-1])]}

def run(mu, scale, seconds=240, dt=.0005, lam=.004, save=False, tilt=TILT, rpm=200, sample_hz=None):
    global OMEGA
    OMEGA=rpm*2*np.pi/60
    name=f'h{mu:.9g}_m{scale:.9g}_l{lam:.9g}_s{seconds:g}_dt{dt:g}'
    if tilt != TILT:
        name+=f'_tilt{tilt:.12g}'
    if rpm != 200:name+=f'_rpm{rpm:.12g}'
    if sample_hz is not None:name+=f'_fps{sample_hz:.12g}'
    path=OUT/(name+'.json')
    if path.exists():return json.loads(path.read_text())
    began=time.perf_counter()
    mod.configure_magnet_moment_scale(scale)
    mod.configure_rolling_resistance(1e-5,mu,lam)
    mod.configure_laser_tilt(tilt,np.pi/2)
    # Recompute from SI inputs rather than reusing launcher constants.
    a=.006;rho=7850;M=1.1e6;eta=.4;mu0=4*np.pi*1e-7
    alpha=K*a;gamma=3*mu0*M*M*K*K/(rho*OMEGA*OMEGA)
    p=mod.SolverParams(alpha,3*1.225*.47/(8*rho*a*K),K*9.81/OMEGA**2,gamma,gamma/(eta*alpha**2))
    p.reverse_push_acceleration=0
    state=animation.prepare_initial_state(mod,p,dt,v0=-.075)
    mode=direction=None
    rows=[np.r_[0,state,0]];steps=round(seconds*OMEGA/dt);step=0
    sample_steps=round(.01*OMEGA/dt)
    sample_index=1
    error=None
    try:
        while step<steps:
            initializing=mode is None
            if sample_hz is not None and mode is not None:
                next_step=round(sample_index*OMEGA/(sample_hz*dt))
                chunk=min(max(1,next_step-step),steps-step)
            else:chunk=min(1 if mode is None else sample_steps,steps-step)
            state,mode,direction=mod.evolve_trajectory(state,dt,chunk,p,mode,direction,step)
            step+=chunk;rows.append(np.r_[step*dt/OMEGA,state,float(mode)])
            if sample_hz is not None and not initializing:sample_index+=1
    except RuntimeError as exc:error=str(exc)
    rows=np.asarray(rows)
    record={'mu_high':mu,'moment_scale':scale,'moment_A_m2':scale*MREF,'lambda':lam,
            'tilt_rad':float(tilt),'rpm':rpm,'sample_hz':sample_hz,'dt':dt,'seconds':seconds,'completed_s':float(rows[-1,0]),
            'error':error,'wall_s':time.perf_counter()-began,'binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest()}
    if error is None:
        width=min(60,seconds/3)
        windows=[metrics(rows,seconds-i*width,seconds-(i-1)*width) for i in [3,2,1]]
        record['windows']=windows
        record['metrics']=metrics(rows,seconds-2*width,seconds)
        periods=[w['period_s'] for w in windows]
        record['stationary']=bool(all(p is not None for p in periods) and np.ptp(periods)/np.mean(periods)<.01 and np.ptp([w['radius_mm'] for w in windows])<.5 and min(w['reverse_fraction'] for w in windows)>.99)
    if save:np.savez_compressed(OUT/(name+'.npz'),rows=rows)
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:record.get(k) for k in ['mu_high','moment_scale','lambda','seconds','error','stationary','metrics','wall_s']}),flush=True)
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mu',type=float,nargs='+',default=[1e-4]);ap.add_argument('--moment',type=float,default=3.04);ap.add_argument('--seconds',type=float,default=240);ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--lam',type=float,default=.004);a=ap.parse_args()
    for mu in a.mu:run(mu,a.moment,a.seconds,a.dt,a.lam,save=True)
