"""Replay the saved 15-component release state with the frozen FP64 build.

Run with the daily Python environment. No q reset or second magnetization occurs.
An optional air impulse is applied after --kick-at seconds, preserving q/spin.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'results'/'small-tilt-release-fit'/'release_case.json'


def load_case(path):
    case=json.loads(path.read_text(encoding='utf-8'))
    directory=ROOT/case['extension_directory']
    manifest=json.loads((directory/'build_config.json').read_text(encoding='utf-8'))
    binary=directory/manifest['binary']
    actual=hashlib.sha256(binary.read_bytes()).hexdigest()
    if actual!=case['binary_sha256']:
        raise RuntimeError('Saved case and extension differ. Rebuild/re-export the matching case first.')
    spec=importlib.util.spec_from_file_location('hysteresis',binary)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return case,mod


def plot_result(times,states,path,kick_at=None,title=None,capture_candidate=True,force_interval=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    pos=states[:,1:3]*17.5
    angle=np.unwrap(np.arctan2(pos[:,1],pos[:,0]));angle-=angle[0]
    omega=100*2*np.pi/60
    speed=np.linalg.norm(states[:,3:5],axis=1)*omega*17.5
    fig=Figure(figsize=(12,8),constrained_layout=True)
    FigureCanvasAgg(fig)
    axs=fig.subplots(2,2)
    fig.suptitle(title or 'Partial candidate — immediate reverse, then oscillatory capture (NOT a full experimental fit)')
    ax=axs[0,0];ax.plot(times,angle,lw=.9)
    if capture_candidate:ax.axhspan(-2,-1,alpha=.12,color='green')
    ax.set(xlabel='Time after release (s)',ylabel='Angular displacement (rad)')
    ax=axs[0,1];mask=times<=8;ax.plot(times[mask],angle[mask],lw=1)
    ax.set(xlabel='Time after release (s)',ylabel='Initial displacement (rad)')
    ax=axs[1,0];ax.plot(pos[:,0],pos[:,1],lw=.7,color='#8b95a5')
    ax.scatter(*pos[0],s=30,label='Release');ax.scatter(*pos[-1],s=30,label='Final')
    ax.set(xlabel='x (mm)',ylabel='y (mm)',aspect='equal');ax.legend()
    ax=axs[1,1];mask=times>=times[-1]-10
    ax.plot(times[mask],angle[mask],lw=.8)
    ax.set(xlabel='Time after replay starts (s)',ylabel='Raw final displacement (rad)')
    for ax in axs.flat:
        ax.grid(alpha=.2)
        if ax is not axs[1,0]:
            left,right=ax.get_xlim()
            if kick_at is not None and left<=kick_at<=right:
                ax.axvline(kick_at,color='red',ls='--',lw=.8)
            if force_interval is not None and force_interval[0]<right and force_interval[1]>left:
                ax.axvspan(max(left,force_interval[0]),min(right,force_interval[1]),color='orange',alpha=.18)
    fig.savefig(path,dpi=170);fig.clear()
    return pos,angle,speed


def animate_result(times,states,frame_stride,title='Partial candidate | visible rebound remains'):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(7,7));ax.set_aspect('equal')
    ax.set(xlim=(-105,105),ylim=(-105,105),xlabel='x (mm)',ylabel='y (mm)')
    theta=np.linspace(0,2*np.pi,500);ax.plot(70*np.cos(theta),70*np.sin(theta),color='#aaa',lw=.7)
    line,=ax.plot([],[],color='#3873ac',lw=.8)
    ball=Circle((70,0),5,color='#3873ac');ax.add_patch(ball)
    magnets=[Circle((0,0),14,color='#cc4a4a' if j%2==0 else '#4387bc',alpha=.28) for j in range(8)]
    for magnet in magnets:ax.add_patch(magnet)
    heading=ax.set_title(title)
    pos=states[:,1:3]*17.5
    indices=np.arange(0,len(times),frame_stride)
    def update(frame):
        i=int(indices[frame]);ball.center=tuple(pos[i]);line.set_data(pos[:i+1,0],pos[:i+1,1])
        for j,magnet in enumerate(magnets):
            phase=states[i,0]+j*2*np.pi/8;magnet.center=(70*np.cos(phase),70*np.sin(phase))
        heading.set_text(f'{title} | t={times[i]:.2f} s')
        return [ball,line,heading,*magnets]
    animation=FuncAnimation(fig,update,frames=len(indices),interval=30,blit=False,repeat=False)
    plt.show()
    return animation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=DEFAULT)
    p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--kick-at',type=float,default=None)
    p.add_argument('--kick-mm-s',type=float,default=0.)
    p.add_argument('--animate',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'prepared-release-replay')
    args=p.parse_args()
    if args.seconds<=0 or args.kick_mm_s<0 or (args.kick_at is not None and not 0<args.kick_at<args.seconds):
        p.error('Time must be positive; kick time must lie inside the run; reverse kick must be nonnegative.')
    if args.kick_mm_s and args.kick_at is None:p.error('Specify --kick-at for an after-capture impulse.')
    case,mod=load_case(args.case)
    params=mod.SolverParams(*case['solver_parameters'])
    state=np.array(case['initial_state'],dtype=mod.real_dtype)
    mode=case['initial_sliding'];direction=np.array(case['initial_sliding_direction'],dtype=mod.real_dtype)
    dt=case['dt'];omega=case['disk_angular_speed_rad_s'];stride=case['sample_stride']
    steps=round(args.seconds*omega/dt)
    kick_step=round(args.kick_at*omega/dt) if args.kick_at is not None else None
    records=[np.r_[0,state,float(mode)]];step=0;kicked=False;kick_state=None
    while step<steps:
        if kick_step is not None and step==kick_step and not kicked:
            tangent=np.array([-state[2],state[1]])/np.linalg.norm(state[1:3])
            state[3:5]-=args.kick_mm_s/1000*(4/.07)/omega*tangent
            slip=state[3:5]+params.alpha*np.array([-state[10],state[9]])
            if np.linalg.norm(slip)>1e-10:mode=True;direction=slip/np.linalg.norm(slip)
            else:mode,direction=mod.release_contact_mode(state,params)
            kick_state=state.copy();kicked=True
            records.append(np.r_[step*dt,state,float(mode)])
        chunk=min(1 if step==0 else stride,steps-step)
        if kick_step is not None and not kicked:chunk=min(chunk,kick_step-step)
        state,mode,direction=mod.evolve_trajectory(state,dt,chunk,params,mode,direction,step)
        step+=chunk;records.append(np.r_[step*dt,state,float(mode)])
    table=np.asarray(records);times=table[:,0]/omega;states=table[:,1:16]
    args.output.mkdir(parents=True,exist_ok=True)
    pos,angle,speed=plot_result(times,states,args.output/'trajectory.png',args.kick_at,
           title=case.get('plot_title'),capture_candidate=case.get('scenario')!='sustained_reverse_after_push')
    np.savez_compressed(args.output/'trajectory.npz',state_table=table,time_s=times,
                        position_mm=pos,angle_rad=angle,speed_mm_s=speed)
    np.savetxt(args.output/'trajectory.csv',np.column_stack([times,pos,angle,speed,table[:,-1]]),
               delimiter=',',header='time_s,x_mm,y_mm,angle_rad,speed_mm_s,sliding',comments='')
    last=times>times[-1]-10
    summary=dict(full_experimental_fit=False,
       final_angle_rad=float(np.mean(angle[last])),
       minimum_angle_rad=float(angle.min()),
       last10s_position_box_mm=float(np.linalg.norm(np.ptp(pos[last],axis=0))),
       last10s_mean_speed_mm_s=float(np.mean(speed[last])),
       last10s_rolling_fraction=float(np.mean(table[last,-1]==0)),
       kick_at_s=args.kick_at,kick_mm_s=args.kick_mm_s,
       post_kick_state=None if kick_state is None else kick_state.tolist())
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),**summary),indent=2))
    if args.animate:animate_result(times,states,max(1,round(.03/(stride*dt/omega))),
                                   title=case.get('animation_title','Partial candidate | visible rebound remains'))


if __name__=='__main__':main()
