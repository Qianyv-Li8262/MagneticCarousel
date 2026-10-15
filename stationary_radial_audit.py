"""Static disk phase and consistent body-field derivative; radial oscillations."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/material-response-20261013'
binary=next((ROOT/'scratch/material-response-20261013-static-source').glob('material_audit*.pyd'));spec=importlib.util.spec_from_file_location('material_audit',binary);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=np.array([605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882]);k=4/.07;omega=199.8313400018371*2*np.pi/60;alpha=k*.006;gamma=3*4*np.pi*1e-7*(1.1e6)**2*k*k/(7850*omega**2)
p=m.SolverParams(alpha,3*1.225*.47/(8*7850*.006*k),k*9.81/omega**2,gamma,gamma/(.4*alpha**2));p.reverse_push_acceleration=0
m.configure_disk_rate(0.);m.configure_laser_tilt(0.,np.pi/2)
records=[];fig,ax=plt.subplots(2,1,figsize=(11,7),layout='constrained')
cases=[('reference',base,1.,0.001),('reference, 3 mm',base,1.,.003),('mur=1e6',np.array([1e6,*base[1:]]),1.,.001),('previous 3.04x moment',base,3.04,.001)]
r0=.06981645555072052
for name,material,scale,amplitude in cases:
 m.configure_material(*material);m.configure_magnet_moment_scale(scale);m.configure_rolling_resistance(1e-5,1e-4,.004)
 state=np.zeros(15);state[5]=1.;state[1]=(r0+amplitude)*k;mode=direction=None;step=0;dt=.00025;rows=[np.r_[0,state,0]]
 for j in range(1,15001):
  n=round(j*.002*omega/dt)-step;state,mode,direction=m.evolve_trajectory(state,dt,n,p,mode,direction,step);step+=n;rows.append(np.r_[step*dt/omega,state,float(mode)])
 a=np.array(rows);r=np.linalg.norm(a[:,2:4],axis=1)/k*1000;peaks,_=find_peaks(r,prominence=.02,distance=50);times=a[peaks,0];early=times[times<10];freq=1/np.diff(early) if len(early)>1 else np.array([])
 rec=dict(name=name,moment_scale=scale,initial_radial_offset_mm=amplitude*1000,material=material.tolist(),disk_angle_change_rad=float(np.ptp(a[:,1])),measured_radial_peak_times_s=times.tolist(),first_10s_frequency_range_hz=[float(freq.min()),float(freq.max())] if len(freq) else None,first_10s_mean_frequency_hz=float(freq.mean()) if len(freq) else None,final_radius_mm=float(r[-1]),rolling_fraction=float(np.mean(a[:,16]==0)))
 records.append(rec);print(json.dumps(rec),flush=True);np.savez_compressed(OUT/f'static_trajectory_{len(records)}.npz',rows=a)
 ax[0].plot(a[:,0],r,label=name);ax[1].plot(a[:,0],r,label=name)
for a in ax:a.set(xlabel='Physical time (s)',ylabel='Ball radius (mm)');a.grid(alpha=.2);a.legend(fontsize=8)
ax[0].set(xlim=(0,5),title='Disk stopped: radial disturbance with full JA/contact dynamics');ax[1].set(xlim=(0,30));fig.savefig(OUT/'static_radial_oscillations.png',dpi=160);plt.close(fig)
(OUT/'stationary_simulation_summary.json').write_text(json.dumps(records,indent=2))
