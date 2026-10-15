"""Measure slow speed oscillation amplitudes in the screenshot's time windows."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent/'runs/laser-tilt-20261009/rolling_candidates'
table=np.load(root/'production_600s_trajectory.npz')['free']
summary=json.loads((root/'production_600s_summary.json').read_text())
t=table[:,0];state=table[:,1:16];omega=200*2*np.pi/60;units_to_mm_s=omega*17.5
tau=t*omega;r=np.linalg.norm(state[:,1:3],axis=1)
speed=-(state[:,1]*state[:,4]-state[:,2]*state[:,3])/r
interval=np.median(np.diff(t));initial_speed=np.linalg.norm(state[0,3:5])
records=[];smooth_curves={}
for seconds in [.1,.2,.3]:
    window=int(round(seconds/interval))|1
    curve=savgol_filter(speed,window,3);smooth_curves[seconds]=curve
    peaks,_=find_peaks(curve,distance=round(2/interval),prominence=.015)
    troughs,_=find_peaks(-curve,distance=round(2/interval),prominence=.015)
    for label,left,right in [('transient',800,2000),('late_shown',3500,5000)]:
        used=(tau>=left)&(tau<=right);pp=peaks[used[peaks]];tt=troughs[used[troughs]]
        mean=float(speed[used].mean());high=float(curve[pp].mean());low=float(curve[tt].mean());half=(high-low)/2
        records.append({'region':label,'dimensionless_time_window':[left,right],'physical_time_window_s':[left/omega,right/omega],'smoothing_window_s':seconds,'mean_speed_2d':mean,'mean_speed_mm_s':mean*units_to_mm_s,'average_peak_2d':high,'average_trough_2d':low,'slow_half_amplitude_2d':half,'slow_half_amplitude_mm_s':half*units_to_mm_s,'slow_half_amplitude_over_local_mean_percent':100*half/mean,'slow_half_amplitude_over_initial_speed_percent':100*half/initial_speed,'raw_full_extrema_half_range_over_local_mean_percent':100*(speed[used].max()-speed[used].min())/(2*mean),'peaks_used':len(pp),'troughs_used':len(tt)})
metrics=summary['metrics'];a1=metrics['mean_tangential_speed_mm_s']*metrics['first_harmonic_relative_half_amplitude_percent']/100
gravity=100*9.81*np.sin(.002777749200346311)*(metrics['radius_mean_mm']/1000)/(1.4*(metrics['mean_tangential_speed_mm_s']/1000)**2)
result={'amplitude_definition':'A=(mean cycle peak - mean cycle trough)/2; percentages use local time-mean speed unless explicitly normalized to initial speed. Signed reverse tangential velocity is converted to positive speed.','initial_speed_2d':float(initial_speed),'initial_speed_mm_s':float(initial_speed*units_to_mm_s),'window_measurements':records,'late_complete_lap_first_harmonic_half_amplitude_percent':metrics['first_harmonic_relative_half_amplitude_percent'],'late_complete_lap_first_harmonic_half_amplitude_mm_s':a1,'gravity_only_rolling_order_estimate_percent':gravity,'source':'production_600s_trajectory.npz','interpretation':'The early oscillation is a transient. Only the late steady periodic modulation should be compared to the steady-orbit tilt estimate.'}
(root/'transient_amplitude.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
shown=tau<=5200;curve=smooth_curves[.2]
axes[0].plot(tau[shown][::5],speed[shown][::5],lw=.7,color='0.65',label='Raw reverse tangential speed')
axes[0].plot(tau[shown][::5],curve[shown][::5],lw=1.1,color='tab:orange',label='Slow oscillation (0.2 s smoothing)')
axes[0].axvspan(800,2000,color='tab:blue',alpha=.12);axes[0].axvspan(3500,5000,color='tab:green',alpha=.12)
axes[0].set(xlabel='Time (2D units, matching animation)',ylabel='Positive tangential speed (2D units)')
for ax,label,left,right,color in [(axes[1],'transient',800,2000,'tab:blue'),(axes[2],'late_shown',3500,5000,'tab:green')]:
    used=(tau>=left)&(tau<=right);row=next(q for q in records if q['region']==label and q['smoothing_window_s']==.2)
    ax.plot(tau[used],speed[used],color='0.7',lw=.6);ax.plot(tau[used],curve[used],color=color,lw=1)
    ax.axhline(row['mean_speed_2d'],ls='--',color='black',label='Local time-mean')
    ax.axhline(row['average_peak_2d'],ls=':',color=color);ax.axhline(row['average_trough_2d'],ls=':',color=color)
    ax.set(xlabel='Time (2D units)',ylabel='Speed (2D units)',title=f'{label}: half amplitude / local mean = {row["slow_half_amplitude_over_local_mean_percent"]:.1f}%; / initial speed = {row["slow_half_amplitude_over_initial_speed_percent"]:.1f}%')
for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=9)
fig.suptitle('Transient versus late periodic modulation | 200 RPM | lambda=0.006')
fig.savefig(root/'transient_amplitude.png',dpi=150);plt.close(fig)
print(json.dumps(result,indent=2))
