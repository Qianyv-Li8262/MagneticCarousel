"""Separate bounded beat modulation from radial escape; engineering tolerances."""
import numpy as np
def assess(table,omega,window=100):
    t=table[:,0];s=table[:,1:16];r=np.linalg.norm(s[:,1:3],axis=1)*17.5
    v=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/np.linalg.norm(s[:,1:3],axis=1)*omega*17.5
    angle=np.unwrap(np.arctan2(s[:,2],s[:,1]))
    chunks=[]
    for left in np.arange(t[-1]-3*window,t[-1],window):
        use=(t>=left)&(t<left+window);phi=angle[use];tt=t[use]
        if len(tt)<20:return {'bounded_stationary':False,'reason':'Too short for three windows'}
        coef=np.linalg.lstsq(np.c_[np.ones(len(phi)),np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi)],-v[use],rcond=None)[0]
        chunks.append({'time_s':[float(tt[0]),float(tt[-1])],'radius_mean_mm':float(r[use].mean()),'radius_min_mm':float(r[use].min()),'radius_max_mm':float(r[use].max()),'speed_mean_mm_s':float(-v[use].mean()),'period_s':float(-2*np.pi*(tt[-1]-tt[0])/(phi[-1]-phi[0])),'amplitude1_percent':float(np.hypot(coef[1],coef[2])/coef[0]*100),'reverse_fraction':float(np.mean(v[use]<0))})
    radii=np.array([c['radius_mean_mm'] for c in chunks]);speeds=np.array([c['speed_mean_mm_s'] for c in chunks]);periods=np.array([c['period_s'] for c in chunks]);amps=np.array([c['amplitude1_percent'] for c in chunks])
    use=t>=t[-1]-3*window
    radius_slope=float(np.polyfit(t[use]-t[use].mean(),r[use],1)[0])
    bounded=bool(np.ptp(radii)<.5 and np.ptp(speeds)/speeds.mean()<.02 and np.ptp(periods)/periods.mean()<.02 and abs(radius_slope)<.001 and min(c['reverse_fraction'] for c in chunks)>.99)
    return {'bounded_stationary':bounded,'window_s':window,'window_metrics':chunks,'mean_window_period_s':float(periods.mean()),'mean_window_speed_mm_s':float(speeds.mean()),'mean_window_amplitude1_percent':float(amps.mean()),'amplitude1_window_span_percentage_points':float(np.ptp(amps)),'radius_mean_mm':float(r[use].mean()),'radius_slope_mm_s':radius_slope,'engineering_criteria':'Three 100 s windows: mean radius span <0.5 mm, mean speed and period span <2%, radial drift slope <0.001 mm/s, >99% reverse motion. Bounded quasiperiodic motion is allowed; not proof of infinite-time stability.'}
