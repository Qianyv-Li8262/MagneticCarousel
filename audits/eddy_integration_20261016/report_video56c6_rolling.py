"""Freeze the selected rolling fit, validate and plot against unchanged CV targets."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_video56c6_rolling as g

def load(r):return pd.read_csv(g.OUT/(r['key']+'.csv'))
def laps(a):
    w=a.time_s.to_numpy()>15;t=a.time_s.to_numpy()[w]+4.9
    phi=-np.unwrap(a.phi_rad.to_numpy())[w]
    if np.any(np.diff(phi)<=0):return np.empty((0,4))
    levels=2*np.pi*np.arange(np.floor(phi[0]/(2*np.pi))+1,np.floor(phi[-1]/(2*np.pi))+1)
    cross=np.interp(levels,phi,t)
    return np.column_stack(((cross[1:]+cross[:-1])/2,np.diff(cross),cross[:-1],cross[1:]))
def smooth(a,k,s=1):
    n=round(1/np.median(np.diff(a.time_s)));kernel=np.ones(n)/n
    return np.convolve(a.time_s+4.9,kernel,mode='valid'),np.convolve(a[k],kernel,mode='valid')/s
def threshold(v,level):
    ix=np.flatnonzero((g.TARGET[:-1,0]>8)&(v[:-1]<level)&(v[1:]>=level))
    if not len(ix):return None
    j=ix[0];return float(np.interp(level,v[j:j+2],g.TARGET[j:j+2,0]))

if __name__=='__main__':
    candidates=[json.loads(p.read_text()) for p in g.OUT.glob('best_*.json')]
    # Freeze and independently rerun inputs, rather than trusting optimizer caches.
    selected=min(candidates,key=lambda r:r['score'])
    best=g.evaluate(selected['x'])
    band=g.evaluate(json.loads((g.OUT/'best_measured_band.json').read_text())['x'])
    (g.OUT/'selected.json').write_text(json.dumps(best,indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        half,long,band_long,band_half=list(pool.map(lambda args:g.evaluate(args[0],**args[1]),[(best['x'],dict(dt=.00025)),(best['x'],dict(seconds=240.)),(band['x'],dict(seconds=240.)),(band['x'],dict(dt=.00025))]))
    a=load(best);ha=load(half);la=load(long);p=laps(a)
    w=(p[:,2]>=35)&(p[:,3]<=72);tail=la.time_s>=220
    original=json.loads((g.OLD/'selected_wait_candidate.json').read_text())
    old=pd.read_csv(g.OLD/(original['native_key']+'.csv'))
    baseline=g.evaluate(np.r_[original['x'][:5],original['x'][6],.0049,.0049,.26])
    ba=load(baseline);n=min(len(old),len(ba));n2=min(len(a),len(ha))
    validation=dict(baseline_equivalence_max_speed_mm_s=float(np.max(abs(old.speed_mm_s.iloc[:n]-ba.speed_mm_s.iloc[:n]))),
        baseline_equivalence_max_radius_mm=float(np.max(abs(old.radius_mm.iloc[:n]-ba.radius_mm.iloc[:n]))),
        half_dt_max_speed_difference_mm_s=float(np.max(abs(a.speed_mm_s.iloc[:n2]-ha.speed_mm_s.iloc[:n2]))),
        half_dt_max_radius_difference_mm=float(np.max(abs(a.radius_mm.iloc[:n2]-ha.radius_mm.iloc[:n2]))),
        half_status=half['status'],long_status=long['status'],long_end_s=float(la.time_s.iloc[-1]),
        long_rolling_fraction=float(1-la.sliding.mean()),long_last20_speed_true_mm_s=float(la.speed_mm_s[tail].mean()) if tail.any() else None,
        long_last20_radius_true_mm=float(la.radius_mm[tail].mean()) if tail.any() else None,
        long_last20_period_s=float(2*np.pi/abs(np.polyfit(la.time_s[tail],np.unwrap(la.phi_rad)[tail],1)[0])) if tail.sum()>10 else None)
    band_a=load(band);band_ha=load(band_half);band_la=load(band_long);band_p=laps(band_a)
    band_w=(band_p[:,2]>=35)&(band_p[:,3]<=72);band_tail=band_la.time_s>=220
    band_validation=dict(half_status=band_half['status'],long_status=band_long['status'],
        half_dt_max_speed_difference_mm_s=float(np.max(abs(band_a.speed_mm_s-band_ha.speed_mm_s))),
        half_dt_max_radius_difference_mm=float(np.max(abs(band_a.radius_mm-band_ha.radius_mm))),
        long_rolling_fraction=float(1-band_la.sliding.mean()),
        long_last20_period_s=float(2*np.pi/abs(np.polyfit(band_la.time_s[band_tail],np.unwrap(band_la.phi_rad)[band_tail],1)[0])) if band_tail.sum()>10 else None,
        complete_lap_period_s=float(band_p[band_w,1].mean()))
    report=dict(selected=best,measured_band_candidate=band,measured_band_long_run=band_long,measured_band_validation=band_validation,validation=validation,target_period_s=g.f.TARGET_PERIOD,
        complete_lap_period_s=float(p[w,1].mean()) if w.any() else None,
        complete_lap_period_error_percent=float((p[w,1].mean()/g.f.TARGET_PERIOD-1)*100) if w.any() else None,
        crossings={str(z):dict(video=threshold(g.TARGET[:,2],z),model=threshold(np.array(best['model_v']),z)) for z in [50,150,250]},
        target_quiet_speed=float(g.TARGET[g.QUIET,2].mean()),
        assumptions='Zero initial speed/spin/magnetic memories; no blowing, no premagnetization; fixed h=32mm and normal magnetic moment, no time rescaling; one spatial factor, pure rolling speed vc=R*horizontal omega_c. Gaussian transition has a tail inside measured speeds. Material and rolling parameters are conditional fit values, not measurements.')
    (g.OUT/'fit_report.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    fig.suptitle('56c6：允许高斯滚阻随速度下降后的联合拟合',fontsize=16)
    axs[0,0].plot(g.TARGET[:,0],g.TARGET[:,2],'k.-',label='视频 1秒平均',lw=1,ms=4)
    axs[0,1].plot(g.TARGET[:,0],g.TARGET[:,3],'k.',label='视频 1秒平均',ms=4)
    for frame,s,color,label in [(old,original['scale'],'#94a3b8','原恒定滚阻：等待优先'),(load(band),band['scale'],'#059669','低速滚阻限制在实测工作范围'),(a,best['scale'],'#e8790c','放开低速滚阻的探索拟合')]:
        for ax,k in [(axs[0,0],'speed_mm_s'),(axs[0,1],'radius_mm')]:
            xx,yy=smooth(frame,k,s);ax.plot(xx,yy,color=color,lw=1.7,label=label)
        pp=laps(frame);axs[1,0].plot(pp[:,0],pp[:,1],color=color,label=label)
    pl=g.f.LAPS;w=(pl['start_pts_s']>=20)&(pl['end_pts_s']<=72)
    axs[1,0].plot((pl['start_pts_s'][w]+pl['end_pts_s'][w])/2,pl['period_s'][w],'k.',label='视频逐圈周期')
    axs[1,0].axhline(g.f.TARGET_PERIOD,color='gray',ls=':',label='视频平均')
    rr=best['rolling'];vv=np.linspace(0,.4,401);mm=rr['mu_high']+(rr['mu_low']-rr['mu_high'])*np.exp(-(vv/rr['vc_m_s'])**2)
    axs[1,1].plot(vv,mm,color='#e8790c',label='拟合的 μ(v)')
    br=band['rolling'];bm=br['mu_high']+(br['mu_low']-br['mu_high'])*np.exp(-(vv/br['vc_m_s'])**2)
    axs[1,1].plot(vv,bm,color='#059669',label='限制低速滚阻的候选')
    axs[1,1].axhline(.0049,color='#94a3b8',ls='--',label='原实测恒定主项 0.0049')
    axs[1,1].axvspan(.026,.135,color='#60a5fa',alpha=.12,label='原测低速段范围')
    axs[1,1].axvspan(.188,.218,color='#60a5fa',alpha=.25,label='原测高速短段范围')
    axs[1,1].axvline(rr['vc_m_s'],color='#e8790c',ls=':',label=f"过渡尺度 {rr['vc_m_s']:.3f} m/s")
    axs[0,0].set(xlim=(5,40),xlabel='视频 PTS (s)，4.9 s释放',ylabel='球心速率，原CV标度 (mm/s)',title='启动与加速：时间轴未缩放')
    axs[0,1].set(xlim=(5,72),xlabel='视频 PTS (s)',ylabel='半径，原CV标度 (mm)',title=f"统一空间尺度 s={best['scale']:.4f}")
    axs[1,0].set(xlim=(20,72),ylim=(1.3,2.3),xlabel='视频 PTS (s)',ylabel='逐圈周期 (s)',title='周期不受空间尺度拟合影响')
    axs[1,1].set(xlabel='真实纯滚动速度 (m/s)',ylabel='有效滚阻系数 μ',title='显示实测范围内的高斯过渡尾部',ylim=(0,max(.007,rr['mu_low']*1.08)))
    for ax in axs.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(g.OUT/'comparison.png',dpi=155)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax2=ax.twinx()
    for k,color,label,axis in [('speed_mm_s','#e8790c','球心速率 / s',ax),('radius_mm','#2563eb','轨道半径 / s',ax2)]:
        xx,yy=smooth(la,k,best['scale']);axis.plot(xx-4.9,yy,color=color,label=label)
    ax.set(xlabel='释放后时间 (s)',ylabel='速率，原CV标度 (mm/s)',title='所选候选：延长到240 s检验');ax2.set_ylabel('半径，原CV标度 (mm)');ax.grid(alpha=.25)
    fig.savefig(g.OUT/'long_run.png',dpi=150)
    print(json.dumps({k:v for k,v in report.items() if k not in ['selected','measured_band_candidate','measured_band_long_run']},indent=2),flush=True)
