"""Experimental comparison and long-time checks for the high-C continuation."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_video56c6_high_c import g,scored
from report_video56c6_rolling import smooth,laps,threshold

def read(name):return json.loads((g.OUT/name).read_text())
def load(r):return pd.read_csv(g.OUT/(r['key']+'.csv'))
def metrics(r,long=None):
    a=load(r);pp=laps(a)
    w=(pp[:,2]>=35)&(pp[:,3]<=72) if len(pp) else np.zeros(0,dtype=bool)
    result=dict(C=r['config']['c'],quiet_speed=r['quiet_speed'],startup_rmse=r['startup_rmse'],
        radius_rmse=r['radius_rmse'],scale=r['scale'],period_s=float(pp[w,1].mean()) if w.any() else None,
        crossings={str(z):threshold(np.array(r['model_v']),z) for z in [50,150,250]})
    if long:
        la=load(long);tail=la.time_s>=220;s=r['scale'];_,sr=smooth(la,'radius_mm',s)
        result.update(long_status=long['status'],long_end_s=float(la.time_s.iloc[-1]),
            long_over_mm=float(sr.max()-la.radius_mm[tail].mean()/s),
            long_tail_speed=float(la.speed_mm_s[tail].mean()/s),
            long_tail_radius=float(la.radius_mm[tail].mean()/s),
            long_period_s=float(2*np.pi/abs(np.polyfit(la.time_s[tail],np.unwrap(la.phi_rad)[tail],1)[0])),
            rolling_fraction=float(1-la.sliding.mean()))
    return result

if __name__=='__main__':
    base=read('baseline.json');best=read('selected.json');small=read('best_smallover_priority.json')
    old=json.loads((g.f.ROOT/'runs/video56c6-rolling-fit-20261017/selected.json').read_text())
    old=scored(old['x'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        long,half,base_long,small_long,small_half=list(pool.map(lambda p:g.evaluate(p[0],**p[1]),
            [(best['x'],dict(seconds=240.)),(best['x'],dict(dt=.00025)),(base['x'],dict(seconds=240.)),
             (small['x'],dict(seconds=240.)),(small['x'],dict(dt=.00025))]))
    a=load(best);ha=load(half)
    n=min(len(a),len(ha))
    report=dict(target=dict(quiet_speed=float(g.TARGET[g.QUIET,2].mean()),period_s=g.f.TARGET_PERIOD,
            crossings={str(z):threshold(g.TARGET[:,2],z) for z in [50,150,250]}),
        previous_behavior_candidate=metrics(base,base_long),previous_video_fit=metrics(old),
        selected=metrics(best,long),smallover_candidate=metrics(small,small_long),
        fixed_C065_candidate=metrics(read('best_fixed_C065_material.json')),
        parameters=best,smallover_parameters=small,validation=dict(half_status=half['status'],
            max_speed_difference=float(np.max(abs(a.speed_mm_s.iloc[:n]-ha.speed_mm_s.iloc[:n]))),
            max_radius_difference=float(np.max(abs(a.radius_mm.iloc[:n]-ha.radius_mm.iloc[:n]))),
            smallover_half_status=small_half['status'],
            smallover_max_speed_difference=float(np.max(abs(load(small).speed_mm_s-load(small_half).speed_mm_s))),
            smallover_max_radius_difference=float(np.max(abs(load(small).radius_mm-load(small_half).radius_mm)))),
        assumptions='h=32mm, normal moment, zero initial velocity/spin/memory, phase=.58, no blowing/premagnetization. One constant spatial factor, no time shift/rescaling. Parameters are conditional fit values; no claim of material measurement or global optimum.')
    (g.OUT/'fit_report.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
    fig,ax=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    fig.suptitle('56c6 实验对比：继续提高 C 的联合调参',fontsize=16)
    ax[0,0].plot(g.TARGET[:,0],g.TARGET[:,2],'k.',label='实验：1 秒平均',ms=4)
    ax[0,1].plot(g.TARGET[:,0],g.TARGET[:,3],'k.',label='实验：1 秒平均',ms=4)
    for r,col,label in [(base,'#94a3b8','上一轮小过冲候选'),(old,'#2563eb','此前实验拟合 C=0.420'),
                        (best,'#e8790c',f"本轮启动优先 C={best['config']['c']:.3f}"),
                        (small,'#059669',f"本轮减过冲优先 C={small['config']['c']:.3f}")]:
        frame=load(r)
        for axis,key in [(ax[0,0],'speed_mm_s'),(ax[0,1],'radius_mm')]:
            xx,yy=smooth(frame,key,r['scale']);axis.plot(xx,yy,color=col,lw=1.7,label=label)
        pp=laps(frame)
        if len(pp):ax[1,0].plot(pp[:,0],pp[:,1],color=col,label=label)
    pl=g.f.LAPS;w=(pl['start_pts_s']>=20)&(pl['end_pts_s']<=72)
    ax[1,0].plot((pl['start_pts_s'][w]+pl['end_pts_s'][w])/2,pl['period_s'][w],'k.',label='实验逐圈周期')
    ax[1,0].axhline(g.f.TARGET_PERIOD,color='gray',ls=':')
    la=load(long)
    for r,col,label in [(base_long,'#94a3b8','上一轮候选'),(long,'#e8790c','本轮启动优先'),
                        (small_long,'#059669','本轮减过冲优先')]:
        xx,yy=smooth(load(r),'radius_mm',r['scale']);ax[1,1].plot(xx-4.9,yy,color=col,label=label)
    ax[0,0].set(xlim=(5,42),xlabel='视频时间 PTS (s)，4.9 s 释放',ylabel='速率，原 CV 标度 (mm/s)',title='等待与启动：没有移动或缩放时间轴')
    ax[0,1].set(xlim=(5,72),xlabel='视频时间 PTS (s)',ylabel='半径，原 CV 标度 (mm)',title='所有位置与速度共用一个空间因子')
    ax[1,0].set(xlim=(22,72),ylim=(1.3,2.6),xlabel='视频时间 PTS (s)',ylabel='逐圈周期 (s)',title='对比观测时段的周期，而非远期稳定周期')
    ax[1,1].set(xlim=(0,240),xlabel='释放后时间 (s)',ylabel='半径，原 CV 标度 (mm)',title=f"240 s 检查：橙色过冲 {report['selected']['long_over_mm']:.2f}、绿色 {report['smallover_candidate']['long_over_mm']:.2f} mm")
    for axis in ax.flat:axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.savefig(g.OUT/'comparison.png',dpi=155)
    profiles=read('profiles.json');fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for prefix,color,label in [('C_only_','#2563eb','仅提高 C'),('C_sigma_','#e8790c','同时补偿电导率')]:
        rr=[z['record'] for z in profiles if z['name'].startswith(prefix)]
        cc=[r['config']['c'] for r in rr]
        axs[0].plot(cc,[r['quiet_speed'] for r in rr],'o-',color=color,label=label)
        # Use 35-72 speed, instead of a meaningless large 'period' in non-orbiting runs.
        axs[1].plot(cc,[np.mean(np.array(r['model_v'])[g.LATE]) for r in rr],'o-',color=color,label=label)
    axs[0].axhline(report['target']['quiet_speed'],color='k',ls=':',label='实验平均')
    axs[1].axhline(float(g.TARGET[g.LATE,2].mean()),color='k',ls=':',label='实验平均')
    axs[0].set(xlabel='C',ylabel='等待段平均速率 (mm/s)',title='等待段：视频 9–18 s')
    axs[1].set(xlabel='C',ylabel='后段平均速率 (mm/s)',title='后段：视频 35–72 s')
    for axis in axs:axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.savefig(g.OUT/'C_profiles.png',dpi=150)
    (g.OUT/'REPORT.md').write_text('# High-C comparison with 56c6\n\n'+
        report['assumptions']+'\n\n```json\n'+json.dumps({k:v for k,v in report.items() if 'parameters' not in k},indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if 'parameters' not in k},indent=2),flush=True)
