"""Export full push protocol plus an unforced initial state for ordinary replay."""
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'sustained-push'
DEST=ROOT/'results'/'sustained-push'


def plot_protocol(path,start,record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.load(path);table=a['state_table'];t=a['time_s'];s=table[:,1:16]
    angle=a['angle_rad'];off=record['actual_push_seconds'];omega=100*2*np.pi/60
    radius=np.linalg.norm(s[:,1:3],axis=1)*17.5
    rate=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/np.sum(s[:,1:3]**2,axis=1)*omega
    velocity=s[:,3:5];slip=velocity+(2/7)*np.column_stack([-s[:,10],s[:,9]])
    ratio=np.linalg.norm(slip,axis=1)/np.maximum(np.linalg.norm(velocity,axis=1),1e-12)
    free=t>=off
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    fig.suptitle('Continuous push unlocks sustained reverse motion — contact slip remains')
    ax=axs[0,0];ax.plot(t,angle/(2*np.pi),lw=1)
    ax.axvspan(0,off,color='orange',alpha=.18,label='External push on')
    ax.axvline(off,color='#a65a00',ls='--',lw=.9);ax.legend()
    ax.set(xlabel='Time after push starts (s)',ylabel='Ball revolutions (reverse < 0)')
    ax=axs[0,1];mask=t<off+20
    ax.plot(t[mask],-rate[mask],lw=.35,alpha=.35,label='Raw')
    window=max(1,round(.15/np.median(np.diff(t))))
    ax.plot(t[mask],-uniform_filter1d(rate,window)[mask],lw=1,label='0.15 s average')
    ax.axvspan(0,off,color='orange',alpha=.18);ax.axvline(off,color='#a65a00',ls='--',lw=.9)
    ax.set(xlabel='Time after push starts (s)',ylabel='Reverse angular speed (rad/s)');ax.legend()
    ax=axs[1,0];mask=free&(t>t[-1]-10)
    ax.plot(t[mask]-off,radius[mask],lw=.7)
    ax.axhline(70,color='#777',ls='--',lw=.8)
    ax.set(xlabel='Time since force removal (s)',ylabel='Orbit radius (mm)')
    ax=axs[1,1];ax.plot(t[mask]-off,100*ratio[mask],lw=.65)
    ax.axhline(100*np.median(ratio[mask]),ls='--',color='red',lw=.8,label='Median')
    ax.set(xlabel='Time since force removal (s)',ylabel='Contact slip / center speed (%)');ax.legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(DEST/'protocol_trajectory.png',dpi=160);plt.close(fig)
    return dict(time_s=t,states=s,radius_mm=radius,angular_speed_rad_s=rate,slip_ratio=ratio,angle_rad=angle)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rows=json.loads((RUN/'adaptive.json').read_text(encoding='utf-8'))
    record=next(r for r in rows if r['force_mN']==.7 and r['dt']==.0005)
    assert record['stable_under_push'] and record['after_force_removal']['sustained_reverse']
    base=json.loads((ROOT/'results'/'small-tilt-release-fit'/'release_case.json').read_text(encoding='utf-8'))
    # An ordinary unforced binary can replay the exact state after the push.
    reverse=dict(base)
    reverse.update(scenario='sustained_reverse_after_push',
        status='sustained_reverse_motion_with_contact_slip_not_full_experimental_fit',
        initial_state=record['state_at_force_removal'],initial_sliding=record['mode_at_force_removal'],
        initial_sliding_direction=record['direction_at_force_removal'],dt=.0005,sample_stride=40,
        initial_state_provenance='Continuous 0.7 mN reverse tangent push until three settled lap periods, then FORCE OFF',
        pre_magnetization='History already included in q and attitude; do not prepare again',
        plot_title='Force-free sustained reverse orbit (contact slip remains)',
        animation_title='Force removed | sustained reverse motion | contact slip remains',
        metric_record=record['after_force_removal'])
    (DEST/'reverse_running_initial_state.json').write_text(json.dumps(reverse,indent=2,ensure_ascii=False),encoding='utf-8')
    # The complete protocol additionally needs the explicit sustained-force API.
    directory=ROOT/'scratch'/'legacy-release-source'/'build'/'release'/'extensions'/'fp64'
    shutil.copytree(directory,DEST/'extension',dirs_exist_ok=True)
    manifest=json.loads((directory/'build_config.json').read_text(encoding='utf-8'))
    forced=dict(base);forced.update(extension_directory='results/sustained-push/extension',binary_sha256=manifest['binary_sha256'])
    (DEST/'force_capable_release_case.json').write_text(json.dumps(forced,indent=2,ensure_ascii=False),encoding='utf-8')
    protocol=dict(force_capable_case='force_capable_release_case.json',
        force_mN=.7,direction='reverse tangent following instantaneous ball position',
        force_applied_at_center=True,direct_external_torque=False,
        capture_after_release_s=45.,dt=.0005,max_push_seconds=30.,
        stop_rule='After at least four reverse laps, last three lap periods differ <1%, mean radii differ <0.5 mm, each lap is >99% reverse',
        observed_push_seconds=record['actual_push_seconds'],force_free_observation_seconds=180.,
        full_experimental_fit=False,remaining_mismatches=['initial release rebound','significant contact slip during steady reverse motion'],
        record=record)
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    for name in ['sweep.json','adaptive.json','baseline_validation.json']:
        shutil.copy2(RUN/name,DEST/name)
    shutil.copy2(record['trace'],DEST/'protocol_trajectory.npz')
    full_replay=RUN/'full-protocol-replay'
    if full_replay.exists():
        for extension in ['npz','csv','png']:
            shutil.copy2(full_replay/f'trajectory.{extension}',DEST/f'full_release_trajectory.{extension}')
        shutil.copy2(full_replay/'summary.json',DEST/'full_release_summary.json')
    values=plot_protocol(record['trace'],45.,record)
    table=np.load(record['trace'])['state_table']
    np.savetxt(DEST/'protocol_trajectory.csv',np.column_stack([values['time_s'],table[:,2:4]*17.5,
        values['angle_rad'],values['angular_speed_rad_s'],values['radius_mm'],values['slip_ratio'],table[:,17],table[:,16]]),
        delimiter=',',header='time_after_push_start_s,x_mm,y_mm,angle_rad,angular_speed_rad_s,radius_mm,slip_ratio,external_force_mN,sliding',comments='')
    post=record['after_force_removal']
    validation=json.loads((RUN/'baseline_validation.json').read_text(encoding='utf-8'))
    overview='\n'.join(f"| {r['force_mN']} | {r['dt']} | {r.get('actual_push_seconds',0):.5f} | {r.get('stable_under_push',False)} | {r.get('after_force_removal',{}).get('net_ball_revolutions',0):.5f} |" for r in rows)
    report=f'''# 持续吹气、达到稳态后撤力

**可以复现“持续推动后，撤力仍自持反向运动”的后半段。仍未复现初始无回摆与严格纯滚动。**

按实验澄清，外力方向跟随球，始终沿局部反向切线；不是在释放时吹气，也不是反复修改速度。几何、JA参数、0.8021409°倾角、摩擦与机械参数都沿用上轮小倾角候选。下磁铁盘始终100 rpm。

这里把吹气理想化为通过球心的恒定水平力，没有加入直接气动力矩或额外法向载荷。0.7 mN是探索值，不是实验风力测量。力在原求解器的全部中间步与接触分支判断中加入，撤力后严格为零。

## 结果

- 从零速释放初态开始，在45秒时（已在近停区数秒）开始持续吹气。
- **0.7 mN持续约 {record['actual_push_seconds']:.5f} 秒**，达到逐圈重复的反向运动后自动撤力。
- 判据：完成至少4圈；最近3圈的周期相差小于1%，平均半径相差小于0.5mm，每圈超过99%时间向反向运动。倾斜盘上的瞬时速度可随位置周期变化，这个判据没有要求匀速。
- 撤力后再算180秒，球净反向 **{-post['net_ball_revolutions']:.5f} 圈**。这里计数的是球实际绕圈，不是仅计算磁铁盘转了多少圈。
- 最后10秒平均角速度 {post['tail_mean_angular_speed_rad_s']:.5f} rad/s，平均轨道半径 {post['tail_radius_mean_mm']:.5f} mm，反向时间占比 {100*post['tail_reverse_time_fraction']:.3f}%。
- 后段每圈约1.94秒。不同持续推动强度进入后段后均趋向相近轨道，说明这组计算确有撤力后自持的反向运行分支。
- **接触滑移仍明显**：滑移速率/球心速率的中位数 {100*post['tail_slip_ratio_median']:.3f}%，95%分位 {100*post['tail_slip_ratio_p95']:.3f}%。严格纯滚动分支仅占 {100*post['tail_pure_rolling_fraction']:.3f}%。不能把这条轨迹称为纯滚动。
- 初始释放至捕获的明显过冲/回摆也仍在。这次改变的是吹气协议，没有靠持续外力掩盖最初释放的模型不符。

0.6 mN即使持续30秒仍未解除捕获；0.7 mN及以上有进入自持反向运行的情况。仅就这组参数，持续时间和推动强度都重要；这不是实测临界风力。

## 可直接塞入原有无外力构建的初态

`reverse_running_initial_state.json`保存了撤力那一刻的全部15维状态、q、姿态、自旋、磁铁盘绝对相位，以及接触分支。它引用上轮已经归档的普通无外力构建；复跑时不需要再吹气，也不要把q或自旋归零。

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_prepared_release.py --case src2d/results/sustained-push/reverse_running_initial_state.json --seconds 180 --animate
```

完整复跑“释放→近停→持续吹到稳态→撤力”的协议：

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_sustained_push.py --animate
```

完整阶段协议、探索力值、撤力判据都在 `protocol.json`；原生产构建没有覆盖。NPZ/CSV保存真实外力开关，并保留原始轨迹，不只截取好看的稳态。

`full_release_trajectory.npz/csv/png`从零速释放开始，包含最初回摆、捕获、持续推动和撤力后的全部阶段。`protocol_trajectory.npz/csv/png`从开始吹气计时，方便观察撤力前后。JSON中的state、q、自旋与接触分支都使用求解器的无量纲单位；CSV的位置与时间用mm、秒。

## 核对与范围

不施加外力时，新加入持续推动接口的构建与上轮普通无外力轨迹，全部状态最大差为 {validation['max_state_difference']}；时间也一致。因此没有把本体阻尼、摩擦或磁参数偷偷改成新的模型。

| 持续力/mN | 步长 | 实际推动秒数 | 推动中逐圈稳态 | 撤力后180秒球净圈数 |
| --- | --- | --- | --- | --- |
{overview}

步长减半时0.7mN的撤力时刻一致，撤力后仍约92.68圈。这里的核对支持持续推动后的自持运动稳健存在；不验证真实风力、材料JA标定或纯滚动。

较强的1.5mN在自动持续推动测试中出现积分失败，该组没有作为成功解。固定时长扫描中能先撤力后进入运行的情况，也不能代替“推动到逐圈稳态才撤力”的协议。
'''
    (DEST/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),push_seconds=record['actual_push_seconds'],post_metrics=post),indent=2))


if __name__=='__main__':main()
