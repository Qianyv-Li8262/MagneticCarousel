"""Report high-C behavior search and independently verify the frozen candidates."""
import json,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import high_c_wait_study as h
import fit_video56c6_rolling as dynamic

OUT=h.OUT;ROLL=OUT/'rolling';GALLERY=h.gallery.f.ROOT/'runs/overshoot-gallery-20261017'
f=h.gallery.f

def characterize(r,folder):
    a=pd.read_csv(folder/(r['key']+'.csv'));t=a.time_s.to_numpy();n=round(1/np.median(np.diff(t)));kernel=np.ones(n)/n
    sm=lambda key:np.convolve(a[key],kernel,mode='valid')
    ts=np.convolve(t,kernel,mode='valid');v=sm('speed_mm_s');wz=sm('wz_rad_s');rad=sm('radius_mm');tail=t>=220
    quiet=(ts>=4)&(ts<=13)
    final=float(a.speed_mm_s[tail].mean());finalr=float(a.radius_mm[tail].mean())
    def cross(frac):
        ix=np.flatnonzero((ts>=3)&(v>=frac*final));m=round(2/np.median(np.diff(ts)))
        for j in ix:
            if j+m<len(v) and np.min(v[j:j+m])>=frac*final:return float(ts[j])
        return None
    t10,t20,t80=map(cross,[.1,.2,.8])
    before=(ts>=3)&(ts<=t20) if t20 is not None else ts<0
    metrics=dict(quiet_mean_mm_s=float(v[quiet].mean()),quiet_p95_mm_s=float(np.percentile(v[quiet],95)),
        t10_s=t10,t20_s=t20,t80_s=t80,rise10_80_s=t80-t10 if t10 is not None and t80 is not None else None,
        rise20_80_s=t80-t20 if t20 is not None and t80 is not None else None,
        overshoot_mm=float(max(0,rad.max()-finalr)),tail_radius_mm=finalr,tail_speed_mm_s=final,
        tail_wz_rad_s=float(a.wz_rad_s[tail].mean()),tail_signed_vt_mm_s=float(a.vt_mm_s[tail].mean()),
        tail_period_s=float(2*np.pi/abs(np.polyfit(t[tail],np.unwrap(a.phi_rad)[tail],1)[0])),
        spin_at_3s_rad_s=float(np.interp(3,ts,wz)),spin_at_t20_rad_s=float(np.interp(t20,ts,wz)) if t20 is not None else None,
        pre_onset_spin_peak_rad_s=float(np.max(wz[before])) if before.any() else None,
        rolling_fraction=float(1-a.sliding.mean()),end_s=float(t[-1]))
    return a,ts,v,rad,wz,metrics

def half_check(record,folder,rolling,name):
    cfg=dict(record['config']);cfg['dt']=.00025;cfg['seconds']=240.
    path=OUT/(name+'_half.csv')
    args=[str(f.SOURCE/'field_lut_fp64.bin'),str(path)]+[str(cfg[k]) for k in f.KEYS]+[str(f.eddy_table(cfg['mu']))]
    if rolling:
        exe=dynamic.SOURCE/'fit_rolling.exe';rr=record['rolling'];args += [str(rr['mu_low']),str(rr['mu_high']),str(rr['omega_c_code'])]
    else:exe=f.SOURCE/'fit_video56c6.exe'
    result=subprocess.run([str(exe),*args],check=True,capture_output=True,text=True)
    a=pd.read_csv(folder/(record['key']+'.csv'));b=pd.read_csv(path)
    w=a.time_s.to_numpy()<=b.time_s.iloc[-1]
    r=dict(status=result.stdout.strip(),end_s=float(b.time_s.iloc[-1]),rolling_fraction=float(1-b.sliding.mean()))
    for col in ['speed_mm_s','radius_mm','wz_rad_s']:
        diff=abs(a[col].to_numpy()[w]-np.interp(a.time_s.to_numpy()[w],b.time_s,b[col]))
        r['max_'+col+'_difference']=float(diff.max())
    return r

if __name__=='__main__':
    material=min([json.loads(p.read_text()) for p in OUT.glob('best_*.json')],key=lambda r:r['score'])
    joint=min([json.loads(p.read_text()) for p in ROLL.glob('best_*.json')],key=lambda r:r['score'])
    baseline=next(r for r in json.loads((GALLERY/'selected.json').read_text()) if r['name']=='C0.44_K02')
    rows=[('原高C小过冲分支',baseline,GALLERY,'#94a3b8'),('仅联合磁参数，滚阻固定',material,OUT,'#2563eb'),('高C＋联合滚阻候选',joint,ROLL,'#ea790c')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks=list(pool.map(lambda z:half_check(*z),[(material,OUT,False,'material'),(joint,ROLL,True,'joint')]))
    report=dict(working_targets=dict(quiet_window_after_release_s=[4,13],quiet_mean_mm_s=8,quiet_p95_mm_s=18,
        onset20_percent_window_s=[6,25],rise20_80_max_s=7,overshoot_mm=.8,tail_speed_mm_s=300),
        note='Working comparison targets, not experimental measurements. No experimental curve fitting or time/space scaling. All initial speeds and memories zero; no premagnetization or blowing.',candidates={},validation=dict(material=checks[0],joint=checks[1]))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    fig.suptitle('高C下联合调参：同时观察等待、启动、过冲和自旋',fontsize=16)
    inset=axs[0,0].inset_axes([.10,.53,.40,.39])
    for name,r,folder,color in rows:
        a,t,v,rad,wz,m=characterize(r,folder)
        report['candidates'][name]=dict(config=r['config'],rolling=r.get('rolling',dict(mu_low=.0049,mu_high=.0049)),metrics=m,key=r['key'],directory=str(folder))
        axs[0,0].plot(t,v,color=color,label=name);inset.plot(t,v,color=color)
        axs[0,1].plot(t,rad-m['tail_radius_mm'],color=color,label=name)
        axs[1,0].plot(t,wz,color=color,label=name)
        vv=np.linspace(0,.4,401);rr=r.get('rolling',dict(mu_low=.0049,mu_high=.0049,vc_m_s=.26))
        mm=rr['mu_high']+(rr['mu_low']-rr['mu_high'])*np.exp(-(vv/rr.get('vc_m_s',.26))**2)
        axs[1,1].plot(vv,mm,color=color,label=name)
    axs[0,0].set(xlim=(0,65),xlabel='释放后时间 (s)',ylabel='1秒平均质心速率 (mm/s)',title='启动过程；小图放大初始低速段')
    inset.set(xlim=(3,16),ylim=(0,35));inset.grid(alpha=.2);inset.tick_params(labelsize=8)
    axs[0,1].axhline(0,color='gray',lw=.7);axs[0,1].set(xlim=(0,120),xlabel='释放后时间 (s)',ylabel='r − 最后20秒平均半径 (mm)',title='径向过冲：正值表示超过最终平均轨道')
    axs[1,0].set(xlim=(0,120),xlabel='释放后时间 (s)',ylabel='1秒平均竖直自旋 (rad/s)',title='检查低速期间是否仍在积累自旋')
    axs[1,1].set(xlabel='真实纯滚速度 (m/s)',ylabel='有效滚阻系数',title='实际使用的滚阻曲线；无时间或空间放缩')
    for ax in axs.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    axs[0,0].legend(fontsize=8,loc='lower right')
    fig.savefig(OUT/'comparison.png',dpi=155)
    (OUT/'fit_report.json').write_text(json.dumps(report,indent=2))
    (OUT/'selected_material.json').write_text(json.dumps(material,indent=2));(OUT/'selected_joint.json').write_text(json.dumps(joint,indent=2))
    profiles=json.loads((OUT/'profiles.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for c,color in [(.44,'#2563eb'),(.46,'#059669')]:
        z=sorted([q for q in profiles if q['tag'].startswith('sigma_C'+str(c)+'_')],key=lambda q:q['x'][0])
        x=[q['x'][0]*1e7 for q in z]
        axs[0,0].plot(x,[q['record']['quiet_mean_mm_s'] for q in z],'o-',color=color,label=f'C={c}')
        axs[0,1].plot(x,[q['record']['tail_speed_mm_s'] for q in z],'o-',color=color,label=f'C={c}')
    for sigma,color in [(.6,'#2563eb'),(1.2,'#ea790c')]:
        z=[q for q in profiles if q['tag'].startswith('K_s'+str(sigma)+'_')]
        z += [q for q in profiles if q['tag']==f'sigma_C0.44_{sigma}']
        z.sort(key=lambda q:q['x'][1]);x=[q['x'][1] for q in z]
        axs[1,0].plot(x,[q['record']['quiet_mean_mm_s'] for q in z],'o-',color=color,label=f'σ={sigma*10:g} 百万 S/m')
        axs[1,1].plot(x,[q['record']['tail_speed_mm_s'] for q in z],'o-',color=color,label=f'σ={sigma*10:g} 百万 S/m')
    for ax in axs[0]:ax.set_xlabel('电导率 σ (S/m)')
    for ax in axs[1]:ax.set_xlabel('K / 参考K');ax.set_xscale('log')
    axs[0,0].set_ylabel('等待窗口平均速率 (mm/s)');axs[0,1].set_ylabel('最终速率 (mm/s)')
    axs[1,0].set_ylabel('等待窗口平均速率 (mm/s)');axs[1,1].set_ylabel('最终速率 (mm/s)')
    for ax in axs.flat:ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.suptitle('单参数剖面：改变一个参数也可能切换运动分支',fontsize=16)
    fig.savefig(OUT/'parameter_profiles.png',dpi=150)
    lines=['# 高C下：等待、自旋、启动和径向过冲的联合调参','',
        '本轮不拟合实验逐点曲线，只考察行为兼顾。全部h=32 mm、正常磁矩4.222661 A m²、200 rpm、球半径6 mm、初始半径67.4 mm。初始质心速度、三轴角速度、磁滞记忆、涡流记忆全零；无预磁化、无吹气；盘初相位0.58 rad。没有时间或空间缩放，生产构建未修改。','',
        '## 比较指标','',
        '等待窗口取释放后4–13 s；启动时间用质心速率持续至少2 s超过末态速率20%的时刻表征，另保存10%阈值；加速宽度取20%到80%。这些是预先设定的工作指标，不是新实测量。过冲为一秒平均半径最大值减220–240 s平均半径。','',
        '| 候选 | 等待均速 mm/s | 20%时刻 s | 20→80% s | 过冲 mm | 末态速率 mm/s | 末态周期 s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in report['candidates'].items():
        m=c['metrics'];lines.append(f"| {name} | {m['quiet_mean_mm_s']:.4f} | {m['t20_s']:.4f} | {m['rise20_80_s']:.4f} | {m['overshoot_mm']:.4f} | {m['tail_speed_mm_s']:.4f} | {m['tail_period_s']:.6f} |")
    lines+=['','## 完整参数','', '| 候选 | σ S/m | μr | Alpha_JA | C_JA | K | 滚阻低速极限 | 滚阻高速极限 | vc m/s |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in report['candidates'].items():
        p=c['config'];rr=c['rolling'];lines.append(f"| {name} | {p['sigma']:.8g} | {p['mu']:.8g} | {p['alpha']:.10g} | {p['c']:.9g} | {p['K']:.10g} | {rr['mu_low']:.9g} | {rr['mu_high']:.9g} | {rr.get('vc_m_s','不适用')} |")
    lines+=['','## 机制判断与边界','',
        '参数确实存在不同作用侧重，但作用重叠，而且会切换运动分支。单独降低电导率能减小早期质心速度，同时也可能使系统长期停在低速状态。K、Alpha_JA和Mu_r都能改变能否重新启动，并影响加速和过冲。提高低速滚阻能恢复较平坦的等待段，降低高速滚阻能缩短后续加速，但也会增加径向过冲；继续提高C可对冲其中一部分。','',
        '本轮先固定滚阻做36组单参数剖面，以两条不同机制的种子做有界局部优化，再沿近似保持sigma*mu或alpha/K的方向做局部剖面；之后使用此前获授权的高斯滚阻自由度，过渡速度尺度限制0.218–0.300 m/s。没有随机参数云，也没有调整初始自旋或磁记忆来制造等待。','',
        '工作目标是等待均速≤8 mm/s、20→80%宽度≤7 s、过冲≤0.8 mm、末态速率接近300 mm/s，同时限制过长等待。最终残差应按表直接判断，不因总分下降就称全部满足。未证明全局最优或模型不可能。','',
        '高斯过渡在实测速度范围内已有尾部，调出的降阻曲线属于探索性假设，不是新的滚阻测量；未来应与补测结果一起检验。磁性材料参数也属于条件拟合值，不能当作独立物性测量。','',
        '## 验证','']
    for name,c in report['validation'].items():lines.append(f"- {name}：{c['status']}，dt减半最大速度差{c['max_speed_mm_s_difference']:.6f} mm/s，最大半径差{c['max_radius_mm_difference']:.8f} mm，纯滚采样比例{c['rolling_fraction']:.6f}。")
    lines+=['','fit_report.json保存全部输入、指标、原始CSV位置及验证；selected_material.json和selected_joint.json冻结结果。comparison.png显示轨迹，parameter_profiles.png显示单参数剖面。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))
