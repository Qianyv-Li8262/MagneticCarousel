"""Controlled zero-state overshoot gallery, without an experimental objective."""
import json, hashlib, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_video56c6 as f

OUT=f.ROOT/'runs/overshoot-gallery-20261017';OUT.mkdir(exist_ok=True)

def run(case):
    name,changes=case
    cfg=dict(f.BASE,r=67.4,seconds=240.,vt=0.,vr=0.,wz=0.,qz=0.,hold=0.)
    cfg.update(changes)
    key=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
    path=OUT/(key+'.csv');meta=OUT/(key+'.json')
    if meta.exists():return json.loads(meta.read_text())
    cmd=[str(f.SOURCE/'fit_video56c6.exe'),str(f.SOURCE/'field_lut_fp64.bin'),str(path)]+[str(cfg[k]) for k in f.KEYS]+[str(f.eddy_table(cfg['mu']))]
    result=subprocess.run(cmd,check=True,capture_output=True,text=True)
    a=pd.read_csv(path);t=a.time_s.to_numpy();n=round(1/np.median(np.diff(t)));kernel=np.ones(n)/n
    smooth=np.convolve(a.radius_mm,kernel,mode='valid');ts=np.convolve(t,kernel,mode='valid');tail=t>=220
    tailr=float(a.radius_mm[tail].mean()) if tail.any() else float('nan')
    r=dict(name=name,key=key,config=cfg,status=result.stdout.strip(),end_s=float(t[-1]),
        tail_radius_mm=tailr,overshoot_mm=float(max(0,smooth.max()-tailr)),peak_radius_mm=float(smooth.max()),
        peak_time_s=float(ts[np.argmax(smooth)]),tail_speed_mm_s=float(a.speed_mm_s[tail].mean()) if tail.any() else None,
        tail_radius_std_mm=float(a.radius_mm[tail].std()) if tail.any() else None,
        tail_period_s=float(2*np.pi/abs(np.polyfit(t[tail],np.unwrap(a.phi_rad)[tail],1)[0])) if tail.sum()>10 else None,
        tail_wz_rad_s=float(a.wz_rad_s[tail].mean()) if tail.any() else None,rolling_fraction=float(1-a.sliding.mean()))
    r['accepted']=bool(t[-1]>239.9 and 'status=0' in result.stdout and r['tail_speed_mm_s']>100 and r['tail_radius_std_mm']<.5)
    meta.write_text(json.dumps(r,indent=2));print(name,round(r['overshoot_mm'],3),r['status'],r['tail_speed_mm_s'],flush=True)
    return r

def render(rows):
    # A slower but genuinely circulating steady branch is valid here: no video fit.
    for r in rows:r['accepted']=bool(r['end_s']>239.9 and 'status=0' in r['status'] and r['tail_speed_mm_s'] is not None and r['tail_speed_mm_s']>100 and r['tail_radius_std_mm']<.5)
    valid=[r for r in rows if r['accepted']]
    # Prefer a matched one-parameter sequence: C=0.30, 0.44 and 0.46,
    # retaining a high-speed near-zero-overshoot branch in the comparison.
    byname={r['name']:r for r in valid}
    small=[byname[n] for n in ['C0.44_K02','C0.46_K02'] if n in byname]
    if len(small)<2:small=sorted(valid,key=lambda r:r['overshoot_mm'])[:2]
    large=sorted(valid,key=lambda r:r['overshoot_mm'],reverse=True)[:2]
    selected=large+small
    if len({r['key'] for r in selected})!=4:raise RuntimeError('Need four distinct stable candidates')
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    fig.suptitle('只比较模拟：相同零初态，明显过冲与小过冲',fontsize=17)
    colors=['#dc2626','#ea8a15','#2563eb','#059669']
    for i,(r,color) in enumerate(zip(selected,colors)):
        a=pd.read_csv(OUT/(r['key']+'.csv'));n=round(1/np.median(np.diff(a.time_s)));k=np.ones(n)/n
        sm=lambda x:np.convolve(x,k,mode='valid');t=sm(a.time_s)
        label=f"{'ABCD'[i]}：过冲 {r['overshoot_mm']:.2f} mm"
        axs[0,0].plot(t,sm(a.radius_mm),color=color,label=label)
        axs[0,1].plot(t,sm(a.radius_mm)-r['tail_radius_mm'],color=color,label=label)
        axs[1,0].plot(t,sm(a.speed_mm_s),color=color,label=label)
        axs[1,1].plot(t,sm(a.wz_rad_s),color=color,label=label)
    for ax in axs.flat:ax.grid(alpha=.25);ax.set_xlim(0,90);ax.set_xlabel('释放后时间 (s)');ax.legend(fontsize=9)
    axs[0,0].set(ylabel='轨道半径 (mm)',title='绝对半径：不做时间或空间缩放')
    axs[0,1].axhline(0,color='gray',lw=.8);axs[0,1].set(ylabel='r − 最后20秒平均半径 (mm)',title='相对最终轨道：超过零线即径向过冲')
    axs[1,0].set(ylabel='质心速率 (mm/s)',title='同时看启动过程，排除“尚未启动”的假小过冲')
    axs[1,1].set(ylabel='竖直自旋角速度 (rad/s)',title='自旋演化：初始全部为零')
    fig.savefig(OUT/'startup_zoom.png',dpi=155)
    for ax in axs.flat:ax.set_xlim(0,240)
    fig.savefig(OUT/'comparison.png',dpi=155)
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for i,(r,color) in enumerate(zip(selected,colors)):
        a=pd.read_csv(OUT/(r['key']+'.csv'));n=round(1/np.median(np.diff(a.time_s)));k=np.ones(n)/n;t=np.convolve(a.time_s,k,mode='valid')
        for ax,col in zip(axs,['radius_mm','speed_mm_s']):ax.plot(t,np.convolve(a[col],k,mode='valid'),color=color,label='ABCD'[i])
    axs[0].set(ylabel='轨道半径 (mm)',title='完整240 s演化复核');axs[1].set(ylabel='质心速率 (mm/s)',xlabel='释放后时间 (s)')
    for ax in axs:ax.grid(alpha=.25);ax.legend()
    fig.savefig(OUT/'long_run.png',dpi=150)
    family=sorted([r for r in rows if r['name'].endswith('_K02')],key=lambda r:r['config']['c'])
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    cc=[r['config']['c'] for r in family]
    axs[0].plot(cc,[r['overshoot_mm'] for r in family],'o-',color='#2563eb')
    axs[1].plot(cc,[r['tail_speed_mm_s'] for r in family],'o-',color='#ea8a15')
    axs[0].set(xlabel='C_JA',ylabel='半径过冲 (mm)',title='只调 C_JA：过冲大小')
    axs[1].set(xlabel='C_JA',ylabel='最后20秒平均速率 (mm/s)',title='只调 C_JA：最终运行速度')
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('固定 σ=1.2×10⁷ S/m、μr=605.897、K=0.00107519')
    fig.savefig(OUT/'C_sensitivity.png',dpi=155)
    lines=['# 相同零初态的过冲比较','', '所有轨迹 h=32 mm，正常磁矩4.222661 A m²，200 rpm，球半径6 mm，初始半径67.4 mm，初始速度/角速度/磁滞与涡流记忆全零，无预磁化、无吹气；滚阻固定0.0049。绝对SI单位，时间从释放起算，不作空间/时间缩放。不是对实验的拟合。','', '| 曲线 | σ (S/m) | μr | C_JA | K | 过冲 (mm) | 最终半径 (mm) | 最终速率 (mm/s) | 周期 (s) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,r in enumerate(selected):
        c=r['config'];lines.append(f"| {'ABCD'[i]} | {c['sigma']:.7g} | {c['mu']:.6f} | {c['c']:.7f} | {c['K']:.9f} | {r['overshoot_mm']:.4f} | {r['tail_radius_mm']:.4f} | {r['tail_speed_mm_s']:.4f} | {r['tail_period_s']:.6f} |")
    lines+=['',f"共同 Alpha_JA={f.BASE['alpha']:.16g}，盘初相位0.58 rad，dt=0.0005（无量纲）。过冲定义为一秒滑动平均半径最大值减去220–240 s平均半径；用完整240 s计算，不把72 s暂态当最终稳态。",'', 'selected.json保存全部参数，sweep.json保存全部已试候选（包括失败与未达稳定筛选者）。曲线可能同时改变不同材料参数，只有同一参数族中的变化才可用于单参数因果比较。生产构建未修改。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(selected,indent=2))

if __name__=='__main__':
    baseK=f.BASE['K']
    big=json.loads((f.ROOT/'runs/video56c6-rest-release-20261017/dd0d31720fc0215c.json').read_text())['config']
    cases=[('old_large',{k:big[k] for k in ['sigma','mu','c','K']}),('reference',dict(sigma=12e6))]
    cases += [(f'C{c}_K02',dict(sigma=12e6,c=c,K=.2*baseK)) for c in [.3,.4,.5,.6,.7,.8]]
    cases += [(f'C04_K{k}',dict(sigma=12e6,c=.4,K=k*baseK)) for k in [.05,.1,.5,1.]]
    cases += [(f'C{c}_K02',dict(sigma=12e6,c=c,K=.2*baseK)) for c in [.42,.44,.46,.48]]
    cases += [(f'C04_K{k}',dict(sigma=12e6,c=.4,K=k*baseK)) for k in [.3,.35,.4]]
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,cases))
    render(rows);(OUT/'sweep.json').write_text(json.dumps(rows,indent=2))
